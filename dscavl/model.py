"""端到端 DSCA-VL 模块：视觉/文本编码 → DSAM（patch 级分解）→ CGRM → 策略头。

对外保持 ``stage1`` / ``stage2`` / ``infer`` 三种 ``forward`` 模式；patch 级
统计经 ``_build_patch_bridge`` 聚合成 frame 级辅助量供 CGRM/Policy 使用。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .arch_registry import apply_arch_variant
from .cgrm import CGRM
from .config import DSCAVLConfig
from .dsam import DSAM
from .encoders import FrozenVisualEncoder, QueryTextEncoder
from .policy import FramePolicyHead
from .variant_modules import ModalitySeparationAdapter, adaptive_budget, bandit_update, event_anchor_select, temporal_nms_1d
from .variant_modules.features import temporal_disparity_scores


class DSCAVL(nn.Module):
    """组合冻结视觉适配器、查询编码、DSAM、因果图模块与帧/事件策略的顶层模型。"""

    def __init__(
        self,
        cfg: DSCAVLConfig,
        vis_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        llm: Optional[nn.Module] = None,
    ):
        """构建各子模块；``vis_encoder`` / ``text_encoder`` 可注入真实 SigLIP/Qwen 实现。"""
        super().__init__()
        self.cfg = apply_arch_variant(cfg, cfg.arch_variant)
        cfg = self.cfg

        self.vis_encoder = vis_encoder or FrozenVisualEncoder(
            dim=cfg.dim,
            patch_tokens_per_frame=cfg.patch_tokens_per_frame,
        )
        self.text_encoder = text_encoder or QueryTextEncoder(dim=cfg.dim)
        self.dsam = DSAM(
            dim=cfg.dim,
            bg_prototypes=cfg.bg_prototypes,
            tau=cfg.tau,
            alpha_init=cfg.alpha_init,
            use_isoclip_debias=cfg.use_isoclip_debias,
            use_riemann_slerp=cfg.use_riemann_slerp,
            use_soft_mask=cfg.use_soft_mask,
            isoclip_debias_strength=cfg.isoclip_debias_strength,
            riemann_slerp_strength=cfg.riemann_slerp_strength,
            soft_mask_strength=cfg.soft_mask_strength,
        )
        self.cgrm = CGRM(
            dim=cfg.dim,
            window=cfg.temp_window,
            stride=cfg.temp_stride,
            sigma_temp=cfg.sigma_temp,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
            causal_decay_tau=cfg.causal_decay_tau,
            bridge_weight=cfg.bridge_weight,
            use_f_bg=cfg.cgrm_use_f_bg,
            mix_temperature=cfg.cgrm_mix_temperature,
            use_sbp=cfg.use_sbp,
            sbp_negative_threshold=cfg.sbp_negative_threshold,
            sbp_negative_weight=cfg.sbp_negative_weight,
        )
        self.policy = FramePolicyHead(
            dim=cfg.dim,
            gamma=cfg.gamma,
            intra_top_ratio=cfg.intra_top_ratio,
        )
        self.ums_adapter = (
            ModalitySeparationAdapter(dim=cfg.dim, bottleneck_dim=cfg.ums_bottleneck_dim)
            if cfg.use_ums_mae
            else None
        )
        self.pretrained_feature_adapter = (
            nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, cfg.dim))
            if cfg.use_pretrained_feature_adapter
            else None
        )

        self.llm = llm

    # ------------------------------------------------------------------
    def _build_patch_bridge(
        self,
        ds_out: Dict[str, torch.Tensor],
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """从 DSAM 的 patch 级输出推导 frame 级 patch 统计（分散度、集中度等），供 CGRM/Policy。

        若缺少 patch 权重或 saliency，返回空 dict，下游按无辅助特征处理。
        """
        patch_weights = ds_out.get("patch_weights")
        s_patch = ds_out.get("S_patch")
        recon_error_patch = ds_out.get("recon_error_patch")
        prototype_usage = ds_out.get("prototype_usage")
        if patch_weights is None or s_patch is None or recon_error_patch is None:
            return {}

        eps = 1e-6
        if patch_weights.ndim != 3:
            return {}

        b, t, p = patch_weights.shape
        # 数值清理，避免 log/softmax 在 NaN 上传播
        patch_weights = torch.nan_to_num(patch_weights, nan=0.0, posinf=0.0, neginf=0.0)
        s_patch = torch.nan_to_num(s_patch, nan=0.0, posinf=0.0, neginf=0.0)
        recon_error_patch = torch.nan_to_num(
            recon_error_patch, nan=0.0, posinf=0.0, neginf=0.0
        )

        if p <= 1:
            patch_dispersion = torch.zeros((b, t), device=patch_weights.device, dtype=patch_weights.dtype)
        else:
            # 归一化熵：衡量 patch 权重是否均匀（高则注意力更分散）
            entropy = -(patch_weights.clamp_min(eps) * patch_weights.clamp_min(eps).log()).sum(dim=-1)
            patch_dispersion = entropy / patch_weights.new_tensor(float(p)).log().clamp_min(eps)
        patch_concentration = patch_weights.max(dim=-1).values
        patch_saliency_peak = s_patch.max(dim=-1).values

        recon_center = ds_out.get("recon_error")
        if recon_center is None:
            # 无 frame 级误差时用加权重构误差作为中心
            recon_center = (patch_weights * recon_error_patch).sum(dim=-1)
        # patch 上重构误差相对 frame 均值的加权标准差（ spread ）
        recon_spread = (
            patch_weights * (recon_error_patch - recon_center.unsqueeze(-1)).square()
        ).sum(dim=-1).sqrt()

        patch_proto_entropy = torch.zeros_like(patch_dispersion)
        if prototype_usage is not None and prototype_usage.ndim == 4:
            k = prototype_usage.shape[-1]
            if k > 1:
                # 将 patch→原型分配按 patch 权重聚到 frame，再算原型使用熵
                frame_proto_usage = (
                    patch_weights.unsqueeze(-1) * torch.nan_to_num(prototype_usage, nan=0.0)
                ).sum(dim=2)
                proto_entropy = -(
                    frame_proto_usage.clamp_min(eps) * frame_proto_usage.clamp_min(eps).log()
                ).sum(dim=-1)
                patch_proto_entropy = proto_entropy / frame_proto_usage.new_tensor(float(k)).log().clamp_min(eps)

        frame_patch_stats = torch.stack(
            [
                patch_dispersion,
                patch_concentration,
                patch_saliency_peak,
                recon_spread,
                patch_proto_entropy,
            ],
            dim=-1,
        )

        if frame_mask is not None:
            # padding 帧的统计置零，避免污染事件池化
            valid = frame_mask.to(frame_patch_stats.dtype).unsqueeze(-1)
            frame_patch_stats = frame_patch_stats * valid
            patch_dispersion = patch_dispersion * frame_mask.to(patch_dispersion.dtype)
            patch_concentration = patch_concentration * frame_mask.to(patch_concentration.dtype)
            patch_saliency_peak = patch_saliency_peak * frame_mask.to(patch_saliency_peak.dtype)
            recon_spread = recon_spread * frame_mask.to(recon_spread.dtype)
            patch_proto_entropy = patch_proto_entropy * frame_mask.to(patch_proto_entropy.dtype)

        return {
            "frame_patch_stats": frame_patch_stats,
            "patch_dispersion": patch_dispersion,
            "patch_concentration": patch_concentration,
            "patch_saliency_peak": patch_saliency_peak,
            "patch_recon_spread": recon_spread,
            "patch_proto_entropy": patch_proto_entropy,
        }

    # ------------------------------------------------------------------
    def _select_actions_infer(
        self,
        probs: torch.Tensor,
        membership: Optional[torch.Tensor] = None,
        event_probs: Optional[torch.Tensor] = None,
        query_features: Optional[torch.Tensor] = None,
        frame_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """推理阶段：按配置在事件多样性与帧数上下限之间选取二值动作 ``[B,T]``。

        有事件信息时先选 top 事件再在事件内取帧，最后用全局 top 填满预算并保证不低于 ``keep_ratio_low``。
        """
        b, t = probs.shape
        k_low = max(1, int(t * self.cfg.keep_ratio_low))
        k_high = max(k_low, int(t * self.cfg.keep_ratio_high))

        has_events = membership is not None and event_probs is not None
        actions = torch.zeros_like(probs)

        for i in range(b):
            k = min(k_high, t)
            if self.cfg.use_adaptive_budget and query_features is not None:
                budget = adaptive_budget(
                    query_features[i : i + 1],
                    n_frames=t,
                    low=self.cfg.adaptive_budget_low,
                    high=min(self.cfg.adaptive_budget_high, k_high),
                )
                k = int(budget[0].item())

            if has_events:
                m = membership.shape[2]
                min_events = max(2, int(m * self.cfg.min_event_ratio))
                max_per_event = max(1, int(self.cfg.max_per_event))

                # Select top events by event_probs.
                n_events = min(min_events, m)
                top_events = torch.topk(event_probs[i], k=n_events).indices

                selected_count = 0
                for eidx in top_events:
                    frame_idx = (membership[i, :, eidx] > 1e-4).nonzero(
                        as_tuple=False
                    ).flatten()
                    if frame_idx.numel() == 0:
                        continue
                    local_probs = probs[i, frame_idx]
                    n_pick = min(max_per_event, frame_idx.numel(), k - selected_count)
                    if n_pick <= 0:
                        break
                    _, top_idx = local_probs.topk(n_pick)
                    actions[i, frame_idx[top_idx]] = 1.0
                    selected_count += n_pick

                # Fill remaining budget with global top-k from unselected frames.
                remaining = k - int(actions[i].sum().item())
                if remaining > 0:
                    available = probs[i] * (1.0 - actions[i])
                    if available.sum() > 0:
                        extra = torch.topk(available, k=min(remaining, t)).indices
                        actions[i, extra] = 1.0
            else:
                idx = torch.topk(probs[i], k=k, dim=-1).indices
                actions[i, idx] = 1.0

            if self.cfg.use_efs_selection and membership is not None and frame_features is not None:
                event_ids = membership[i].argmax(dim=-1)
                anchors = event_anchor_select(
                    probs[i],
                    event_ids,
                    frame_features[i],
                    budget=k,
                    diversity_weight=self.cfg.efs_diversity_weight,
                )
                actions[i].zero_()
                actions[i, anchors] = 1.0

            if actions[i].sum() < k_low:
                need = int(k_low - actions[i].sum().item())
                extra = torch.topk(
                    probs[i] * (1 - actions[i]), k=need, dim=-1
                ).indices
                actions[i, extra] = 1.0

            if self.cfg.use_st_nms:
                selected = (actions[i] > 0.5).nonzero(as_tuple=False).flatten()
                if selected.numel() > 0:
                    kept = temporal_nms_1d(
                        selected,
                        probs[i, selected],
                        min_gap=self.cfg.st_nms_min_gap,
                        max_keep=k,
                    )
                    actions[i].zero_()
                    actions[i, kept] = 1.0

        return actions

    def _variant_diagnostics(
        self,
        ds: Dict[str, torch.Tensor],
        cg: Dict[str, torch.Tensor],
        ums: Optional[Dict[str, torch.Tensor]],
        everest_scores: Optional[torch.Tensor],
        f_q: torch.Tensor,
    ) -> Dict[str, float]:
        """Collect scalar diagnostics used to judge whether R17 issues moved."""
        diag: Dict[str, float] = {}
        if ums is not None:
            diag["ums_mdi_mean"] = float(ums["mdi"].detach().mean().cpu())
            diag["ums_orth_loss"] = float(ums["orth_loss"].detach().cpu())
            diag["ums_mae_gate_mean"] = float(ums["mae_gate"].detach().mean().cpu())
        if everest_scores is not None:
            diag["everest_score_mean"] = float(everest_scores.detach().mean().cpu())
        if self.cfg.use_adaptive_budget:
            budget = adaptive_budget(
                f_q.detach(),
                n_frames=ds["F_fg"].shape[1],
                low=self.cfg.adaptive_budget_low,
                high=self.cfg.adaptive_budget_high,
            )
            diag["adaptive_budget_mean"] = float(budget.float().mean().cpu())
        if "dir_pref" in cg:
            diag["dir_pref_mean"] = float(cg["dir_pref"].detach().mean().cpu())
        if "mix_logits" in cg:
            diag["dir_pref_raw_mean"] = float(cg["mix_logits"].detach().mean().cpu())
        if "mix_lambdas" in cg:
            lambdas = cg["mix_lambdas"].detach()
            diag["lambda_sem_mean"] = float(lambdas[:, 0].mean().cpu())
            diag["lambda_causal_mean"] = float(lambdas[:, 1].mean().cpu())
        if self.cfg.use_hornet_policy:
            diag["hornet_policy_weight"] = float(self.cfg.hornet_policy_weight)
        if self.cfg.use_efs_selection and "membership" in cg and "S_graph" in cg:
            event_ids = cg["membership"][0].argmax(dim=-1)
            selected = event_anchor_select(
                cg["S_graph"][0].detach(),
                event_ids.detach(),
                ds["F_fg"][0].detach(),
                budget=max(1, min(ds["F_fg"].shape[1], int(ds["F_fg"].shape[1] * self.cfg.keep_ratio_high))),
                diversity_weight=self.cfg.efs_diversity_weight,
            )
            diag["efs_selected_count"] = float(selected.numel())
        if self.cfg.use_dynamic_logit_threshold and "S_graph" in cg:
            s = cg["S_graph"].detach()
            diag["dynamic_threshold_mean"] = float((s.mean(dim=-1) + s.std(dim=-1, unbiased=False)).mean().cpu())
        if self.cfg.use_hroute:
            complexity = f_q.detach().float().std(dim=-1, unbiased=False)
            diag["hroute_complexity_mean"] = float(complexity.mean().cpu())
        if self.cfg.use_tta_mab and "S_graph" in cg:
            prior = torch.softmax(cg["S_graph"][0].detach(), dim=-1)
            top = torch.topk(prior, k=min(2, prior.numel())).indices
            updated = bandit_update(prior, top, reward=1.0, lr=self.cfg.tta_mab_lr)
            entropy = -(updated.clamp_min(1e-8) * updated.clamp_min(1e-8).log()).sum()
            diag["tta_mab_entropy"] = float(entropy.cpu())
        return diag

    # ------------------------------------------------------------------
    def encode(
        self,
        video_frames: Optional[torch.Tensor],
        query_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        precomputed_features: Optional[torch.Tensor] = None,
        return_patch_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """视觉路径 + 文本路径编码；支持离线 ``[T,D]``/``[T,P,D]`` 与在线帧张量。"""
        if precomputed_features is not None and precomputed_features.ndim == 2:
            precomputed_features = precomputed_features.unsqueeze(0)
        elif (
            precomputed_features is not None
            and precomputed_features.ndim == 3
            and query_tokens.shape[0] == 1
            and precomputed_features.shape[0] != query_tokens.shape[0]
        ):
            # Accept unbatched cached tensors such as [T, D] or [T, P, D].
            precomputed_features = precomputed_features.unsqueeze(0)
        f_raw = self.vis_encoder(
            video_frames=video_frames,
            precomputed_features=precomputed_features,
            return_patch_tokens=return_patch_tokens,
        )
        f_q = self.text_encoder(query_tokens, attention_mask=attention_mask)
        return {
            "F_raw": f_raw,
            "f_q": f_q,
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        video_frames: Optional[torch.Tensor],
        query_tokens: torch.Tensor,
        mode: str = "stage1",
        gt_mask: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        precomputed_features: Optional[torch.Tensor] = None,
        policy_intra_event_topk: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """前向：``stage1`` / ``stage2_backbone`` 仅 DSAM+CGRM；``stage2`` 加策略；``infer`` 确定性选帧。

        ``policy_intra_event_topk``：仅 ``stage2`` 有效；GRPO 训练应置 ``False`` 以使轨迹为因子化 Bernoulli，
        并与 ``trajectory_log_prob`` 一致。推理可保持 ``True`` 做事件内 top-k。
        """
        enc = self.encode(
            video_frames,
            query_tokens,
            attention_mask=attention_mask,
            precomputed_features=precomputed_features,
            return_patch_tokens=True,
        )
        f_raw, f_q = enc["F_raw"], enc["f_q"]

        if self.pretrained_feature_adapter is not None:
            f_raw = f_raw + self.cfg.pretrained_feature_blend * self.pretrained_feature_adapter(f_raw)

        ds = self.dsam(f_raw, f_q, gt_mask=gt_mask, frame_mask=frame_mask)
        ums = None
        if self.ums_adapter is not None:
            ums = self.ums_adapter(ds["F_fg"], f_q)
            ds["F_fg"] = ums["ums_fused"]
            ds["ums_lac"] = ums["lac"]
            ds["ums_vac"] = ums["vac"]
            ds["loss_ums_orth"] = ums["orth_loss"]
            ds["ums_mdi"] = ums["mdi"]
            ds["ums_mae_gate"] = ums["mae_gate"]
        everest_scores = None
        if self.cfg.use_everest_pruning:
            everest_scores = temporal_disparity_scores(ds["F_fg"])
            ds["everest_scores"] = everest_scores
        bridge = self._build_patch_bridge(ds, frame_mask=frame_mask)
        f_bg = ds["F_bg"] if self.cfg.cgrm_use_f_bg else None
        cg = self.cgrm(
            ds["F_fg"],
            ds["S_sem"],
            f_q,
            frame_patch_stats=bridge.get("frame_patch_stats"),
            f_bg=f_bg,
        )
        variant_diag = self._variant_diagnostics(ds, cg, ums, everest_scores, f_q)

        if mode in ("stage1", "stage2_backbone"):
            return {
                **enc,
                **ds,
                **bridge,
                **cg,
                "variant_diagnostics": variant_diag,
            }

        sample = mode == "stage2"
        # 传入事件级张量以启用「先事件后帧」的两级策略
        po = self.policy(
            cg["H_graph"],
            f_q,
            cg["S_graph"],
            sample=sample,
            h_event=cg.get("H_event"),
            s_event=cg.get("S_event"),
            membership=cg.get("membership"),
            frame_aux=bridge.get("frame_patch_stats"),
            event_aux=cg.get("event_patch_stats"),
            intra_event_topk=policy_intra_event_topk,
        )

        if mode == "infer":
            if self.cfg.use_hornet_policy:
                graph_probs = torch.softmax(cg["S_graph"], dim=-1)
                po["probs"] = (
                    (1.0 - self.cfg.hornet_policy_weight) * po["probs"]
                    + self.cfg.hornet_policy_weight * graph_probs
                )
                po["probs"] = po["probs"] / po["probs"].sum(dim=-1, keepdim=True).clamp_min(1e-6)
            if self.cfg.use_dynamic_logit_threshold:
                s = cg["S_graph"]
                threshold = s.mean(dim=-1, keepdim=True) + s.std(dim=-1, keepdim=True, unbiased=False)
                po["probs"] = po["probs"] * torch.sigmoid(s - threshold)
                po["probs"] = po["probs"] / po["probs"].sum(dim=-1, keepdim=True).clamp_min(1e-6)
            po["actions"] = self._select_actions_infer(
                po["probs"],
                membership=cg.get("membership"),
                event_probs=po.get("event_probs"),
                query_features=f_q,
                frame_features=ds["F_fg"],
            )

        return {
            **enc,
            **ds,
            **bridge,
            **cg,
            **po,
            "variant_diagnostics": variant_diag,
        }
