"""Decoupled Semantic Alignment Module（DSAM）：patch 级前景/背景分解与 frame 聚合桥。

输入 ``[B,T,D]`` 或 ``[B,T,P,D]``；内部统一为 patch token；输出 frame 级 ``F_fg`` 等
及 patch 级中间量供损失与可视化。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


class _ProtoDecoderBlock(nn.Module):
    """轻量 cross-attn：patch token 作为 query，背景原型作为 K/V，再接 FFN。"""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        self.ln_x = nn.LayerNorm(dim)
        self.ln_p = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True
        )
        self.ln_ff = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
        """``x``: [B, T*P, D] patch 序列；``proto``: [B, K, D] 背景原型。返回同形状更新后的 x。"""
        x_n = self.ln_x(x)
        p_n = self.ln_p(proto)
        attn_out, _ = self.cross_attn(query=x_n, key=p_n, value=p_n)
        x = x + attn_out
        x = x + self.ffn(self.ln_ff(x))
        return x


class DSAM(nn.Module):
    """Patch-level DSAM：背景原型池化、重建、前景残差、query-aware patch→frame 聚合。"""

    def __init__(
        self,
        dim: int,
        bg_prototypes: int = 8,
        tau: float = 0.07,
        alpha_init: float = 1.0,
        eps: float = 1e-6,
        agg_temperature: float = 0.5,
        use_isoclip_debias: bool = False,
        use_riemann_slerp: bool = False,
        use_soft_mask: bool = False,
        isoclip_debias_strength: float = 0.35,
        riemann_slerp_strength: float = 0.35,
        soft_mask_strength: float = 0.35,
    ):
        """``bg_prototypes`` 为背景原型数；``agg_temperature`` 控制聚合 softmax 锐度。"""
        super().__init__()
        self.dim = dim
        self.bg_prototypes = bg_prototypes
        self.tau = tau
        self.eps = eps
        self.agg_temperature = agg_temperature
        self.use_isoclip_debias = bool(use_isoclip_debias)
        self.use_riemann_slerp = bool(use_riemann_slerp)
        self.use_soft_mask = bool(use_soft_mask)
        self.isoclip_debias_strength = float(isoclip_debias_strength)
        self.riemann_slerp_strength = float(riemann_slerp_strength)
        self.soft_mask_strength = float(soft_mask_strength)

        self.q_bg = nn.Parameter(torch.randn(bg_prototypes, dim) * 0.02)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.patch_ln = nn.LayerNorm(dim)
        self.frame_ln = nn.LayerNorm(dim)
        self.frame_bg_ln = nn.LayerNorm(dim)
        self.fusion_gate = nn.Parameter(torch.tensor(0.0))

        self.bottleneck = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.decoder = _ProtoDecoderBlock(dim=dim, heads=4)
        self.frame_bridge = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def _match_norm(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Rescale ``x`` to the per-token norm of ``ref`` to avoid norm collapse."""
        return F.normalize(x, dim=-1, eps=self.eps) * ref.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def _apply_isoclip_debias(self, f_raw_patch: torch.Tensor) -> torch.Tensor:
        """Approximate IsoCLIP-style cone debiasing using per-video DC component removal."""
        if not self.use_isoclip_debias:
            return f_raw_patch
        strength = max(0.0, min(1.0, self.isoclip_debias_strength))
        dc = f_raw_patch.mean(dim=(1, 2), keepdim=True)
        debiased = f_raw_patch - strength * dc
        return self._match_norm(debiased, f_raw_patch)

    def _ema_anchor(self, f_bg_patch: torch.Tensor, beta: float = 0.85) -> torch.Tensor:
        """Build a dynamic background anchor along time for each patch channel."""
        anchors = []
        cur = f_bg_patch[:, 0]
        anchors.append(cur)
        for idx in range(1, f_bg_patch.shape[1]):
            cur = beta * cur + (1.0 - beta) * f_bg_patch[:, idx]
            anchors.append(cur)
        return torch.stack(anchors, dim=1)

    def _slerp_away_from_anchor(self, f_raw_patch: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Move features away from the background anchor on the hypersphere."""
        strength = max(0.0, min(1.0, self.riemann_slerp_strength))
        x_n = F.normalize(f_raw_patch, dim=-1, eps=self.eps)
        a_n = F.normalize(anchor, dim=-1, eps=self.eps)
        dot = (x_n * a_n).sum(dim=-1, keepdim=True)
        tangent = F.normalize(x_n - dot * a_n, dim=-1, eps=self.eps)
        moved = F.normalize((1.0 - strength) * x_n + strength * tangent, dim=-1, eps=self.eps)
        return moved * f_raw_patch.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def _soft_mask_background(self, f_raw_patch: torch.Tensor, f_bg_patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hadamard channel mask that softly suppresses likely background activations."""
        strength = max(0.0, min(1.0, self.soft_mask_strength))
        bg_scale = f_bg_patch.abs().mean(dim=-1, keepdim=True).clamp_min(self.eps)
        channel_bg = torch.sigmoid(f_bg_patch.abs() / bg_scale - 1.0)
        soft_mask = 1.0 - strength * channel_bg
        return f_raw_patch * soft_mask, soft_mask

    def _ensure_patch_tokens(self, f_raw: torch.Tensor) -> torch.Tensor:
        """将 ``[B,T,D]`` 扩一维为单 patch，或校验 ``[B,T,P,D]``。"""
        if f_raw.ndim == 3:
            return f_raw.unsqueeze(2)
        if f_raw.ndim == 4:
            return f_raw
        raise ValueError(f"Expected [B,T,D] or [B,T,P,D], got {tuple(f_raw.shape)}")

    def _build_patch_mask(
        self,
        f_raw_patch: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """由 frame 级 valid mask 广播得到 patch 级 mask ``[B,T,P]``。"""
        b, t, p, _ = f_raw_patch.shape
        if frame_mask is None:
            return torch.ones((b, t, p), dtype=torch.bool, device=f_raw_patch.device)
        return frame_mask.unsqueeze(-1).expand(-1, -1, p)

    def _masked_softmax(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor],
        dim: int,
    ) -> torch.Tensor:
        """无效位置填大负值后 softmax，再乘 mask 并归一化，避免 padding 泄漏。"""
        if mask is None:
            return torch.softmax(logits, dim=dim)
        mask = mask.to(torch.bool)
        masked_logits = logits.masked_fill(~mask, -1e4)
        probs = torch.softmax(masked_logits, dim=dim)
        probs = probs * mask.to(probs.dtype)
        return probs / probs.sum(dim=dim, keepdim=True).clamp_min(self.eps)

    def _compute_background_prototypes(
        self, f_raw: torch.Tensor, frame_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """全视频 patch 对可学习 ``q_bg`` 做注意力读出，得到 batch 级背景原型 ``[B,K,D]``。"""
        f_raw_patch = self._ensure_patch_tokens(f_raw)
        patch_mask = self._build_patch_mask(f_raw_patch, frame_mask=frame_mask)
        b, _, _, d = f_raw_patch.shape
        tokens = f_raw_patch.reshape(b, -1, d)
        token_mask = patch_mask.reshape(b, -1)
        q = self.q_bg.unsqueeze(0).expand(b, -1, -1)
        attn = torch.matmul(q, tokens.transpose(1, 2)) / (d ** 0.5)
        attn = self._masked_softmax(attn, token_mask.unsqueeze(1), dim=-1)
        p_bg = torch.matmul(attn, tokens)
        return p_bg

    def _reconstruct(self, f_raw: torch.Tensor, p_bg: torch.Tensor) -> torch.Tensor:
        """瓶颈 MLP + 原型 cross-attn 重建 patch 特征 ``[B,T,P,D]``。"""
        f_raw_patch = self._ensure_patch_tokens(f_raw)
        b, t, p, d = f_raw_patch.shape
        x = self.bottleneck(f_raw_patch).reshape(b, t * p, d)
        x = self.decoder(x, p_bg)
        return x.reshape(b, t, p, d)

    def _reconstruction_error(
        self,
        f_raw: torch.Tensor,
        f_recon: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """逐 patch 1-cos 归一化到 [0,1] 的重建误差，乘 patch mask。"""
        f_raw = self._ensure_patch_tokens(f_raw)
        f_recon = self._ensure_patch_tokens(f_recon)
        err = 1.0 - F.cosine_similarity(f_raw, f_recon, dim=-1)
        err = err / 2.0
        patch_mask = self._build_patch_mask(f_raw, frame_mask=frame_mask)
        err = err * patch_mask.to(err.dtype)
        return err

    def _compactness_loss(
        self,
        f_raw: torch.Tensor,
        p_bg: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """鼓励每个 patch 靠近某一背景原型（最小角距离均值）。"""
        f_raw = self._ensure_patch_tokens(f_raw)
        patch_mask = self._build_patch_mask(f_raw, frame_mask=frame_mask)
        f_n = F.normalize(f_raw, dim=-1)
        p_n = F.normalize(p_bg, dim=-1)
        dot = torch.matmul(
            f_n.reshape(f_n.shape[0], -1, f_n.shape[-1]),
            p_n.transpose(1, 2),
        )
        dist = 1.0 - dot
        min_dist = dist.min(dim=-1).values
        if frame_mask is None:
            return min_dist.mean()
        valid = patch_mask.reshape(patch_mask.shape[0], -1).to(min_dist.dtype)
        return (min_dist * valid).sum() / valid.sum().clamp_min(1.0)

    def _project_to_bg(self, f_raw: torch.Tensor, p_bg: torch.Tensor) -> torch.Tensor:
        """将 patch 向量投影到各原型子空间并求和，得到背景分量 ``f_bg_patch``。"""
        f_raw = self._ensure_patch_tokens(f_raw)
        proto = p_bg[:, None, None, :, :]
        numer = (f_raw.unsqueeze(-2) * proto).sum(dim=-1)
        denom = proto.pow(2).sum(dim=-1) + self.eps
        coef = numer / denom
        f_bg = (coef.unsqueeze(-1) * proto).sum(dim=-2)
        return f_bg

    def _orthogonality_loss(
        self,
        f_fg: torch.Tensor,
        f_bg: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """最小化前景与背景 patch 的余弦相似绝对值（解耦）。"""
        f_fg = self._ensure_patch_tokens(f_fg)
        f_bg = self._ensure_patch_tokens(f_bg)
        cos = F.cosine_similarity(f_fg, f_bg, dim=-1).abs()
        if frame_mask is None:
            return cos.mean()
        valid = self._build_patch_mask(f_fg, frame_mask=frame_mask).to(cos.dtype)
        return (cos * valid).sum() / valid.sum().clamp_min(1.0)

    def _semantic_scores(
        self,
        f_fg: torch.Tensor,
        f_q: torch.Tensor,
        recon_error: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """patch 与查询的归一化点积经 sigmoid，并与重建误差门控融合为 saliency。"""
        f_fg = self._ensure_patch_tokens(f_fg)
        f_q = F.normalize(f_q, dim=-1)
        while f_q.ndim < f_fg.ndim:
            f_q = f_q.unsqueeze(1)
        f_fg = F.normalize(f_fg, dim=-1)
        sim = (f_fg * f_q).sum(dim=-1)
        # 时间维度均值中心化：缓解预提取大模型特征帧间过相似、Sigmoid 饱和
        if sim.shape[1] > 1:
            mu = sim.mean(dim=1, keepdim=True)
            sim = sim - mu
        s_sem = torch.sigmoid(sim / self.tau)
        if recon_error is not None:
            gate = torch.sigmoid(self.fusion_gate)
            s_sem = (1.0 - gate) * s_sem + gate * recon_error
        patch_mask = self._build_patch_mask(f_fg, frame_mask=frame_mask)
        return s_sem * patch_mask.to(s_sem.dtype)

    def _patch_proto_usage(
        self,
        f_raw_patch: torch.Tensor,
        p_bg: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """每个 patch 对 K 个原型的软分配 ``[B,T,P,K]``，用于熵类统计。"""
        f_n = F.normalize(f_raw_patch, dim=-1)
        p_n = F.normalize(p_bg, dim=-1)
        logits = torch.matmul(
            f_n.reshape(f_n.shape[0], -1, f_n.shape[-1]),
            p_n.transpose(1, 2),
        ) / self.tau
        usage = self._masked_softmax(
            logits,
            patch_mask.reshape(patch_mask.shape[0], -1, 1).expand(-1, -1, p_bg.shape[1]),
            dim=-1,
        )
        return usage.reshape(f_raw_patch.shape[0], f_raw_patch.shape[1], f_raw_patch.shape[2], p_bg.shape[1])

    def _aggregate_patch_features(
        self,
        values: torch.Tensor,
        patch_weights: torch.Tensor,
    ) -> torch.Tensor:
        """对最后一维 patch 做加权求和得到 frame 向量 ``[B,T,D]``。"""
        return (patch_weights.unsqueeze(-1) * values).sum(dim=2)

    def _aggregate_patch_scores(
        self,
        values: torch.Tensor,
        patch_weights: torch.Tensor,
    ) -> torch.Tensor:
        """标量 patch 分数加权到 frame 标量 ``[B,T]``。"""
        return (patch_weights * values).sum(dim=2)

    def _patch_to_frame_bridge(
        self,
        f_fg_patch: torch.Tensor,
        f_q: torch.Tensor,
        s_patch: torch.Tensor,
        recon_error_patch: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """query-aware patch 聚合权重：MLP(logit) + detach 的 saliency/误差偏置，再 masked softmax。"""
        q = F.normalize(f_q, dim=-1).unsqueeze(1).unsqueeze(1)
        q = q.expand(-1, f_fg_patch.shape[1], f_fg_patch.shape[2], -1)
        bridge_in = torch.cat([self.patch_ln(f_fg_patch), q], dim=-1)
        logits = self.frame_bridge(bridge_in).squeeze(-1)
        logits = logits + s_patch.detach() + recon_error_patch.detach()
        return self._masked_softmax(logits / self.agg_temperature, patch_mask, dim=-1)

    def _alignment_loss(
        self,
        s_sem: torch.Tensor,
        gt_mask: Optional[torch.Tensor],
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """弱监督：使用标准 BCE 同时优化正负样本，避免全局输出趋同。"""
        if gt_mask is None:
            return torch.zeros((), device=s_sem.device, dtype=s_sem.dtype)
        target = gt_mask.to(s_sem.dtype)
        loss = F.binary_cross_entropy(s_sem, target, reduction="none")
        if frame_mask is not None:
            valid = frame_mask.to(s_sem.dtype)
            return (loss * valid).sum() / valid.sum().clamp_min(1.0)
        return loss.mean()

    def forward(
        self,
        f_raw: torch.Tensor,
        f_q: torch.Tensor,
        gt_mask: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """完整 patch 分解前向：返回 frame/patch 特征、分数、各项损失张量。"""
        f_raw_patch = self._ensure_patch_tokens(f_raw)
        f_work_patch = self._apply_isoclip_debias(f_raw_patch)
        patch_mask = self._build_patch_mask(f_raw_patch, frame_mask=frame_mask)
        p_bg = self._compute_background_prototypes(f_work_patch, frame_mask=frame_mask)

        f_recon_patch = self._reconstruct(f_work_patch, p_bg)
        recon_error_patch = self._reconstruction_error(f_work_patch, f_recon_patch, frame_mask)

        f_bg_patch = self._project_to_bg(f_work_patch, p_bg)
        riemann_anchor_patch = None
        soft_mask_patch = None
        if self.use_riemann_slerp:
            riemann_anchor_patch = self._ema_anchor(f_bg_patch)
            f_fg_raw_patch = self._slerp_away_from_anchor(f_work_patch, riemann_anchor_patch)
        else:
            # 可学习标量 alpha 控制背景减除强度
            f_fg_raw_patch = f_work_patch - self.alpha * f_bg_patch
        if self.use_soft_mask:
            f_fg_raw_patch, soft_mask_patch = self._soft_mask_background(f_fg_raw_patch, f_bg_patch)
        # 重建难区域放大前景残差（detach 门控稳定反传）
        gate = recon_error_patch.unsqueeze(-1).detach()
        f_fg_patch = self.patch_ln(f_fg_raw_patch * (1.0 + gate))
        s_patch = self._semantic_scores(
            f_fg_patch,
            f_q,
            recon_error=recon_error_patch,
            frame_mask=frame_mask,
        )
        patch_weights = self._patch_to_frame_bridge(
            f_fg_patch,
            f_q,
            s_patch,
            recon_error_patch,
            patch_mask,
        )

        f_fg = self.frame_ln(self._aggregate_patch_features(f_fg_patch, patch_weights))
        f_bg = self.frame_bg_ln(self._aggregate_patch_features(f_bg_patch, patch_weights))
        f_recon = self._aggregate_patch_features(f_recon_patch, patch_weights)
        recon_error = self._aggregate_patch_scores(recon_error_patch, patch_weights)
        s_sem = self._aggregate_patch_scores(s_patch, patch_weights)
        if frame_mask is not None:
            valid = frame_mask.to(f_fg.dtype).unsqueeze(-1)
            f_fg = f_fg * valid
            f_bg = f_bg * valid
            f_recon = f_recon * valid
            recon_error = recon_error * frame_mask.to(recon_error.dtype)
            s_sem = s_sem * frame_mask.to(s_sem.dtype)

        prototype_usage = self._patch_proto_usage(f_raw_patch, p_bg, patch_mask)

        loss_compact = self._compactness_loss(f_work_patch, p_bg, frame_mask=frame_mask)
        loss_orth = self._orthogonality_loss(f_fg_patch, f_bg_patch, frame_mask=frame_mask)
        loss_align = self._alignment_loss(s_sem, gt_mask, frame_mask=frame_mask)
        loss_recon = (
            recon_error_patch.sum() / patch_mask.to(recon_error_patch.dtype).sum().clamp_min(1.0)
            if frame_mask is not None
            else recon_error_patch.mean()
        )

        out = {
            "F_fg": f_fg,
            "F_bg": f_bg,
            "F_recon": f_recon,
            "F_debiased_patch": f_work_patch,
            "F_fg_patch": f_fg_patch,
            "F_bg_patch": f_bg_patch,
            "F_recon_patch": f_recon_patch,
            "P_bg": p_bg,
            "S_sem": s_sem,
            "S_patch": s_patch,
            "recon_error": recon_error,
            "recon_error_patch": recon_error_patch,
            "patch_weights": patch_weights,
            "prototype_usage": prototype_usage,
            "loss_compact": loss_compact,
            "loss_orth": loss_orth,
            "loss_align": loss_align,
            "loss_recon": loss_recon,
        }
        if riemann_anchor_patch is not None:
            out["riemann_anchor_patch"] = riemann_anchor_patch
        if soft_mask_patch is not None:
            out["soft_mask_patch"] = soft_mask_patch
        return out
