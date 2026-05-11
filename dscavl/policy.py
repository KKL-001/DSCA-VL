"""事件感知帧策略头：先选时间事件再在事件内 top-k，对外仍为 ``[B,T]`` 的 logits/probs/actions。"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn


class FramePolicyHead(nn.Module):
    """两级策略：事件级 gate + 帧级打分；无事件张量时退化为扁平帧策略（兼容 Stage1）。"""

    def __init__(
        self,
        dim: int,
        gamma: float = 1.0,
        hidden_dim: int = 512,
        intra_top_ratio: float = 0.5,
    ):
        """``intra_top_ratio``：每个已选事件内保留的帧比例上限（用于 top-k）。"""
        super().__init__()
        self.gamma = gamma
        self.intra_top_ratio = intra_top_ratio

        # ---- event-level gate: decide which events to select ----
        self.event_gate = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # ---- frame-level scorer: rank frames within a selected event ----
        self.frame_scorer = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.frame_aux_scorer = nn.Linear(5, 1)
        self.event_aux_scorer = nn.Linear(5, 1)

        nn.init.zeros_(self.frame_aux_scorer.weight)
        nn.init.zeros_(self.frame_aux_scorer.bias)
        nn.init.zeros_(self.event_aux_scorer.weight)
        nn.init.zeros_(self.event_aux_scorer.bias)

        # 可验收计划 §7 矩阵 A1：在 concat 前对图表示做 query 条件缩放。权重/bias 全零 → sigmoid(0)=0.5 → ×2 为 1，与旧前向一致；
        # 旧 checkpoint 在 strict=False 下缺失本层时，仍保持该初值，避免随机初始化扰动行为。
        self.h_graph_query_film = nn.Linear(dim, dim, bias=True)
        self.h_event_query_film = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.h_graph_query_film.weight)
        nn.init.zeros_(self.h_graph_query_film.bias)
        nn.init.zeros_(self.h_event_query_film.weight)
        nn.init.zeros_(self.h_event_query_film.bias)

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        """最后一维标准化，稳定与 ``gamma * score`` 的加法尺度。"""
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (x - mu) / std

    @staticmethod
    def _aux_bias(aux: torch.Tensor | None, scorer: nn.Linear) -> torch.Tensor | None:
        """将 patch 统计等辅助特征经线性层得到有界 tanh 偏置；形状不匹配则忽略。"""
        if aux is None or aux.ndim != 3 or aux.shape[-1] != scorer.in_features:
            return None
        bias = scorer(torch.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0)).squeeze(-1)
        return torch.tanh(torch.nan_to_num(bias, nan=0.0, posinf=5.0, neginf=-5.0))

    def _policy_scoring(
        self,
        h_graph: torch.Tensor,
        f_q: torch.Tensor,
        s_graph: torch.Tensor,
        h_event: torch.Tensor | None,
        s_event: torch.Tensor | None,
        membership: torch.Tensor | None,
        frame_aux: torch.Tensor | None,
        event_aux: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | bool | None]:
        """共享：帧/事件 logits 与 Bernoulli 概率（无采样）。"""
        b, t, d = h_graph.shape
        g_h = torch.sigmoid(self.h_graph_query_film(f_q)).unsqueeze(1) * 2.0
        h_graph = h_graph * g_h
        f_q_t = f_q.unsqueeze(1).expand(b, t, -1)
        frame_aux_bias = self._aux_bias(frame_aux, self.frame_aux_scorer)

        has_events = (
            h_event is not None
            and s_event is not None
            and membership is not None
        )
        if has_events and h_event is not None:
            g_e = torch.sigmoid(self.h_event_query_film(f_q)).unsqueeze(1) * 2.0
            h_event = h_event * g_e

        frame_in = torch.cat([h_graph, f_q_t], dim=-1)
        s_graph_norm = self._normalize(s_graph)
        frame_logits_raw = (
            self.frame_scorer(frame_in).squeeze(-1)
            + self.gamma * s_graph_norm
        )
        if frame_aux_bias is not None and frame_aux_bias.shape == frame_logits_raw.shape:
            frame_logits_raw = frame_logits_raw + frame_aux_bias
        frame_logits = torch.nan_to_num(
            frame_logits_raw, nan=0.0, posinf=30.0, neginf=-30.0
        ).clamp(-30.0, 30.0)
        frame_scores = torch.sigmoid(frame_logits).clamp(1e-6, 1.0 - 1e-6)

        if not has_events:
            return {
                "has_events": False,
                "frame_logits": frame_logits,
                "frame_scores": frame_scores,
            }

        m = h_event.shape[1]
        f_q_e = f_q.unsqueeze(1).expand(b, m, -1)
        event_aux_bias = self._aux_bias(event_aux, self.event_aux_scorer)
        event_in = torch.cat([h_event, f_q_e], dim=-1)
        s_event_norm = self._normalize(s_event)
        event_logits_raw = (
            self.event_gate(event_in).squeeze(-1)
            + self.gamma * s_event_norm
        )
        if event_aux_bias is not None and event_aux_bias.shape == event_logits_raw.shape:
            event_logits_raw = event_logits_raw + event_aux_bias
        event_logits = torch.nan_to_num(
            event_logits_raw, nan=0.0, posinf=30.0, neginf=-30.0
        ).clamp(-30.0, 30.0)
        event_probs = torch.sigmoid(event_logits).clamp(1e-6, 1.0 - 1e-6)

        return {
            "has_events": True,
            "frame_logits": frame_logits,
            "frame_scores": frame_scores,
            "event_logits": event_logits,
            "event_probs": event_probs,
            "membership": membership,
        }

    def trajectory_log_prob(
        self,
        h_graph: torch.Tensor,
        f_q: torch.Tensor,
        s_graph: torch.Tensor,
        actions: torch.Tensor,
        *,
        h_event: torch.Tensor | None = None,
        s_event: torch.Tensor | None = None,
        membership: torch.Tensor | None = None,
        frame_aux: torch.Tensor | None = None,
        event_aux: torch.Tensor | None = None,
        event_actions: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """在给定轨迹下计算 \\(\\log \\pi(a,e)\\)；与 ``intra_event_topk=False`` 的采样一致。

        层次策略：先事件 Bernoulli，再帧 Bernoulli，帧成功参数为 ``sigmoid(frame_logit)*event_mask``；
        ``event_mask==0`` 时要求 ``actions==0``（否则贡献 ``-inf`` 近似）。
        扁平策略：仅帧维独立 Bernoulli。
        返回 ``[B]``（对时间/事件维求和）。
        """
        sc = self._policy_scoring(
            h_graph, f_q, s_graph, h_event, s_event, membership, frame_aux, event_aux
        )
        a = actions.clamp(0.0, 1.0)
        if not sc["has_events"]:
            p = sc["frame_scores"].clamp(eps, 1.0 - eps)
            return (a * p.log() + (1.0 - a) * (1.0 - p).log()).sum(dim=-1)

        if event_actions is None:
            raise ValueError("trajectory_log_prob: event_actions required for hierarchical policy")
        ea = event_actions.clamp(0.0, 1.0)
        ep = sc["event_probs"].clamp(eps, 1.0 - eps)
        lp_e = (ea * ep.log() + (1.0 - ea) * (1.0 - ep).log()).sum(dim=-1)

        mem = sc["membership"]
        event_mask = torch.matmul(mem, ea.unsqueeze(-1)).squeeze(-1).clamp(0.0, 1.0)
        fs = sc["frame_scores"].clamp(eps, 1.0 - eps)
        p_f = (fs * event_mask).clamp(eps, 1.0 - eps)
        mask_zero = event_mask < 1e-8
        lp_f_elem = a * p_f.log() + (1.0 - a) * (1.0 - p_f).log()
        neg_large = torch.full_like(lp_f_elem, -1e8)
        lp_f_elem = torch.where(
            mask_zero,
            torch.where(a < 0.5, torch.zeros_like(lp_f_elem), neg_large),
            lp_f_elem,
        )
        lp_f = lp_f_elem.sum(dim=-1)
        return lp_e + lp_f

    def kl_logits_pairs_with(
        self,
        ref: "FramePolicyHead",
        h_graph: torch.Tensor,
        f_q: torch.Tensor,
        s_graph: torch.Tensor,
        *,
        h_event: torch.Tensor | None = None,
        s_event: torch.Tensor | None = None,
        membership: torch.Tensor | None = None,
        frame_aux: torch.Tensor | None = None,
        event_aux: torch.Tensor | None = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """与参考策略的逐块 Bernoulli KL：``(frame_logits, frame_logits_ref)`` 及可选事件对。"""
        sn = self._policy_scoring(
            h_graph, f_q, s_graph, h_event, s_event, membership, frame_aux, event_aux
        )
        sr = ref._policy_scoring(
            h_graph, f_q, s_graph, h_event, s_event, membership, frame_aux, event_aux
        )
        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (sn["frame_logits"], sr["frame_logits"])
        ]
        if sn["has_events"] and sr["has_events"]:
            pairs.append((sn["event_logits"], sr["event_logits"]))
        return pairs

    # ------------------------------------------------------------------
    def forward(
        self,
        h_graph: torch.Tensor,        # [B, T, D]
        f_q: torch.Tensor,            # [B, D]
        s_graph: torch.Tensor,        # [B, T]
        sample: bool = True,
        threshold: float = 0.5,
        intra_event_topk: bool = True,
        # event-level tensors (supplied by model.forward)
        h_event: torch.Tensor | None = None,   # [B, M, D]
        s_event: torch.Tensor | None = None,   # [B, M]
        membership: torch.Tensor | None = None, # [B, T, M]
        frame_aux: torch.Tensor | None = None,  # [B, T, C]
        event_aux: torch.Tensor | None = None,  # [B, M, C]
    ) -> Dict[str, torch.Tensor]:
        """``sample=True`` 用 Bernoulli 采样；否则阈值化。

        ``intra_event_topk=False`` 时（GRPO 推荐）帧动作为独立 Bernoulli(probs)，与 ``trajectory_log_prob`` 一致。
        ``logits`` 仅为帧 logits（不再使用误导性的 composite logits）。
        """
        sc = self._policy_scoring(
            h_graph, f_q, s_graph, h_event, s_event, membership, frame_aux, event_aux
        )
        frame_logits = sc["frame_logits"]
        frame_scores = sc["frame_scores"]

        if not sc["has_events"]:
            logits = frame_logits
            probs = frame_scores
            if sample:
                actions = torch.bernoulli(probs)
            else:
                actions = (probs >= threshold).to(probs.dtype)
            return {
                "logits": logits,
                "frame_logits": frame_logits,
                "probs": probs,
                "actions": actions,
            }

        event_logits = sc["event_logits"]
        event_probs = sc["event_probs"]
        membership = sc["membership"]

        if sample:
            event_actions = torch.bernoulli(event_probs)
        else:
            event_actions = (event_probs >= threshold).to(event_probs.dtype)

        event_mask = torch.matmul(
            membership, event_actions.unsqueeze(-1)
        ).squeeze(-1).clamp(0.0, 1.0)
        probs = frame_scores * event_mask

        if intra_event_topk:
            actions = self._intra_event_topk(
                probs, membership, event_actions, sample, threshold,
            )
        elif sample:
            actions = torch.bernoulli(probs)
        else:
            actions = (probs >= threshold).to(probs.dtype)

        return {
            "logits": frame_logits,
            "frame_logits": frame_logits,
            "event_logits": event_logits,
            "probs": probs,
            "actions": actions,
            "event_probs": event_probs,
            "event_actions": event_actions,
        }

    # ------------------------------------------------------------------
    def _intra_event_topk(
        self,
        probs: torch.Tensor,          # [B, T]
        membership: torch.Tensor,      # [B, T, M]
        event_actions: torch.Tensor,   # [B, M]
        sample: bool,
        threshold: float,
    ) -> torch.Tensor:
        """在每个已选事件内按 ``intra_top_ratio`` 截断帧数；采样模式下保证至少选一帧。"""
        b, t = probs.shape
        m = membership.shape[2]
        actions = torch.zeros_like(probs)

        for bi in range(b):
            for ei in range(m):
                if event_actions[bi, ei] < 0.5:
                    continue
                frame_idx = (membership[bi, :, ei] > 1e-4).nonzero(
                    as_tuple=False
                ).squeeze(-1)
                if frame_idx.numel() == 0:
                    continue

                k = max(1, int(frame_idx.numel() * self.intra_top_ratio))
                local_probs = probs[bi, frame_idx]

                if sample:
                    selected = torch.bernoulli(local_probs)
                    n_sel = int(selected.sum().item())
                    if n_sel < 1:
                        top1 = local_probs.argmax()
                        selected[top1] = 1.0
                    elif n_sel > k:
                        _, top_idx = local_probs.topk(k)
                        selected = torch.zeros_like(local_probs)
                        selected[top_idx] = 1.0
                else:
                    _, top_idx = local_probs.topk(k)
                    selected = torch.zeros_like(local_probs)
                    selected[top_idx] = 1.0

                actions[bi, frame_idx] = selected

        return actions
