"""Causal Graph Reasoning Module（CGRM）：滑动窗口事件池化 + 时序/语义/因果邻接与一步消息传递。

输出 frame 图表示 ``H_graph``、事件分数等；可选融合 DSAM 的 frame 级 patch 统计。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def build_membership_matrix(t: int, window: int, stride: int, device: torch.device) -> torch.Tensor:
    """构造帧→重叠时间窗事件的软归属矩阵 ``[T,M]``（按行归一化）。"""
    starts = list(range(0, max(1, t - window + 1), stride))
    if not starts:
        starts = [0]
    m = len(starts)
    mem = torch.zeros(t, m, device=device)
    for idx, s in enumerate(starts):
        e = min(t, s + window)
        mem[s:e, idx] = 1.0
    # Normalize row-wise for stable frame <- event projection.
    mem = mem / mem.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return mem


class EventPooler(nn.Module):
    """按固定窗口/步长将帧特征池化为事件节点，并加正弦位置编码。"""

    def __init__(self, dim: int, window: int = 8, stride: int = 4):
        super().__init__()
        self.window = window
        self.stride = stride
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, f_fg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回事件嵌入 ``[B,M,D]`` 与帧-事件归属 ``[T,M]``（与 batch 无关）。"""
        # f_fg: [B, T, D]
        _, t, d = f_fg.shape
        mem = build_membership_matrix(t, self.window, self.stride, f_fg.device)  # [T, M]
        m = mem.shape[1]

        pooled = []
        for k in range(m):
            w = mem[:, k].unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            x = (f_fg * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
            pooled.append(x)
        e = torch.stack(pooled, dim=1)  # [B, M, D]
        e = self.proj(e)

        # Add simple sinusoidal position encoding on event index.
        pos = torch.arange(m, device=f_fg.device, dtype=f_fg.dtype).unsqueeze(-1)
        div = torch.exp(torch.arange(0, d, 2, device=f_fg.device, dtype=f_fg.dtype) * (-torch.log(torch.tensor(10000.0, device=f_fg.device)) / d))
        pe = torch.zeros(m, d, device=f_fg.device, dtype=f_fg.dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        e = e + pe.unsqueeze(0)

        return e, mem


class CGRM(nn.Module):
    """多图融合（时序高斯 + 事件语义 + 因果上三角）与 query 自适应边权混合。"""

    def __init__(
        self,
        dim: int,
        window: int = 8,
        stride: int = 4,
        sigma_temp: float = 3.0,
        beta1: float = 0.5,
        beta2: float = 0.5,
        causal_decay_tau: float = 6.0,
        bridge_weight: float = 0.25,
        use_f_bg: bool = False,
        mix_temperature: float = 1.0,
        use_sbp: bool = False,
        sbp_negative_threshold: float = 0.18,
        sbp_negative_weight: float = 0.25,
    ):
        """``bridge_weight`` 调节「中介桥接」项在事件分数中的比重。"""
        super().__init__()
        self.dim = dim
        self.use_f_bg = bool(use_f_bg)
        self.use_sbp = bool(use_sbp)
        self.sigma_temp = sigma_temp
        self.beta1 = beta1
        self.beta2 = beta2
        self.causal_decay_tau = max(float(causal_decay_tau), 1e-3)
        self.bridge_weight = float(bridge_weight)
        self.mix_temperature = max(float(mix_temperature), 1e-3)
        self.sbp_negative_threshold = max(float(sbp_negative_threshold), 0.0)
        self.sbp_negative_weight = max(float(sbp_negative_weight), 0.0)

        self.event_pooler = EventPooler(dim=dim, window=window, stride=stride)
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)

        self.mix = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
        )
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.constant_(self.mix[-1].bias, 0.0)
        self.mix[-1].bias.data[0] = 0.5
        # Query-adaptive direction preference:
        # high -> prefer cause-like hubs (out-degree), low -> prefer effect-like hubs (in-degree).
        self.dir_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        self.frame_update = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.event_update = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.frame_aux_proj = nn.Sequential(
            nn.Linear(5, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.frame_aux_gate = nn.Linear(5, 1)
        self.event_aux_gate = nn.Linear(5, 1)

        if self.use_f_bg:
            self.fg_bg_fuse = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
            )
        else:
            self.fg_bg_fuse = None

        nn.init.zeros_(self.frame_aux_proj[-1].weight)
        nn.init.zeros_(self.frame_aux_proj[-1].bias)
        nn.init.zeros_(self.frame_aux_gate.weight)
        nn.init.zeros_(self.frame_aux_gate.bias)
        nn.init.zeros_(self.event_aux_gate.weight)
        nn.init.zeros_(self.event_aux_gate.bias)
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.score_bias = nn.Parameter(torch.tensor(0.0))

    def _temporal_adjacency(self, t: int, window: int, device: torch.device) -> torch.Tensor:
        """帧级高斯核邻接（窗口内），对角线为 1。"""
        idx = torch.arange(t, device=device)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs().float()
        mask = (dist < float(window)).float()
        a = torch.exp(-(dist ** 2) / (2 * (self.sigma_temp ** 2))) * mask
        a.fill_diagonal_(1.0)
        return a

    def _semantic_event_adjacency(self, e: torch.Tensor) -> torch.Tensor:
        """事件余弦相似度，按行去均值后 ReLU，抑制高相似度背景、提取显著边。"""
        e_norm = F.normalize(e, dim=-1)
        sim = torch.matmul(e_norm, e_norm.transpose(1, 2))
        mu = sim.mean(dim=-1, keepdim=True)
        return F.relu(sim - mu)

    def _causal_event_adjacency(self, e: torch.Tensor) -> torch.Tensor:
        """可学习 QK 得到有向边；logits 行方向中心化后 Sigmoid，防高度相关时饱和为 1。"""
        q = self.w_q(e)
        k = self.w_k(e)
        logits = torch.matmul(q, k.transpose(1, 2)) / (self.dim ** 0.5)
        logits = logits - logits.mean(dim=-1, keepdim=True)
        a = torch.sigmoid(logits)
        m = a.shape[-1]
        causal_mask = torch.triu(torch.ones(m, m, device=e.device), diagonal=1)
        idx = torch.arange(m, device=e.device)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).float().abs()
        decay = torch.exp(-dist / self.causal_decay_tau)
        return a * causal_mask.unsqueeze(0) * decay.unsqueeze(0)

    @staticmethod
    def _normalize_score(x: torch.Tensor) -> torch.Tensor:
        """按最后一维做零均值单位方差，用于度统计可比性。

        ``std`` 下限取 ``1e-2``（而非 ``1e-6``）：过小会在反传时放大梯度，弱监督
        首次反传经 ``S_event`` 回 CGRM 时易触发 NaN/Inf（``stage1`` 主损失不经 CGRM）。
        """
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-2)
        return (x - mu) / std

    def _one_step_message_passing(self, x: torch.Tensor, a: torch.Tensor, updater: nn.Module) -> torch.Tensor:
        """行归一化邻接后聚合邻居，再经 ``updater`` MLP。"""
        # x: [B, N, D], a: [B, N, N] or [N, N]
        if a.dim() == 2:
            a = a.unsqueeze(0).expand(x.size(0), -1, -1)
        row_sum = a.sum(dim=-1, keepdim=True).clamp_min(1e-3)
        a = a / row_sum
        msg = torch.matmul(a, x)
        return updater(msg)

    def _negative_event_adjacency(self, e: torch.Tensor) -> torch.Tensor:
        """Build SBP negative edges from event cosine distance."""
        e_norm = F.normalize(e, dim=-1)
        sim = torch.matmul(e_norm, e_norm.transpose(1, 2))
        dist = (1.0 - sim).clamp_min(0.0)
        a_neg = F.relu(dist - self.sbp_negative_threshold)
        eye = torch.eye(a_neg.shape[-1], device=e.device, dtype=a_neg.dtype).unsqueeze(0)
        return a_neg * (1.0 - eye)

    def _one_step_signed_message_passing(
        self,
        x: torch.Tensor,
        a_pos: torch.Tensor,
        a_neg: torch.Tensor,
        updater: nn.Module,
    ) -> torch.Tensor:
        """SBP propagation: positive aggregation minus normalized negative-edge repulsion."""
        pos_sum = a_pos.sum(dim=-1, keepdim=True).clamp_min(1e-3)
        neg_sum = a_neg.sum(dim=-1, keepdim=True).clamp_min(1e-3)
        pos_msg = torch.matmul(a_pos / pos_sum, x)
        neg_msg = torch.matmul(a_neg / neg_sum, x)
        return updater(pos_msg - self.sbp_negative_weight * neg_msg)

    @staticmethod
    def _sanitize_patch_stats(
        frame_patch_stats: Optional[torch.Tensor],
        expected_frames: int,
    ) -> Optional[torch.Tensor]:
        """校验 ``[B,T,C]`` 与帧数一致并清理非有限值。"""
        if frame_patch_stats is None or frame_patch_stats.ndim != 3:
            return None
        if frame_patch_stats.shape[1] != expected_frames:
            return None
        return torch.nan_to_num(frame_patch_stats, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _pool_event_patch_stats(
        frame_patch_stats: torch.Tensor,
        membership: torch.Tensor,
    ) -> torch.Tensor:
        """用帧-事件归属把 frame 级 patch 统计池到事件级 ``[B,M,C]``。"""
        weights = membership.transpose(0, 1).unsqueeze(0)  # [1, M, T]
        numer = torch.matmul(weights, frame_patch_stats)   # [B, M, C]
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return numer / denom

    def build_frame_input(
        self,
        f_fg: torch.Tensor,
        f_bg: Optional[torch.Tensor],
        frame_patch_stats: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """与 ``forward`` 一致：Fg（及可选 Bg）融合 + patch 统计投影，得到 EventPooler 输入 ``[B,T,D]``。"""
        _, t, _ = f_fg.shape
        stats = self._sanitize_patch_stats(frame_patch_stats, expected_frames=t)
        if self.use_f_bg:
            if f_bg is None:
                raise ValueError("CGRM(use_f_bg=True) requires f_bg with the same shape as f_fg.")
            if f_bg.shape != f_fg.shape:
                raise ValueError(f"f_bg shape {tuple(f_bg.shape)} must match f_fg {tuple(f_fg.shape)}.")
            fuse = self.fg_bg_fuse
            assert fuse is not None
            frame_input = fuse(torch.cat([f_fg, f_bg], dim=-1))
        else:
            frame_input = f_fg
        if stats is not None:
            frame_input = frame_input + self.frame_aux_proj(stats)
        return frame_input, stats

    def forward(
        self,
        f_fg: torch.Tensor,
        s_sem: torch.Tensor,
        f_q: torch.Tensor,
        frame_patch_stats: Optional[torch.Tensor] = None,
        f_bg: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """融合图传播与 query 门控，得到 ``H_graph``、``S_graph``、``A_event`` 等字典。"""
        # f_fg: [B, T, D], s_sem: [B, T], f_q: [B, D]
        b, t, _ = f_fg.shape
        frame_input, frame_patch_stats = self.build_frame_input(f_fg, f_bg, frame_patch_stats)

        e, mem = self.event_pooler(frame_input)  # e: [B, M, D], mem: [T, M]
        event_patch_stats = None
        if frame_patch_stats is not None:
            event_patch_stats = self._pool_event_patch_stats(frame_patch_stats, mem)

        a_temp = self._temporal_adjacency(t, self.event_pooler.window, f_fg.device)  # [T, T]
        a_sem = self._semantic_event_adjacency(e)  # [B, M, M]
        a_causal = self._causal_event_adjacency(e)  # [B, M, M]

        mix_logits = self.mix(f_q)
        lambdas = torch.softmax(mix_logits / self.mix_temperature, dim=-1)  # [B, 2]
        lam_sem = lambdas[:, 0].view(b, 1, 1)
        lam_cau = lambdas[:, 1].view(b, 1, 1)
        a_event = lam_sem * a_sem + lam_cau * a_causal
        a_negative = self._negative_event_adjacency(e) if self.use_sbp else None
        a_event_signed = (
            a_event - self.sbp_negative_weight * a_negative
            if a_negative is not None
            else a_event
        )

        h_frame = self._one_step_message_passing(frame_input, a_temp, self.frame_update)
        if a_negative is not None:
            h_event = self._one_step_signed_message_passing(e, a_event, a_negative, self.event_update)
        else:
            h_event = self._one_step_message_passing(e, a_event, self.event_update)

        # Project event states back to frames via membership.
        h_event_to_frame = torch.matmul(mem.unsqueeze(0), h_event)  # [B, T, D]
        h_graph = self.fuse(torch.cat([h_frame, h_event_to_frame], dim=-1))

        indegree = a_event.sum(dim=-2)   # [B, M]
        outdegree = a_event.sum(dim=-1)  # [B, M]

        m_events = e.shape[1]
        idx = torch.arange(m_events, device=e.device, dtype=e.dtype)
        lam_sem_val = lam_sem.squeeze(-1).squeeze(-1)  # [B]
        lam_cau_val = lam_cau.squeeze(-1).squeeze(-1)  # [B]

        # 理论最大「加权边数」：语义全连接 m_events 条 + 因果上三角，且因果部分与 a_causal 同等指数衰减
        dist_matrix = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        causal_decay_matrix = torch.exp(-dist_matrix / self.causal_decay_tau)
        causal_mask_out = torch.triu(
            torch.ones(m_events, m_events, device=e.device, dtype=causal_decay_matrix.dtype),
            diagonal=1,
        )
        # 与 a_causal 同支撑：出度为行上三角加权和；入度为列上三角加权和
        max_out_causal = (causal_decay_matrix * causal_mask_out).sum(dim=1)  # [M]
        max_in_causal = (causal_decay_matrix * causal_mask_out).sum(dim=0)  # [M]

        valid_out_edges = lam_sem_val.unsqueeze(1) * m_events + lam_cau_val.unsqueeze(1) * max_out_causal.unsqueeze(0)
        valid_in_edges = lam_sem_val.unsqueeze(1) * m_events + lam_cau_val.unsqueeze(1) * max_in_causal.unsqueeze(0)

        out_strength = outdegree / valid_out_edges.clamp_min(1e-3)
        in_strength = indegree / valid_in_edges.clamp_min(1e-3)

        in_norm = self._normalize_score(in_strength)
        out_norm = self._normalize_score(out_strength)

        dir_pref = torch.sigmoid(self.dir_gate(f_q))  # [B, 1]
        oriented = dir_pref * out_norm + (1.0 - dir_pref) * in_norm

        bridge = torch.sqrt((in_strength.clamp_min(0.0) + 1e-3) * (out_strength.clamp_min(0.0) + 1e-3))
        bridge_norm = self._normalize_score(bridge)
        event_score = oriented + self.bridge_weight * bridge_norm

        event_score = event_score * getattr(self, "score_scale", 1.0) + getattr(self, "score_bias", 0.0)
        event_patch_bias = None
        if event_patch_stats is not None:
            event_patch_bias = torch.tanh(
                torch.nan_to_num(
                    self.event_aux_gate(event_patch_stats).squeeze(-1),
                    nan=0.0,
                    posinf=5.0,
                    neginf=-5.0,
                )
            )
            event_score = event_score + event_patch_bias

        centrality_on_frames = torch.matmul(mem.unsqueeze(0), event_score.unsqueeze(-1)).squeeze(-1)  # [B, T]
        s_graph = self.beta1 * s_sem + self.beta2 * centrality_on_frames
        frame_patch_bias = None
        if frame_patch_stats is not None:
            frame_patch_bias = torch.tanh(
                torch.nan_to_num(
                    self.frame_aux_gate(frame_patch_stats).squeeze(-1),
                    nan=0.0,
                    posinf=5.0,
                    neginf=-5.0,
                )
            )
            s_graph = s_graph + 0.1 * frame_patch_bias

        out = {
            "H_graph": h_graph,
            "H_event": h_event,
            "E": e,
            "A_temp": a_temp.unsqueeze(0).expand(b, -1, -1),
            "A_sem": a_sem,
            "A_causal": a_causal,
            "A_event": a_event,
            "A_event_signed": a_event_signed,
            "S_graph": s_graph,
            "S_event": event_score,
            "centrality_on_frames": centrality_on_frames,
            "in_strength": in_strength,
            "out_strength": out_strength,
            "in_norm": in_norm,
            "out_norm": out_norm,
            "bridge": bridge,
            "bridge_norm": bridge_norm,
            "dir_pref": dir_pref.squeeze(-1),
            "membership": mem.unsqueeze(0).expand(b, -1, -1),
            "mix_logits": mix_logits,
            "mix_lambdas": lambdas,
        }
        if a_negative is not None:
            out["A_negative"] = a_negative
            out["sbp_negative_strength"] = a_negative.mean(dim=(1, 2))
        if event_patch_stats is not None:
            out["event_patch_stats"] = event_patch_stats
        if frame_patch_bias is not None:
            out["frame_patch_bias"] = frame_patch_bias
        if event_patch_bias is not None:
            out["event_patch_bias"] = event_patch_bias
        return out
