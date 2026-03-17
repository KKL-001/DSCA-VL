from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def build_membership_matrix(t: int, window: int, stride: int, device: torch.device) -> torch.Tensor:
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
    def __init__(self, dim: int, window: int = 8, stride: int = 4, sigma_temp: float = 3.0, beta1: float = 0.5, beta2: float = 0.5):
        super().__init__()
        self.dim = dim
        self.sigma_temp = sigma_temp
        self.beta1 = beta1
        self.beta2 = beta2

        self.event_pooler = EventPooler(dim=dim, window=window, stride=stride)
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)

        self.mix = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
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

    def _temporal_adjacency(self, t: int, window: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(t, device=device)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs().float()
        mask = (dist < float(window)).float()
        a = torch.exp(-(dist ** 2) / (2 * (self.sigma_temp ** 2))) * mask
        a.fill_diagonal_(1.0)
        return a

    def _semantic_event_adjacency(self, e: torch.Tensor) -> torch.Tensor:
        # e: [B, M, D]
        e_norm = F.normalize(e, dim=-1)
        sim = torch.matmul(e_norm, e_norm.transpose(1, 2))
        return F.relu(sim)

    def _causal_event_adjacency(self, e: torch.Tensor) -> torch.Tensor:
        q = self.w_q(e)
        k = self.w_k(e)
        a = torch.sigmoid(torch.matmul(q, k.transpose(1, 2)) / (self.dim ** 0.5))
        m = a.shape[-1]
        causal_mask = torch.triu(torch.ones(m, m, device=e.device), diagonal=1)
        return a * causal_mask.unsqueeze(0)

    def _one_step_message_passing(self, x: torch.Tensor, a: torch.Tensor, updater: nn.Module) -> torch.Tensor:
        # x: [B, N, D], a: [B, N, N] or [N, N]
        if a.dim() == 2:
            a = a.unsqueeze(0).expand(x.size(0), -1, -1)
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        msg = torch.matmul(a, x)
        return updater(msg)

    def forward(self, f_fg: torch.Tensor, s_sem: torch.Tensor, f_q: torch.Tensor) -> Dict[str, torch.Tensor]:
        # f_fg: [B, T, D], s_sem: [B, T], f_q: [B, D]
        b, t, _ = f_fg.shape
        e, mem = self.event_pooler(f_fg)  # e: [B, M, D], mem: [T, M]

        a_temp = self._temporal_adjacency(t, self.event_pooler.window, f_fg.device)  # [T, T]
        a_sem = self._semantic_event_adjacency(e)  # [B, M, M]
        a_causal = self._causal_event_adjacency(e)  # [B, M, M]

        lambdas = torch.softmax(self.mix(f_q), dim=-1)  # [B, 2]
        lam_sem = lambdas[:, 0].view(b, 1, 1)
        lam_cau = lambdas[:, 1].view(b, 1, 1)
        a_event = lam_sem * a_sem + lam_cau * a_causal

        h_frame = self._one_step_message_passing(f_fg, a_temp, self.frame_update)
        h_event = self._one_step_message_passing(e, a_event, self.event_update)

        # Project event states back to frames via membership.
        h_event_to_frame = torch.matmul(mem.unsqueeze(0), h_event)  # [B, T, D]
        h_graph = self.fuse(torch.cat([h_frame, h_event_to_frame], dim=-1))

        indegree = a_event.sum(dim=-2)  # [B, M]
        centrality_on_frames = torch.matmul(mem.unsqueeze(0), indegree.unsqueeze(-1)).squeeze(-1)  # [B, T]
        s_graph = self.beta1 * s_sem + self.beta2 * centrality_on_frames

        return {
            "H_graph": h_graph,
            "E": e,
            "A_temp": a_temp.unsqueeze(0).expand(b, -1, -1),
            "A_event": a_event,
            "S_graph": s_graph,
            "membership": mem.unsqueeze(0).expand(b, -1, -1),
        }
