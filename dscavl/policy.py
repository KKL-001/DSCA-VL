from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class FramePolicyHead(nn.Module):
    def __init__(self, dim: int, gamma: float = 1.0, hidden_dim: int = 512):
        super().__init__()
        self.gamma = gamma
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_graph: torch.Tensor,
        f_q: torch.Tensor,
        s_graph: torch.Tensor,
        sample: bool = True,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        b, t, _ = h_graph.shape
        f_q_exp = f_q.unsqueeze(1).expand(b, t, -1)
        policy_in = torch.cat([h_graph, f_q_exp], dim=-1)
        logits_raw = self.mlp(policy_in).squeeze(-1) + self.gamma * s_graph
        logits = torch.nan_to_num(logits_raw, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        probs = torch.sigmoid(logits)
        probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-6, 1.0 - 1e-6)

        if sample:
            actions = torch.bernoulli(probs)
        else:
            actions = (probs >= threshold).to(probs.dtype)

        return {
            "logits": logits,
            "probs": probs,
            "actions": actions,
        }
