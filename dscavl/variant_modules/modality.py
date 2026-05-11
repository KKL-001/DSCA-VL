"""UMS/UniMoS-inspired lightweight modality separation blocks.

Reference: https://github.com/TL-UESTC/UniMoS and the Split-to-Merge / UniMoS
papers. This is a clean-room, project-native adapter: it keeps the paper's
linear LAC/VAC separation and ensemble idea, but operates on DSCA-VL tensors.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class ModalitySeparationAdapter(nn.Module):
    """Split features into language-associated and vision-associated components."""

    def __init__(self, dim: int, bottleneck_dim: int = 256, eps: float = 1e-6):
        super().__init__()
        hidden = max(1, min(int(bottleneck_dim), int(dim)))
        self.eps = eps
        self.lac = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.vac = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.query_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.mdi_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, features: torch.Tensor, query: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return separated components, sample MDI score, MaE gate, and orth loss."""
        if features.ndim != 3:
            raise ValueError(f"features must be [B,T,D], got {tuple(features.shape)}")
        if query.ndim != 2 or query.shape[0] != features.shape[0] or query.shape[-1] != features.shape[-1]:
            raise ValueError("query must be [B,D] and match features batch/dim")

        lac = self.lac(features)
        vac = self.vac(features)
        gate = torch.sigmoid(self.query_gate(query)).view(query.shape[0], 1, 1)
        fused = gate * lac + (1.0 - gate) * vac

        lac_n = F.normalize(lac, dim=-1, eps=self.eps)
        vac_n = F.normalize(vac, dim=-1, eps=self.eps)
        orth_loss = (lac_n * vac_n).sum(dim=-1).abs().mean()

        pooled = fused.mean(dim=1)
        mdi = torch.sigmoid(self.mdi_head(pooled)).squeeze(-1)
        return {
            "lac": lac,
            "vac": vac,
            "mae_gate": gate,
            "mdi": mdi,
            "ums_fused": fused,
            "orth_loss": orth_loss,
        }
