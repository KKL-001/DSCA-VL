"""Feature-side helpers for lightweight pretrained and EVEREST-style variants."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_disparity_scores(features: torch.Tensor) -> torch.Tensor:
    """Score frames by adjacent feature disparity as a lightweight EVEREST proxy."""
    if features.ndim != 3:
        raise ValueError(f"features must be [B,T,D], got {tuple(features.shape)}")
    if features.shape[1] <= 1:
        return torch.ones(features.shape[:2], device=features.device, dtype=features.dtype)
    x = F.normalize(features, dim=-1)
    delta = 1.0 - (x[:, 1:] * x[:, :-1]).sum(dim=-1)
    first = delta[:, :1]
    return torch.cat([first, delta], dim=1).clamp_min(0.0)


def top_disparity_mask(features: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Return a boolean frame mask that keeps high-disparity frames."""
    scores = temporal_disparity_scores(features)
    b, t = scores.shape
    k = max(1, min(t, int(round(t * float(keep_ratio)))))
    idx = torch.topk(scores, k=k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask
