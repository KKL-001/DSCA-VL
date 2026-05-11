"""Post-processing operators for R18+ frame/event selection variants."""
from __future__ import annotations

import torch


def temporal_nms_1d(indices: torch.Tensor, scores: torch.Tensor, min_gap: int = 1, max_keep: int = 0) -> torch.Tensor:
    """Suppress lower-scored selected frames that are too close in time.

    ``min_gap`` is inclusive: with ``min_gap=2``, frames at distance 1 or 2 from
    a higher-scored frame are suppressed. Returned indices are chronological.
    """
    if indices.numel() == 0:
        return indices.to(dtype=torch.long)
    if indices.shape != scores.shape:
        raise ValueError("indices and scores must have the same shape")

    order = torch.argsort(scores, descending=True)
    kept: list[int] = []
    gap = max(int(min_gap), 0)
    limit = int(max_keep) if max_keep and max_keep > 0 else indices.numel()

    for pos in order.tolist():
        idx = int(indices[pos].item())
        if all(abs(idx - prev) > gap for prev in kept):
            kept.append(idx)
            if len(kept) >= limit:
                break

    return torch.tensor(sorted(kept), device=indices.device, dtype=torch.long)
