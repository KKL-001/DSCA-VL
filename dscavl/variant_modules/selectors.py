"""Event-aware frame selectors inspired by EFS and hierarchical keyframe methods."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def event_anchor_select(
    frame_scores: torch.Tensor,
    event_ids: torch.Tensor,
    features: torch.Tensor,
    budget: int,
    diversity_weight: float = 0.3,
) -> torch.Tensor:
    """Select per-event anchors first, then refine with a simple adaptive MMR score."""
    if frame_scores.ndim != 1 or event_ids.ndim != 1 or features.ndim != 2:
        raise ValueError("expected frame_scores [T], event_ids [T], features [T,D]")
    if frame_scores.shape[0] != event_ids.shape[0] or features.shape[0] != frame_scores.shape[0]:
        raise ValueError("all inputs must have the same frame count")

    t = int(frame_scores.numel())
    k = max(1, min(int(budget), t))
    selected: list[int] = []

    for event in torch.unique(event_ids, sorted=True).tolist():
        mask = event_ids == int(event)
        frame_idx = mask.nonzero(as_tuple=False).flatten()
        if frame_idx.numel() == 0:
            continue
        local = frame_scores[frame_idx]
        selected.append(int(frame_idx[torch.argmax(local)].item()))
        if len(selected) >= k:
            break

    feat_n = F.normalize(features, dim=-1)
    diversity = max(0.0, float(diversity_weight))
    while len(selected) < k:
        candidates = [idx for idx in range(t) if idx not in selected]
        if not candidates:
            break
        cand_t = torch.tensor(candidates, device=features.device, dtype=torch.long)
        if selected:
            sel_t = torch.tensor(selected, device=features.device, dtype=torch.long)
            sim = torch.matmul(feat_n[cand_t], feat_n[sel_t].transpose(0, 1)).max(dim=-1).values
        else:
            sim = torch.zeros(cand_t.numel(), device=features.device, dtype=frame_scores.dtype)
        mmr = frame_scores[cand_t] - diversity * sim
        selected.append(int(cand_t[torch.argmax(mmr)].item()))

    return torch.tensor(sorted(selected), device=features.device, dtype=torch.long)
