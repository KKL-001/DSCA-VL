"""Query-aware routing and test-time sampling helpers for R18+ variants."""
from __future__ import annotations

import torch


def adaptive_budget(query: torch.Tensor, n_frames: int, low: int, high: int) -> torch.Tensor:
    """Map query dispersion to a bounded frame budget."""
    if query.ndim != 2:
        raise ValueError(f"query must be [B,D], got {tuple(query.shape)}")
    lo = max(1, int(low))
    hi = max(lo, min(int(high), int(n_frames)))
    if hi == lo:
        return torch.full((query.shape[0],), lo, device=query.device, dtype=torch.long)

    complexity = query.float().std(dim=-1, unbiased=False)
    scale = complexity / complexity.detach().max().clamp_min(1e-6)
    budget = lo + torch.round(scale * float(hi - lo)).long()
    return budget.clamp(min=lo, max=hi)


def bandit_update(probs: torch.Tensor, selected: torch.Tensor, reward: float, lr: float = 0.1) -> torch.Tensor:
    """One-step multiplicative-weights update for TTA-Vid/MAB-style frame priors."""
    if probs.ndim != 1:
        raise ValueError("probs must be [T]")
    updated = probs.float().clamp_min(1e-8).clone()
    if selected.numel() > 0:
        idx = selected.to(device=probs.device, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < probs.numel())]
        if idx.numel() > 0:
            updated[idx] = updated[idx] * torch.exp(updated.new_tensor(float(lr) * float(reward)))
    return updated / updated.sum().clamp_min(1e-8)
