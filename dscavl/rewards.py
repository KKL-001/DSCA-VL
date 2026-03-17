from __future__ import annotations

from typing import Callable, Dict, Optional

import torch


def accuracy_reward(
    predicted_answer: str,
    target_answer: str,
    judge_fn: Optional[Callable[[str, str], float]] = None,
) -> float:
    if judge_fn is not None:
        return float(judge_fn(predicted_answer, target_answer))
    # Default fallback for closed-set reproduction.
    return 1.0 if predicted_answer.strip().lower() == target_answer.strip().lower() else 0.0


def connectivity_reward(a_frame: torch.Tensor, actions: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # a_frame: [B, T, T], actions: [B, T] in {0,1}
    actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    b, _, _ = a_frame.shape
    out = torch.zeros(b, device=a_frame.device, dtype=a_frame.dtype)

    for i in range(b):
        sel = actions[i] > 0.5
        idx = sel.nonzero(as_tuple=False).flatten()
        if idx.numel() < 2:
            out[i] = 0.0
            continue
        sub = a_frame[i].index_select(0, idx).index_select(1, idx)
        # exclude diagonal self-connections
        density = (sub.sum() - sub.diag().sum()) / (idx.numel() * (idx.numel() - 1) + eps)
        out[i] = density
    return out


def sparsity_reward(actions: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # actions: [B, T]
    actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    ratio = actions.mean(dim=-1).clamp_min(0.0)
    return -(ratio + eps).log()


def composite_reward(
    reward_acc: torch.Tensor,
    reward_cons: torch.Tensor,
    reward_sparse: torch.Tensor,
    omega_cons: float,
    omega_sparse: float,
) -> torch.Tensor:
    return reward_acc + omega_cons * reward_cons + omega_sparse * reward_sparse


def build_frame_graph(a_temp: torch.Tensor, a_event: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
    # a_temp: [B, T, T], a_event: [B, M, M], membership: [B, T, M]
    # Lift event edges to frames: mem * A_event * mem^T
    lifted = torch.matmul(torch.matmul(membership, a_event), membership.transpose(1, 2))
    a_frame = a_temp + lifted
    a_min = a_frame.amin(dim=(1, 2), keepdim=True)
    a_max = a_frame.amax(dim=(1, 2), keepdim=True)
    return (a_frame - a_min) / (a_max - a_min + 1e-6)


def reward_bundle(
    reward_acc: torch.Tensor,
    a_temp: torch.Tensor,
    a_event: torch.Tensor,
    membership: torch.Tensor,
    actions: torch.Tensor,
    omega_cons: float,
    omega_sparse: float,
) -> Dict[str, torch.Tensor]:
    a_frame = build_frame_graph(a_temp, a_event, membership)
    reward_cons = connectivity_reward(a_frame, actions)
    reward_sparse = sparsity_reward(actions)
    reward_total = composite_reward(reward_acc, reward_cons, reward_sparse, omega_cons, omega_sparse)

    return {
        "R_acc": reward_acc,
        "R_cons": reward_cons,
        "R_sparse": reward_sparse,
        "R_total": reward_total,
        "A_frame": a_frame,
    }
