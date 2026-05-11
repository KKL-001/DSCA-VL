"""RL 奖励拼装：帧图构建、因果链连通、稀疏性、事件覆盖与可选 patch 诊断项。"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch


def accuracy_reward(
    predicted_answer: str,
    target_answer: str,
    judge_fn: Optional[Callable[[str, str], float]] = None,
) -> float:
    """字符串级准确率；可注入自定义 ``judge_fn`` 做模糊匹配。"""
    if judge_fn is not None:
        return float(judge_fn(predicted_answer, target_answer))
    return 1.0 if predicted_answer.strip().lower() == target_answer.strip().lower() else 0.0


# ======================================================================
# Frame-level graph construction (unchanged interface)
# ======================================================================

def build_frame_graph(
    a_temp: torch.Tensor,
    a_event: torch.Tensor | None,
    membership: torch.Tensor | None,
) -> torch.Tensor:
    """将事件邻接提升到帧级并与时序邻接相加，再 min-max 归一化到 [0,1]。"""
    # a_temp: [B, T, T], a_event: [B, M, M], membership: [B, T, M]
    if a_event is None or membership is None:
        a_frame = a_temp
        a_min = a_frame.amin(dim=(1, 2), keepdim=True)
        a_max = a_frame.amax(dim=(1, 2), keepdim=True)
        return (a_frame - a_min) / (a_max - a_min + 1e-6)

    lifted = torch.matmul(torch.matmul(membership, a_event), membership.transpose(1, 2))
    a_frame = a_temp + lifted
    a_min = a_frame.amin(dim=(1, 2), keepdim=True)
    a_max = a_frame.amax(dim=(1, 2), keepdim=True)
    return (a_frame - a_min) / (a_max - a_min + 1e-6)


def _resolve_event_selection(
    membership: torch.Tensor | None,
    actions: torch.Tensor,
    event_actions: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """得到每个事件是否被选中：优先用语义 ``event_actions``，否则由帧动作与 membership 推导。"""
    if event_actions is not None:
        return torch.nan_to_num(
            event_actions,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
    if membership is None:
        return None
    return torch.matmul(
        membership.transpose(1, 2),
        torch.nan_to_num(actions, nan=0.0).unsqueeze(-1),
    ).squeeze(-1)


# ======================================================================
# Causal chain connectivity (replaces old density-based connectivity)
# ======================================================================

def causal_chain_connectivity(
    a_event: torch.Tensor | None,
    membership: torch.Tensor | None,
    actions: torch.Tensor,
    event_actions: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """对已选事件子图，只统计 ``a_event`` 上三角（因果向）边密度并归一化。"""
    actions = torch.nan_to_num(actions, nan=0.0).clamp(0.0, 1.0)
    selected_events = _resolve_event_selection(membership, actions, event_actions=event_actions)
    if a_event is None or selected_events is None:
        return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

    b = a_event.shape[0]
    m = a_event.shape[1]
    out = torch.zeros(b, device=a_event.device, dtype=a_event.dtype)

    causal_mask = torch.triu(torch.ones(m, m, device=a_event.device), diagonal=1)

    for i in range(b):
        sel_idx = (selected_events[i] > 0.5).nonzero(as_tuple=False).flatten()

        if sel_idx.numel() < 2:
            out[i] = 0.0
            continue

        sub_a = a_event[i].index_select(0, sel_idx).index_select(1, sel_idx)
        sub_mask = causal_mask.index_select(0, sel_idx).index_select(1, sel_idx)
        causal_edges = sub_a * sub_mask
        n = sel_idx.numel()
        max_edges = n * (n - 1) / 2.0
        out[i] = causal_edges.sum() / (max_edges + eps)

    return out


def event_coverage_reward(
    membership: torch.Tensor | None,
    actions: torch.Tensor,
    event_actions: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """被至少一帧命中的事件数 / 总事件数，鼓励时间覆盖。"""
    actions = torch.nan_to_num(actions, nan=0.0).clamp(0.0, 1.0)
    selected_events = _resolve_event_selection(membership, actions, event_actions=event_actions)
    if selected_events is None:
        return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

    b = selected_events.shape[0]
    m = selected_events.shape[1]
    out = torch.zeros(b, device=selected_events.device, dtype=selected_events.dtype)
    for i in range(b):
        covered = (selected_events[i] > 0.5).sum().float()
        out[i] = covered / (float(m) + eps)
    return out


# ======================================================================
# Legacy connectivity_reward kept for backward compatibility but no
# longer used in the default reward_bundle.
# ======================================================================

def connectivity_reward(a_frame: torch.Tensor, actions: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """（遗留）选中帧在 ``a_frame`` 上的无向密度；默认 bundle 已改用因果连通。"""
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
        density = (sub.sum() - sub.diag().sum()) / (idx.numel() * (idx.numel() - 1) + eps)
        out[i] = density
    return out


def sparsity_reward(actions: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对选中比例取负对数，鼓励少选帧。"""
    actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    ratio = actions.mean(dim=-1).clamp_min(0.0)
    return -(ratio + eps).log()


def patch_diagnostic_reward(
    frame_patch_stats: torch.Tensor | None,
    actions: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    """按配置加权 patch 统计得到「选中帧质量」；默认权重为 0 时不进入总奖励。"""
    if frame_patch_stats is None or frame_patch_stats.ndim != 3:
        return None

    stats = torch.nan_to_num(frame_patch_stats, nan=0.0, posinf=0.0, neginf=0.0)
    actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if stats.shape[:2] != actions.shape:
        return None

    dispersion = stats[..., 0].clamp(0.0, 1.0)
    concentration = stats[..., 1].clamp(0.0, 1.0)
    saliency = stats[..., 2].clamp(0.0, 1.0)
    recon_spread = stats[..., 3].clamp(0.0, 1.0)
    proto_entropy = stats[..., 4].clamp(0.0, 1.0)

    quality = (
        0.30 * concentration
        + 0.30 * saliency
        + 0.15 * (1.0 - dispersion)
        + 0.15 * (1.0 - proto_entropy)
        + 0.10 * (1.0 - recon_spread)
    )
    denom = actions.sum(dim=-1).clamp_min(1.0)
    return (quality * actions).sum(dim=-1) / (denom + eps)


def composite_reward(
    reward_acc: torch.Tensor,
    reward_cons: torch.Tensor,
    reward_sparse: torch.Tensor,
    omega_cons: float,
    omega_sparse: float,
    reward_coverage: torch.Tensor | None = None,
    omega_coverage: float = 0.0,
    reward_patch_diag: torch.Tensor | None = None,
    omega_patch_diag: float = 0.0,
) -> torch.Tensor:
    """线性加权各子奖励；coverage/patch 项系数为 0 时跳过。"""
    r = reward_acc + omega_cons * reward_cons + omega_sparse * reward_sparse
    if reward_coverage is not None and omega_coverage > 0:
        r = r + omega_coverage * reward_coverage
    if reward_patch_diag is not None and omega_patch_diag > 0:
        r = r + omega_patch_diag * reward_patch_diag
    return r


# ======================================================================
# Public bundle (default interface for training scripts)
# ======================================================================

def reward_bundle(
    reward_acc: torch.Tensor,
    a_temp: torch.Tensor,
    a_event: torch.Tensor | None,
    membership: torch.Tensor | None,
    actions: torch.Tensor,
    omega_cons: float,
    omega_sparse: float,
    event_actions: torch.Tensor | None = None,
    omega_coverage: float = 0.2,
    frame_patch_stats: torch.Tensor | None = None,
    omega_patch_diag: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """训练脚本调用的统一出口：返回分项奖励、总奖励与 ``A_frame``（供日志/可视化）。"""
    a_frame = build_frame_graph(a_temp, a_event, membership)

    reward_cons = causal_chain_connectivity(
        a_event,
        membership,
        actions,
        event_actions=event_actions,
    )
    reward_sparse = sparsity_reward(actions)
    reward_cover = event_coverage_reward(
        membership,
        actions,
        event_actions=event_actions,
    )
    reward_patch = patch_diagnostic_reward(frame_patch_stats, actions)

    reward_total = composite_reward(
        reward_acc, reward_cons, reward_sparse,
        omega_cons, omega_sparse,
        reward_coverage=reward_cover,
        omega_coverage=omega_coverage,
        reward_patch_diag=reward_patch,
        omega_patch_diag=omega_patch_diag,
    )

    out = {
        "R_acc": reward_acc,
        "R_cons": reward_cons,
        "R_sparse": reward_sparse,
        "R_coverage": reward_cover,
        "R_total": reward_total,
        "A_frame": a_frame,
    }
    if reward_patch is not None:
        out["R_patch_diag"] = reward_patch
    return out
