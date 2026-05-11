"""CGRM 弱监督损失：帧级 ``gt_mask`` 经与 EventPooler 相同的 ``membership`` 池化到事件维，与 ``S_event`` 对齐。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pool_gt_mask_to_events(
    gt_float: torch.Tensor,
    membership: torch.Tensor,
    pool: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """将帧级软/硬标签池化到事件节点。

    ``gt_float``: ``[B, T]``；``membership``: ``[B, T, M]``（与 CGRM 输出一致）。
    返回 ``g_m``: ``[B, M]``，取值约在 ``[0, 1]``（输入为 0/1 时）。

    * ``mean``：\\(g_m[m]=\\sum_t g_t\\,\\mu_{t,m}/\\sum_t\\mu_{t,m}\\)（与 EventPooler 按窗加权平均一致）。
    * ``max``：每个事件在 ``\\mu_{t,m}>0`` 的帧上取 ``g_t`` 的最大值。
    """
    if pool == "mean":
        numer = torch.bmm(gt_float.unsqueeze(1), membership).squeeze(1)
        denom = membership.sum(dim=1).clamp_min(eps)
        return numer / denom
    if pool == "max":
        neg_inf = gt_float.new_full((), float("-inf"))
        masked = torch.where(membership > eps, gt_float.unsqueeze(-1), neg_inf)
        out = masked.max(dim=1).values
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    raise ValueError(f"Unknown event pool mode: {pool!r} (expected 'mean' or 'max').")


def loss_cgrm_weak(
    s_event: torch.Tensor,
    membership: torch.Tensor,
    gt_mask: torch.Tensor,
    *,
    frame_mask: torch.Tensor | None = None,
    event_pool: str = "mean",
    loss_type: str = "bce_logits",
    tv_weight: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """``S_event``（``[B,M]``）与字幕弱标签在事件维上的对齐损失（标量）。

    仅在 ``gt_mask`` 该行至少一帧为真时计入该样本；否则跳过该行。
    若整批无正样本行，返回 ``s_event.sum() * 0``，避免断梯度图且梯度为 0。
    """
    if gt_mask.dtype != torch.bool:
        gt_mask = gt_mask.bool()
    g = gt_mask.to(dtype=s_event.dtype)
    if frame_mask is not None:
        g = g * frame_mask.to(dtype=s_event.dtype)

    row_has_pos = g > 1e-8
    row_has_pos = row_has_pos.any(dim=1)
    if not row_has_pos.any():
        return s_event.sum() * 0.0

    g_m = pool_gt_mask_to_events(g, membership, pool=event_pool, eps=eps)
    g_m = g_m.clamp(0.0, 1.0)
    g_m = torch.nan_to_num(g_m, nan=0.0, posinf=1.0, neginf=0.0)

    if loss_type == "bce_logits":
        # 与 CGRM 中 ``S_event`` 的取值范围解耦：限幅 logits 避免极端处 BCE 反传与下游算子叠加后溢出
        logits = s_event.clamp(-24.0, 24.0)
        per_row = F.binary_cross_entropy_with_logits(logits, g_m, reduction="none").mean(dim=-1)
    elif loss_type == "mse":
        lo = s_event.min(dim=-1, keepdim=True).values
        hi = s_event.max(dim=-1, keepdim=True).values
        pred = (s_event - lo) / (hi - lo).clamp_min(eps)
        per_row = (pred - g_m).pow(2).mean(dim=-1)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r} (expected 'bce_logits' or 'mse').")

    w = row_has_pos.to(dtype=per_row.dtype)
    main = (per_row * w).sum() / w.sum().clamp_min(1.0)

    if tv_weight > 0.0 and s_event.size(1) > 1:
        tv_per_row = (s_event[:, 1:] - s_event[:, :-1]).abs().mean(dim=-1)
        tv = (tv_per_row * w).sum() / w.sum().clamp_min(1.0)
        main = main + tv_weight * tv

    return main
