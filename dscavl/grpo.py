"""GRPO：Bernoulli 策略对数概率、组内优势、PPO 式 clip 与对参考策略的 KL。"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


def bernoulli_logprob(probs: torch.Tensor, actions: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """逐帧 Bernoulli 对数似然并在时间维求和，得到轨迹 log prob。"""
    probs = probs.clamp(min=eps, max=1 - eps)

    return (actions * probs.log() + (1 - actions) * (1 - probs).log()).sum(dim=-1)


def compute_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """组内（最后一维 G）减均值除标准差，得到每条轨迹的优势。"""
    # rewards: [B, G]
    mu = rewards.mean(dim=-1, keepdim=True)
    std = (rewards - mu).pow(2).mean(dim=-1, keepdim=True).sqrt()
    return (rewards - mu) / (std + eps)


def grpo_clip_loss(
    logprob_new: torch.Tensor,
    logprob_old: torch.Tensor,
    adv: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """重要性采样比 clip 前后取 min，再对 batch 求均值损失（负号在调用方语义中体现）。"""
    ratio = torch.exp(logprob_new - logprob_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    return -torch.min(unclipped, clipped).mean()


def kl_to_reference(logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """逐维 Bernoulli KL(current || ref) 的均值。"""
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    q = torch.sigmoid(ref_logits).clamp(1e-6, 1 - 1e-6)
    kl = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
    return kl.mean()


def grpo_objective(
    logprob_new: torch.Tensor,
    logprob_old: torch.Tensor,
    advantages: torch.Tensor,
    beta_kl: float,
    kl_logits_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    clip_eps: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """组合 clip 策略梯度损失与 KL 正则。

    ``logprob_*``：形状 ``[B]``，与 ``trajectory_log_prob`` 一致。
    ``kl_logits_pairs``：若干 ``(logits, ref_logits)``，通常帧一对 +（若有）事件一对。
    """
    loss_pg = grpo_clip_loss(logprob_new, logprob_old, advantages, clip_eps=clip_eps)
    if kl_logits_pairs:
        loss_kl = sum(
            kl_to_reference(a, b) for a, b in kl_logits_pairs
        ) / float(len(kl_logits_pairs))
    else:
        loss_kl = torch.zeros((), device=logprob_new.device, dtype=logprob_new.dtype)
    loss = loss_pg + beta_kl * loss_kl

    return {
        "loss_rl": loss,
        "loss_pg": loss_pg,
        "loss_kl": loss_kl,
        "logprob_new": logprob_new,
        "logprob_old": logprob_old,
    }


def cgrm_policy_kw_from_output(out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor | None]:
    """从 ``DSCAVL`` backbone 输出构造 ``FramePolicyHead`` 的事件与辅助张量关键字。"""
    return {
        "h_event": out.get("H_event"),
        "s_event": out.get("S_event"),
        "membership": out.get("membership"),
        "frame_aux": out.get("frame_patch_stats"),
        "event_aux": out.get("event_patch_stats"),
    }


def compute_stage2_grpo_terms(
    model: nn.Module,
    old_policy: nn.Module,
    ref_policy: nn.Module,
    backbone_out: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    event_actions: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """在当前图状态下：轨迹对数似然（新/旧策略）及相对参考策略的 KL 张量对列表。"""
    pol = model.policy
    kw = cgrm_policy_kw_from_output(backbone_out)
    lp_new = pol.trajectory_log_prob(
        backbone_out["H_graph"],
        backbone_out["f_q"],
        backbone_out["S_graph"],
        actions,
        event_actions=event_actions,
        **kw,
    )
    with torch.no_grad():
        lp_old = old_policy.trajectory_log_prob(
            backbone_out["H_graph"],
            backbone_out["f_q"],
            backbone_out["S_graph"],
            actions,
            event_actions=event_actions,
            **kw,
        )
        kl_pairs = pol.kl_logits_pairs_with(
            ref_policy,
            backbone_out["H_graph"],
            backbone_out["f_q"],
            backbone_out["S_graph"],
            **kw,
        )
    return lp_new, lp_old, kl_pairs
