from __future__ import annotations

from typing import Dict

import torch


def bernoulli_logprob(probs: torch.Tensor, actions: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = probs.clamp(min=eps, max=1 - eps)
    return (actions * probs.log() + (1 - actions) * (1 - probs).log()).sum(dim=-1)


def compute_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
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
    ratio = torch.exp(logprob_new - logprob_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    return -torch.min(unclipped, clipped).mean()


def kl_to_reference(logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    q = torch.sigmoid(ref_logits).clamp(1e-6, 1 - 1e-6)
    kl = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
    return kl.mean()


def grpo_objective(
    probs_new: torch.Tensor,
    probs_old: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    logits_new: torch.Tensor,
    logits_ref: torch.Tensor,
    beta_kl: float,
    clip_eps: float = 0.2,
) -> Dict[str, torch.Tensor]:
    logprob_new = bernoulli_logprob(probs_new, actions)
    logprob_old = bernoulli_logprob(probs_old, actions)

    loss_pg = grpo_clip_loss(logprob_new, logprob_old, advantages, clip_eps=clip_eps)
    loss_kl = kl_to_reference(logits_new, logits_ref)
    loss = loss_pg + beta_kl * loss_kl

    return {
        "loss_rl": loss,
        "loss_pg": loss_pg,
        "loss_kl": loss_kl,
        "logprob_new": logprob_new,
        "logprob_old": logprob_old,
    }
