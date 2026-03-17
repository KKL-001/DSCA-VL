from __future__ import annotations

import re
from typing import Sequence

import torch
import torch.nn.functional as F


_OPTION_LABEL_RE = re.compile(r"^\s*([A-Z])\s*[.:)]")
_ANSWER_LABEL_RE = re.compile(r"([A-Z])")


def normalize_option_text(option: str) -> tuple[str, str]:
    label_match = _OPTION_LABEL_RE.match(option)
    label = label_match.group(1) if label_match else ""
    text = option
    if label_match:
        text = option[label_match.end() :].strip()
    return label, text if text else option.strip()


def extract_answer_label(answer: str) -> str:
    match = _ANSWER_LABEL_RE.search(answer.strip().upper())
    return match.group(1) if match else answer.strip().upper()


@torch.no_grad()
def score_mcq_options(model, out: dict, options: Sequence[str], tokenizer, device: torch.device) -> tuple[list[str], torch.Tensor]:
    if not options:
        return [], torch.empty(0, device=device)

    actions = out["actions"][0] > 0.5
    if actions.any():
        evidence = out["F_fg"][0][actions].mean(dim=0)
    else:
        idx = out["probs"][0].argmax()
        evidence = out["F_fg"][0, idx]

    labels = []
    texts = []
    for idx, option in enumerate(options):
        label, text = normalize_option_text(option)
        if not label:
            label = chr(ord("A") + idx)
        labels.append(label)
        texts.append(text)

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    option_ids = encoded["input_ids"].to(device)
    option_mask = encoded.get("attention_mask")
    if option_mask is not None:
        option_mask = option_mask.to(device)

    option_emb = model.text_encoder(option_ids, attention_mask=option_mask)
    evidence = F.normalize(evidence.unsqueeze(0), dim=-1)
    option_emb = F.normalize(option_emb, dim=-1)
    scores = torch.matmul(option_emb, evidence.transpose(0, 1)).squeeze(-1)
    return labels, scores


@torch.no_grad()
def compute_mcq_exact_reward(model, out: dict, options: Sequence[str], answer: str, tokenizer, device: torch.device) -> tuple[torch.Tensor, str, str, torch.Tensor]:
    labels, scores = score_mcq_options(model, out, options, tokenizer, device)
    if not labels:
        zero = torch.zeros(1, device=device)
        return zero, "", extract_answer_label(answer), scores

    pred_idx = int(scores.argmax().item())
    pred_label = labels[pred_idx]
    gt_label = extract_answer_label(answer)
    reward = 1.0 if pred_label == gt_label else 0.0
    return torch.tensor([reward], device=device), pred_label, gt_label, scores