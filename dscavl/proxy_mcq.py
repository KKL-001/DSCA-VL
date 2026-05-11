"""MCQ 代理打分：从模型输出取证据向量，与选项文本嵌入算相似度，得到选择题奖励。"""
from __future__ import annotations

import re
from typing import Sequence

import torch
import torch.nn.functional as F


_OPTION_LABEL_RE = re.compile(r"^\s*([A-Z])\s*[.:)]")
_ANSWER_LABEL_RE = re.compile(r"([A-Z])")


def normalize_option_text(option: str) -> tuple[str, str]:
    """拆分选项前缀字母与正文；无前缀则 label 为空。"""
    label_match = _OPTION_LABEL_RE.match(option)
    label = label_match.group(1) if label_match else ""
    text = option
    if label_match:
        text = option[label_match.end() :].strip()
    return label, text if text else option.strip()


def extract_answer_label(answer: str) -> str:
    """从标准答案串中提取首个大写字母标签。"""
    match = _ANSWER_LABEL_RE.search(answer.strip().upper())
    return match.group(1) if match else answer.strip().upper()


def _resolve_frame_features(out: dict) -> torch.Tensor:
    """兼容 frame 级 ``F_fg`` 或 patch 级 ``F_fg_patch``+``patch_weights`` 聚合。"""
    f_fg = out.get("F_fg")
    if isinstance(f_fg, torch.Tensor) and f_fg.ndim == 3:
        return torch.nan_to_num(f_fg[0], nan=0.0, posinf=0.0, neginf=0.0)

    f_fg_patch = out.get("F_fg_patch")
    patch_weights = out.get("patch_weights")
    if (
        isinstance(f_fg_patch, torch.Tensor)
        and isinstance(patch_weights, torch.Tensor)
        and f_fg_patch.ndim == 4
        and patch_weights.ndim == 3
    ):
        frame_features = (patch_weights[0].unsqueeze(-1) * f_fg_patch[0]).sum(dim=1)
        return torch.nan_to_num(frame_features, nan=0.0, posinf=0.0, neginf=0.0)

    raise KeyError("MCQ proxy expects `F_fg` or (`F_fg_patch`, `patch_weights`) in model output.")


def _select_evidence(out: dict) -> torch.Tensor:
    """优先用 ``actions`` 选中帧加权平均证据；否则用 ``probs`` argmax 或首帧。"""
    frame_features = _resolve_frame_features(out)
    probs = out.get("probs")
    actions = out.get("actions")

    action_mask = None
    if isinstance(actions, torch.Tensor) and actions.ndim >= 2:
        action_mask = actions[0] > 0.5

    if action_mask is not None and action_mask.any():
        if isinstance(probs, torch.Tensor) and probs.ndim >= 2 and probs.shape[-1] == frame_features.shape[0]:
            weights = probs[0, action_mask].clamp_min(1e-6)
            selected = frame_features[action_mask]
            return (selected * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
        return frame_features[action_mask].mean(dim=0)

    if isinstance(probs, torch.Tensor) and probs.ndim >= 2 and probs.shape[-1] == frame_features.shape[0]:
        idx = int(probs[0].argmax().item())
    else:
        idx = 0
    return frame_features[idx]


@torch.no_grad()
def score_mcq_options(model, out: dict, options: Sequence[str], tokenizer, device: torch.device) -> tuple[list[str], torch.Tensor]:
    """对各选项编码后与证据向量点积，返回 (标签列表, 分数向量)。"""
    if not options:
        return [], torch.empty(0, device=device)

    evidence = _select_evidence(out)

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
    """最高分选项标签与 GT 是否一致，返回 {0,1} 标量奖励及预测/标签/全部分数。"""
    labels, scores = score_mcq_options(model, out, options, tokenizer, device)
    if not labels:
        zero = torch.zeros(1, device=device)
        return zero, "", extract_answer_label(answer), scores

    pred_idx = int(scores.argmax().item())
    pred_label = labels[pred_idx]
    gt_label = extract_answer_label(answer)
    reward = 1.0 if pred_label == gt_label else 0.0
    return torch.tensor([reward], device=device), pred_label, gt_label, scores