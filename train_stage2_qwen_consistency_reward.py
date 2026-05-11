"""Stage2 Qwen 闭环 + 一致性/时序附加奖励：在 GRPO 总奖励上叠加 R_conf、R_temporal 等。

继承「从 Stage1 加载 + 可选 DSAM/CGRM 微调」结构；超参通过 CLI omega-* 调节。
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from dscavl import (
    DSCAVL,
    DSCAVLConfig,
    QueryTextEncoder,
    QuestionFeatureDataset,
    variable_feature_collate,
    add_swanlab_train_args,
    resolve_swanlab_project,
    resolve_train_checkpoint_dir,
    swanlab_try_init,
)
from dscavl.grpo import compute_advantages, compute_stage2_grpo_terms, grpo_objective
from dscavl.rewards import reward_bundle
from dscavl.train_utils import (
    add_cgrm_use_f_bg_cli_args,
    merge_cgrm_use_f_bg_cfg,
    model_state_dict_from_stage1_payload,
)
from qwen_reward_utils import QwenAnswerRewarder, resolve_path_from_repo


@dataclass
class RolloutRewardParts:
    """单次 rollout 分解后的标量奖励分量，便于日志。"""
    r_base: float
    r_conf: float
    r_temporal: float
    r_coverage: float
    r_stability: float
    r_total: float


def parse_args() -> argparse.Namespace:
    """一致性/时序/覆盖等附加项权重与数据路径。"""
    parser = argparse.ArgumentParser(
        description="Stage2 GRPO + Qwen closed-loop reward with additional consistency rewards."
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--feature-root", type=str, default="features")
    parser.add_argument("--stage1-checkpoint", type=str, default="output/stage1/checkpoint-best.pt")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory. Default: output/stage2_qwen_consistency_reward/<YYYYMMDD_HHMMSS> under repo root.",
    )
    parser.add_argument("--qwen-model-path", type=str, default="/root/autodl-tmp/Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=32)
    parser.add_argument("--max-selected-frames", type=int, default=8)
    parser.add_argument("--eval-mcq-only", action="store_true")
    parser.add_argument("--finetune-dsam-cgrm", action="store_true")
    parser.add_argument(
        "--omega-conf",
        type=float,
        default=0.10,
        help="Weight of answer-format confidence reward.",
    )
    parser.add_argument(
        "--omega-stability",
        type=float,
        default=0.10,
        help="Weight of group answer-consistency reward.",
    )
    parser.add_argument(
        "--omega-temporal",
        type=float,
        default=0.10,
        help="Weight of temporal span reward.",
    )
    parser.add_argument(
        "--temporal-min-span-sec",
        type=float,
        default=4.0,
        help="Minimum preferred selected-frame time span for temporal reward normalization.",
    )
    parser.add_argument(
        "--omega-coverage",
        type=float,
        default=0.10,
        help="Weight of temporal multi-local coverage reward.",
    )
    parser.add_argument(
        "--coverage-bins",
        type=int,
        default=4,
        help="Number of temporal bins for coverage reward.",
    )
    parser.add_argument(
        "--sanity-steps",
        type=int,
        default=0,
        help="Run only N optimizer steps per epoch for quick sanity validation (<=0 means full epoch).",
    )
    parser.add_argument("--train-batch-size", type=int, default=None, help="Override cfg.train_batch_size.")
    parser.add_argument("--group-size", type=int, default=None, help="Override cfg.group_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override cfg.num_workers.")
    add_swanlab_train_args(parser)
    add_cgrm_use_f_bg_cli_args(parser)
    return parser.parse_args()


def build_tokenizer(cfg: DSCAVLConfig, repo_root: Path):
    """DSCA query tokenizer。"""
    tokenizer_path = repo_root / cfg.text_model_name_or_path
    source = str(tokenizer_path) if tokenizer_path.exists() else cfg.text_model_name_or_path
    if source == "Qwen2-VL-2B-Instruct":
        local_qwen = Path("/root/autodl-tmp/Qwen/Qwen2-VL-2B-Instruct")
        if local_qwen.exists():
            source = str(local_qwen)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataloader(cfg: DSCAVLConfig, tokenizer, data_root: Path, feature_root: Path) -> DataLoader:
    """训练 DataLoader（无字幕）。"""
    dataset = QuestionFeatureDataset(
        data_root=str(data_root),
        feature_root=str(feature_root),
        tokenizer=tokenizer,
        max_length=cfg.query_max_length,
        include_subtitles=False,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=variable_feature_collate,
    )


def load_stage1_weights(
    model: DSCAVL,
    ckpt_path: Path,
    device: torch.device,
    state: dict | None = None,
) -> None:
    """从 Stage1 checkpoint 加载 ``model_state_dict``；可传入已加载的 ``state`` 避免重复读盘。"""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage1 checkpoint not found: {ckpt_path}")
    payload = state if state is not None else torch.load(ckpt_path, map_location=device)
    state_dict = model_state_dict_from_stage1_payload(payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        "[Stage2-Qwen-Consistency] loaded stage1 checkpoint | "
        f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
    )


def set_trainable_params(model: DSCAVL, finetune_dsam_cgrm: bool) -> None:
    """默认可训练仅 policy；可选打开 DSAM/CGRM。"""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.policy.parameters():
        p.requires_grad = True
    if finetune_dsam_cgrm:
        for p in model.dsam.parameters():
            p.requires_grad = True
        for p in model.cgrm.parameters():
            p.requires_grad = True


def _extract_batch_tensors(batch, device: torch.device):
    """query 张量 device 迁移。"""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _format_confidence_reward(pred_label: str, pred_text: str) -> float:
    """鼓励单行单字母答案；冗长自由文本给低分。"""
    if not pred_label:
        return 0.0
    text = pred_text.strip().upper()
    if not text:
        return 0.0
    # Avoid easy saturation to 1.0: only strict one-letter outputs get full score.
    if text == pred_label:
        return 1.0
    if text.startswith(pred_label) and len(text) <= 8:
        return 0.5
    return 0.0


def _temporal_span_reward(
    actions: torch.Tensor,
    timestamps: torch.Tensor | None,
    min_span_sec: float,
) -> float:
    """选中帧时间跨度与 ``min_span_sec`` 的比值截断到 [0,1]。"""
    if timestamps is None or timestamps.numel() == 0:
        return 0.0
    idx = torch.nonzero(actions > 0.5, as_tuple=False).squeeze(-1)
    if idx.numel() <= 1:
        return 0.0
    ts = timestamps.float()
    selected = ts[idx]
    span = float((selected.max() - selected.min()).item())
    if min_span_sec <= 1e-6:
        return 1.0 if span > 0 else 0.0
    return max(0.0, min(1.0, span / min_span_sec))


def _temporal_coverage_reward(
    actions: torch.Tensor,
    timestamps: torch.Tensor | None,
    coverage_bins: int,
) -> float:
    """将时间轴分桶，统计选中帧落入的不同桶比例。"""
    if timestamps is None or timestamps.numel() == 0:
        return 0.0
    idx = torch.nonzero(actions > 0.5, as_tuple=False).squeeze(-1)
    if idx.numel() <= 1:
        return 0.0
    bins = max(int(coverage_bins), 2)
    ts = timestamps.float()
    selected = ts[idx]
    t_min = float(ts.min().item())
    t_max = float(ts.max().item())
    if t_max - t_min <= 1e-6:
        return 0.0
    norm = (selected - t_min) / (t_max - t_min + 1e-6)
    bid = torch.clamp((norm * bins).long(), min=0, max=bins - 1)
    covered = torch.unique(bid).numel()
    return float(covered / bins)


def _stability_rewards(labels: list[str]) -> list[float]:
    """组内答案与众数一致则给分；有效答案≤1 时全零（无信息量）。"""
    valid = [x for x in labels if x]
    if not valid:
        return [0.0 for _ in labels]
    # Stability with <=1 valid prediction is uninformative for policy learning.
    if len(valid) <= 1:
        return [0.0 for _ in labels]
    counter = Counter(valid)
    majority_label, majority_count = counter.most_common(1)[0]
    n_valid = len(valid)
    out = []
    for lb in labels:
        if not lb:
            out.append(0.0)
        elif lb == majority_label:
            out.append(majority_count / max(n_valid, 1))
        else:
            out.append(0.0)
    return out


def _compute_sample_stage2_loss(
    model: DSCAVL,
    old_policy,
    ref_policy,
    cfg: DSCAVLConfig,
    rewarder: QwenAnswerRewarder,
    query: torch.Tensor,
    query_mask: torch.Tensor | None,
    precomputed_features: torch.Tensor,
    video_path: str,
    timestamps: torch.Tensor | None,
    question: str,
    options: list[str],
    answer: str,
    device: torch.device,
    omega_conf: float,
    omega_stability: float,
    omega_temporal: float,
    omega_coverage: float,
    temporal_min_span_sec: float,
    coverage_bins: int,
):
    """组内 rollout：基础 Qwen+graph 奖励 + conf/stability/temporal/coverage 加权，再 GRPO。"""
    precomputed_features = F.layer_norm(precomputed_features, (precomputed_features.shape[-1],))

    rollout_outputs = []
    group_labels: list[str] = []
    group_parts: list[RolloutRewardParts] = []
    group_size = max(int(cfg.group_size), 1)

    for _ in range(group_size):
        backbone = model(
            None,
            query,
            mode="stage2_backbone",
            attention_mask=query_mask,
            precomputed_features=precomputed_features,
        )
        policy_kw = {
            "h_event": backbone.get("H_event"),
            "s_event": backbone.get("S_event"),
            "membership": backbone.get("membership"),
            "frame_aux": backbone.get("frame_patch_stats"),
            "event_aux": backbone.get("event_patch_stats"),
        }
        with torch.no_grad():
            old_out = old_policy(
                backbone["H_graph"],
                backbone["f_q"],
                backbone["S_graph"],
                sample=True,
                intra_event_topk=False,
                **policy_kw,
            )

        rollout_out = {
            **backbone,
            "actions": old_out["actions"],
            "probs": old_out["probs"],
            "probs_old": old_out["probs"],
            "event_actions": old_out.get("event_actions"),
        }

        pred_text = rewarder.generate_answer_from_selected_frames(
            video_path=video_path,
            timestamps=timestamps,
            actions=rollout_out["actions"][0].detach().cpu(),
            probs=rollout_out["probs"][0].detach().cpu(),
            question=question,
            options=options,
        )
        reward_result = rewarder.score_mcq_answer(pred_text=pred_text, gt_answer=answer, device=device)
        rb = reward_bundle(
            reward_acc=reward_result.reward_acc,
            a_temp=rollout_out["A_temp"],
            a_event=rollout_out["A_event"],
            membership=rollout_out["membership"],
            actions=rollout_out["actions"],
            omega_cons=cfg.omega_cons,
            omega_sparse=cfg.omega_sparse,
            omega_coverage=getattr(cfg, "omega_coverage", 0.2),
            frame_patch_stats=rollout_out.get("frame_patch_stats"),
        )

        r_base = float(torch.nan_to_num(rb["R_total"].mean().detach(), nan=0.0, posinf=0.0, neginf=0.0).item())
        r_conf = _format_confidence_reward(reward_result.pred_label, reward_result.pred_text)
        r_temporal = _temporal_span_reward(
            actions=rollout_out["actions"][0].detach().cpu(),
            timestamps=timestamps.detach().cpu() if isinstance(timestamps, torch.Tensor) else None,
            min_span_sec=temporal_min_span_sec,
        )
        r_coverage = _temporal_coverage_reward(
            actions=rollout_out["actions"][0].detach().cpu(),
            timestamps=timestamps.detach().cpu() if isinstance(timestamps, torch.Tensor) else None,
            coverage_bins=coverage_bins,
        )

        rollout_outputs.append(rollout_out)
        group_labels.append(reward_result.pred_label)
        group_parts.append(
            RolloutRewardParts(
                r_base=r_base,
                r_conf=r_conf,
                r_temporal=r_temporal,
                r_coverage=r_coverage,
                r_stability=0.0,
                r_total=r_base,  # placeholder, updated after stability is computed
            )
        )

    if not rollout_outputs:
        zero = torch.zeros((), device=device)
        return zero, {"R_total": 0.0, "R_conf": 0.0, "R_stability": 0.0, "R_temporal": 0.0}

    st_rewards = _stability_rewards(group_labels)
    final_rewards = []
    for i, part in enumerate(group_parts):
        r_stability = st_rewards[i]
        r_total = (
            part.r_base
            + omega_conf * part.r_conf
            + omega_stability * r_stability
            + omega_temporal * part.r_temporal
            + omega_coverage * part.r_coverage
        )
        group_parts[i] = RolloutRewardParts(
            r_base=part.r_base,
            r_conf=part.r_conf,
            r_temporal=part.r_temporal,
            r_coverage=part.r_coverage,
            r_stability=r_stability,
            r_total=r_total,
        )
        final_rewards.append(r_total)

    rewards = torch.tensor(final_rewards, device=device, dtype=torch.float32)
    reward_std = float(rewards.std(unbiased=False).detach().item()) if rewards.numel() > 0 else 0.0
    # GRPO requires within-group relative ranking. If reward variance collapses,
    # skip this sample instead of replacing advantages with raw rewards.
    if reward_std < 1e-8:
        nan_loss = torch.tensor(float("nan"), device=device)
        return nan_loss, {
            "R_total": float(rewards.mean().item()) if rewards.numel() > 0 else 0.0,
            "R_conf": float(sum(x.r_conf for x in group_parts) / len(group_parts)) if group_parts else 0.0,
            "R_stability": float(sum(x.r_stability for x in group_parts) / len(group_parts)) if group_parts else 0.0,
            "R_temporal": float(sum(x.r_temporal for x in group_parts) / len(group_parts)) if group_parts else 0.0,
            "R_coverage": float(sum(x.r_coverage for x in group_parts) / len(group_parts)) if group_parts else 0.0,
            "reward_std": reward_std,
            "degenerate_group": 1.0,
        }
    advantages = compute_advantages(rewards.unsqueeze(0)).squeeze(0)
    advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
    if advantages.abs().max().detach().item() < 1e-8:
        nan_loss = torch.tensor(float("nan"), device=device)
        return nan_loss, {
            "R_total": float(rewards.mean().item()),
            "R_conf": float(sum(x.r_conf for x in group_parts) / len(group_parts)),
            "R_stability": float(sum(x.r_stability for x in group_parts) / len(group_parts)),
            "R_temporal": float(sum(x.r_temporal for x in group_parts) / len(group_parts)),
            "R_coverage": float(sum(x.r_coverage for x in group_parts) / len(group_parts)),
            "reward_std": reward_std,
            "degenerate_group": 1.0,
        }
    advantages = advantages.clamp(-5.0, 5.0)

    total_loss = torch.zeros((), device=device)
    valid_rollouts = 0
    for ridx, out in enumerate(rollout_outputs):
        lp_new, lp_old, kl_pairs = compute_stage2_grpo_terms(
            model,
            old_policy,
            ref_policy,
            out,
            out["actions"],
            out.get("event_actions"),
        )
        rl = grpo_objective(
            lp_new,
            lp_old,
            advantages[ridx : ridx + 1],
            cfg.beta_kl,
            kl_logits_pairs=kl_pairs,
        )
        dsam_reg = cfg.stage2_dsam_reg_scale * (
            0.5 * out["loss_orth"]
            + 0.5 * out["loss_recon"]
        )
        rollout_loss = rl["loss_rl"] + dsam_reg
        if not torch.isfinite(rollout_loss):
            continue
        total_loss = total_loss + rollout_loss
        valid_rollouts += 1

    if valid_rollouts == 0:
        zero = torch.zeros((), device=device)
        return zero, {
            "R_total": float(rewards.mean().item()),
            "R_conf": 0.0,
            "R_stability": 0.0,
            "R_temporal": 0.0,
            "R_coverage": 0.0,
            "reward_std": reward_std,
            "degenerate_group": 0.0,
        }

    sample_loss = total_loss / valid_rollouts
    diag = {
        "R_total": float(sum(x.r_total for x in group_parts) / len(group_parts)),
        "R_conf": float(sum(x.r_conf for x in group_parts) / len(group_parts)),
        "R_stability": float(sum(x.r_stability for x in group_parts) / len(group_parts)),
        "R_temporal": float(sum(x.r_temporal for x in group_parts) / len(group_parts)),
        "R_coverage": float(sum(x.r_coverage for x in group_parts) / len(group_parts)),
        "reward_std": reward_std,
        "degenerate_group": 0.0,
    }
    return sample_loss, diag


def _backward_loss(model: DSCAVL, optimizer: AdamW, loss: torch.Tensor, cfg: DSCAVLConfig) -> bool:
    """简化反传：无逐参数 NaN 检查，直接 clip + step。"""
    if not torch.isfinite(loss):
        return False
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
    optimizer.step()
    return True


def run_stage2_epoch(
    model: DSCAVL,
    ref_policy,
    rewarder: QwenAnswerRewarder,
    optimizer: AdamW,
    dataloader: DataLoader,
    cfg: DSCAVLConfig,
    device: torch.device,
    eval_mcq_only: bool,
    omega_conf: float,
    omega_stability: float,
    omega_temporal: float,
    omega_coverage: float,
    temporal_min_span_sec: float,
    coverage_bins: int,
    epoch_id: int,
    sanity_steps: int = 0,
) -> dict[str, float]:
    """返回 epoch 平均 loss 与各项奖励诊断字典。"""
    total_loss = 0.0
    total_r_total = 0.0
    total_r_conf = 0.0
    total_r_stability = 0.0
    total_r_temporal = 0.0
    total_r_coverage = 0.0
    total_reward_std = 0.0
    total_degenerate_group = 0.0
    step_count = 0
    skipped_nonfinite_samples = 0
    skipped_empty_batches = 0
    skipped_backward = 0

    progress = tqdm(dataloader, desc=f"[Stage2-Qwen-Consistency] Epoch {epoch_id}", leave=False)
    for batch in progress:
        features_list = batch.get("precomputed_features")
        if features_list is None:
            continue
        input_ids, attention_mask = _extract_batch_tensors(batch, device)
        options_batch = batch.get("options")
        answers_batch = batch.get("answer")
        questions = batch.get("question")
        video_paths = batch.get("video_path")
        timestamps_list = batch.get("timestamps")
        old_policy = copy.deepcopy(model.policy).eval()

        batch_loss = torch.zeros((), device=device)
        batch_diags = []
        valid_count = 0

        for i, features in enumerate(features_list):
            if features is None:
                continue
            if eval_mcq_only and not options_batch[i]:
                continue

            precomputed_features = features.unsqueeze(0).to(device)
            query = input_ids[i : i + 1]
            query_mask = attention_mask[i : i + 1] if attention_mask is not None else None
            timestamps = timestamps_list[i]
            if isinstance(timestamps, torch.Tensor):
                timestamps = timestamps.cpu()

            sample_loss, diag = _compute_sample_stage2_loss(
                model=model,
                old_policy=old_policy,
                ref_policy=ref_policy,
                cfg=cfg,
                rewarder=rewarder,
                query=query,
                query_mask=query_mask,
                precomputed_features=precomputed_features,
                video_path=video_paths[i],
                timestamps=timestamps,
                question=questions[i],
                options=options_batch[i],
                answer=answers_batch[i],
                device=device,
                omega_conf=omega_conf,
                omega_stability=omega_stability,
                omega_temporal=omega_temporal,
                omega_coverage=omega_coverage,
                temporal_min_span_sec=temporal_min_span_sec,
                coverage_bins=coverage_bins,
            )
            if not torch.isfinite(sample_loss):
                skipped_nonfinite_samples += 1
                continue
            batch_loss = batch_loss + sample_loss
            batch_diags.append(diag)
            valid_count += 1

        if valid_count == 0:
            skipped_empty_batches += 1
            continue

        loss = batch_loss / valid_count
        if not _backward_loss(model, optimizer, loss, cfg):
            skipped_backward += 1
            continue

        mean_diag = {
            "R_total": sum(d["R_total"] for d in batch_diags) / len(batch_diags),
            "R_conf": sum(d["R_conf"] for d in batch_diags) / len(batch_diags),
            "R_stability": sum(d["R_stability"] for d in batch_diags) / len(batch_diags),
            "R_temporal": sum(d["R_temporal"] for d in batch_diags) / len(batch_diags),
            "R_coverage": sum(d["R_coverage"] for d in batch_diags) / len(batch_diags),
            "reward_std": sum(float(d.get("reward_std", 0.0)) for d in batch_diags) / len(batch_diags),
            "degenerate_group": sum(float(d.get("degenerate_group", 0.0)) for d in batch_diags) / len(batch_diags),
        }
        total_loss += float(loss.detach().item())
        total_r_total += mean_diag["R_total"]
        total_r_conf += mean_diag["R_conf"]
        total_r_stability += mean_diag["R_stability"]
        total_r_temporal += mean_diag["R_temporal"]
        total_r_coverage += mean_diag["R_coverage"]
        total_reward_std += mean_diag["reward_std"]
        total_degenerate_group += mean_diag["degenerate_group"]
        step_count += 1
        progress.set_postfix(
            loss=f"{(total_loss / max(step_count, 1)):.4f}",
            r_total=f"{(total_r_total / max(step_count, 1)):.4f}",
            r_std=f"{(total_reward_std / max(step_count, 1)):.4f}",
            deg=f"{(total_degenerate_group / max(step_count, 1)):.2f}",
        )
        if sanity_steps > 0 and step_count >= sanity_steps:
            break

    print(
        "[Stage2-Qwen-Consistency][diag] "
        f"steps={step_count}, "
        f"skip_nonfinite_samples={skipped_nonfinite_samples}, "
        f"skip_empty_batches={skipped_empty_batches}, "
        f"skip_backward={skipped_backward}, "
        f"reward_std_mean={(total_reward_std / max(step_count, 1)):.4f}, "
        f"degenerate_group_rate={(total_degenerate_group / max(step_count, 1)):.3f}"
    )

    denom = max(step_count, 1)
    return {
        "loss": total_loss / denom,
        "R_total": total_r_total / denom,
        "R_conf": total_r_conf / denom,
        "R_stability": total_r_stability / denom,
        "R_temporal": total_r_temporal / denom,
        "R_coverage": total_r_coverage / denom,
        "reward_std": total_reward_std / denom,
        "degenerate_group": total_degenerate_group / denom,
    }


def save_checkpoint(
    ckpt_dir: Path,
    model: DSCAVL,
    optimizer: AdamW,
    cfg: DSCAVLConfig,
    epoch: int,
    metrics: dict[str, float],
    is_best: bool,
):
    """与 FromStage1 脚本一致的 checkpoint 格式。"""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "policy_state_dict": model.policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg.__dict__,
        "metrics": metrics,
    }
    torch.save(payload, ckpt_dir / f"checkpoint-epoch{epoch}.pt")
    torch.save(payload, ckpt_dir / "checkpoint-latest.pt")
    if is_best:
        torch.save(payload, ckpt_dir / "checkpoint-best.pt")


def main():
    """组装 Qwen、加载 Stage1、校验 group_size，训练并记录扩展指标。"""
    args = parse_args()
    cfg = DSCAVLConfig()
    if args.train_batch_size is not None:
        cfg.train_batch_size = max(int(args.train_batch_size), 1)
    if args.group_size is not None:
        cfg.group_size = max(int(args.group_size), 1)
    if args.num_workers is not None:
        cfg.num_workers = max(int(args.num_workers), 0)
    if cfg.group_size < 2:
        print(
            "[Stage2-Qwen-Consistency][warn] group_size < 2 makes GRPO advantages degenerate; "
            "please set --group-size >= 2."
        )

    repo_root = Path(__file__).resolve().parent
    data_root = resolve_path_from_repo(repo_root, args.data_root)
    feature_root = resolve_path_from_repo(repo_root, args.feature_root)
    stage1_ckpt = resolve_path_from_repo(repo_root, args.stage1_checkpoint)
    if not stage1_ckpt.exists():
        raise FileNotFoundError(f"Stage1 checkpoint not found: {stage1_ckpt}")
    stage1_payload = torch.load(stage1_ckpt, map_location="cpu")
    merge_cgrm_use_f_bg_cfg(
        cfg,
        stage1_payload if isinstance(stage1_payload, dict) else None,
        args.cgrm_use_f_bg,
        args.no_cgrm_f_bg,
    )
    print(
        "[Stage2-Qwen-Consistency] runtime cfg: "
        f"batch_size={cfg.train_batch_size}, group_size={cfg.group_size}, num_workers={cfg.num_workers}, "
        f"cgrm_use_f_bg={cfg.cgrm_use_f_bg}"
    )

    tokenizer = build_tokenizer(cfg, repo_root)
    dataloader = build_dataloader(cfg, tokenizer, data_root, feature_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = QueryTextEncoder(vocab_size=len(tokenizer), dim=cfg.dim)
    model = DSCAVL(cfg, text_encoder=text_encoder).to(device)
    load_stage1_weights(model, stage1_ckpt, device, state=stage1_payload)
    ref_policy = copy.deepcopy(model.policy).eval()
    set_trainable_params(model, finetune_dsam_cgrm=args.finetune_dsam_cgrm)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.stage2_lr, weight_decay=cfg.weight_decay)
    rewarder = QwenAnswerRewarder(
        model_path=args.qwen_model_path,
        device=device,
        max_new_tokens=args.qwen_max_new_tokens,
        max_selected_frames=args.max_selected_frames,
    )

    ckpt_dir = resolve_train_checkpoint_dir(repo_root, args.checkpoint_dir, "stage2_qwen_consistency_reward")
    print(f"[Stage2-Qwen-Consistency] checkpoint_dir={ckpt_dir}")
    project = resolve_swanlab_project(args, cfg)
    swan_ctx = swanlab_try_init(
        disabled=args.disable_swanlab,
        project=project,
        experiment_name=args.swanlab_experiment_name,
        experiment_name_prefix="stage2-qwen-consistency",
        workspace=args.swanlab_workspace,
        log_prefix="Stage2-Qwen-Consistency",
        config={
            "stage": "stage2_qwen_consistency_reward",
            "swanlab_project": project,
            "data_root": str(data_root),
            "feature_root": str(feature_root),
            "stage1_checkpoint": str(stage1_ckpt),
            "qwen_model_path": args.qwen_model_path,
            "eval_mcq_only": args.eval_mcq_only,
            "finetune_dsam_cgrm": args.finetune_dsam_cgrm,
            "cgrm_use_f_bg": cfg.cgrm_use_f_bg,
            "epochs": cfg.stage2_epochs,
            "group_size": cfg.group_size,
            "beta_kl": cfg.beta_kl,
            "omega_cons": cfg.omega_cons,
            "omega_sparse": cfg.omega_sparse,
            "omega_conf": args.omega_conf,
            "omega_stability": args.omega_stability,
            "omega_temporal": args.omega_temporal,
            "omega_coverage": args.omega_coverage,
            "coverage_bins": args.coverage_bins,
            "temporal_min_span_sec": args.temporal_min_span_sec,
            "lr": cfg.stage2_lr,
            "checkpoint_dir": str(ckpt_dir),
        },
    )

    best_reward = float("-inf")
    try:
        for epoch in range(cfg.stage2_epochs):
            epoch_id = epoch + 1
            metrics = run_stage2_epoch(
                model=model,
                ref_policy=ref_policy,
                rewarder=rewarder,
                optimizer=optimizer,
                dataloader=dataloader,
                cfg=cfg,
                device=device,
                eval_mcq_only=args.eval_mcq_only,
                omega_conf=args.omega_conf,
                omega_stability=args.omega_stability,
                omega_temporal=args.omega_temporal,
                omega_coverage=args.omega_coverage,
                temporal_min_span_sec=args.temporal_min_span_sec,
                coverage_bins=args.coverage_bins,
                epoch_id=epoch_id,
                sanity_steps=args.sanity_steps,
            )
            print(
                f"[Stage2-Qwen-Consistency] epoch={epoch_id}, "
                f"loss={metrics['loss']:.4f}, "
                f"R_total={metrics['R_total']:.4f}, "
                f"R_conf={metrics['R_conf']:.4f}, "
                f"R_stability={metrics['R_stability']:.4f}, "
                f"R_temporal={metrics['R_temporal']:.4f}, "
                f"R_coverage={metrics['R_coverage']:.4f}, "
                f"reward_std={metrics['reward_std']:.4f}, "
                f"degenerate_group={metrics['degenerate_group']:.3f}"
            )

            swan_ctx.log(
                {
                    "stage2_qwen_consistency/epoch": epoch_id,
                    "stage2_qwen_consistency/loss": metrics["loss"],
                    "stage2_qwen_consistency/R_total": metrics["R_total"],
                    "stage2_qwen_consistency/R_conf": metrics["R_conf"],
                    "stage2_qwen_consistency/R_stability": metrics["R_stability"],
                    "stage2_qwen_consistency/R_temporal": metrics["R_temporal"],
                    "stage2_qwen_consistency/R_coverage": metrics["R_coverage"],
                    "stage2_qwen_consistency/reward_std": metrics["reward_std"],
                    "stage2_qwen_consistency/degenerate_group": metrics["degenerate_group"],
                },
                log_prefix="Stage2-Qwen-Consistency",
            )

            is_best = metrics["R_total"] > best_reward
            if is_best:
                best_reward = metrics["R_total"]
            save_checkpoint(
                ckpt_dir=ckpt_dir,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch_id,
                metrics=metrics,
                is_best=is_best,
            )
    finally:
        swan_ctx.finish(log_prefix="Stage2-Qwen-Consistency")


if __name__ == "__main__":
    main()
