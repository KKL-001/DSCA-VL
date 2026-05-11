"""Stage2（Qwen 闭环）：强制从 Stage1 checkpoint 初始化，用 Qwen2-VL 答题作奖励做 GRPO。

可选 ``--finetune-dsam-cgrm`` 同时微调分解与图模块；默认仅优化策略头。
"""
from __future__ import annotations

import argparse
import copy
import os
import shutil
from pathlib import Path
from typing import Any, Dict

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
    ensure_frozen_visual_encoder,
    check_gradients_finite,
    get_peak_memory_mb,
    resolve_train_checkpoint_dir,
)
from dscavl.train_utils import (
    add_cgrm_use_f_bg_cli_args,
    merge_cgrm_use_f_bg_cfg,
    model_state_dict_from_stage1_payload,
)
from dscavl.swanlab_train import (
    add_swanlab_train_args,
    resolve_swanlab_project,
    swanlab_try_init,
)
from dscavl.grpo import compute_advantages, compute_stage2_grpo_terms, grpo_objective
from dscavl.rewards import reward_bundle
from qwen_reward_utils import QwenAnswerRewarder, resolve_path_from_repo


def parse_args() -> argparse.Namespace:
    """解析数据/特征路径、Stage1 ckpt、Qwen 路径及训练规模相关参数。"""
    parser = argparse.ArgumentParser(description="Stage2 GRPO training with Qwen2-VL answer reward, initialized from Stage1.")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset root (query/video/subtitle).")
    parser.add_argument("--feature-root", type=str, default="features", help="Cached feature directory.")
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        default="output/stage1/checkpoint-best.pt",
        help="Stage1 checkpoint path used to initialize stage2 model.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Stage2 checkpoint output directory. Default: output/stage2_qwen_grpo_from_stage1/<YYYYMMDD_HHMMSS>.",
    )
    parser.add_argument(
        "--qwen-model-path",
        type=str,
        default="/root/autodl-tmp/Qwen/Qwen2-VL-2B-Instruct",
        help="Local path or HF id for Qwen2-VL reward model.",
    )
    parser.add_argument("--qwen-max-new-tokens", type=int, default=32)
    parser.add_argument("--max-selected-frames", type=int, default=8)
    parser.add_argument("--eval-mcq-only", action="store_true", help="Skip samples with empty options.")
    parser.add_argument("--train-batch-size", type=int, default=None, help="Override cfg.train_batch_size.")
    parser.add_argument("--group-size", type=int, default=None, help="Override cfg.group_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override cfg.num_workers.")
    parser.add_argument(
        "--sanity-steps",
        type=int,
        default=0,
        help="Run only N optimizer steps per epoch for quick sanity validation (<=0 means full epoch).",
    )
    add_swanlab_train_args(parser)
    parser.add_argument(
        "--finetune-dsam-cgrm",
        action="store_true",
        help="Also finetune DSAM/CGRM during stage2 (default policy-only).",
    )
    add_cgrm_use_f_bg_cli_args(parser)
    parser.add_argument(
        "--no-checkpoint-epoch",
        action="store_true",
        help="不写入 checkpoint-epoch{n}.pt，仅保留 checkpoint-latest.pt（及 is_best 时的 checkpoint-best.pt），省磁盘。",
    )
    return parser.parse_args()


def build_tokenizer(cfg: DSCAVLConfig, repo_root: Path):
    """加载与 DSCA 侧 query 编码一致的 HF tokenizer。"""
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
    """训练集 DataLoader；collate 为 ``variable_feature_collate``。"""
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
    """``strict=False`` 加载整模 state_dict，打印 missing/unexpected 计数。

    若传入 ``state``（已 ``torch.load`` 的整包），避免重复读盘。
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage1 checkpoint not found: {ckpt_path}")
    payload = state if state is not None else torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = model_state_dict_from_stage1_payload(payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        "[Stage2-Qwen-FromStage1] loaded stage1 checkpoint | "
        f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
    )


def set_trainable_params(model: DSCAVL, finetune_dsam_cgrm: bool) -> None:
    """默认只训练 policy；打开开关时追加 DSAM 与 CGRM。"""
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
    """将 batch 中的 query token 与 mask 搬到目标 device。"""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _tmean(x) -> float:
    """任意张量/标量转为 Python float（先 mean 再 item）。"""
    return float(torch.as_tensor(x).detach().float().mean().cpu().item())


def _empty_stage2_metric_dict(cfg: DSCAVLConfig) -> Dict[str, float]:
    """无有效 rollout 时的占位指标（beta_kl 仍记录配置值）。"""
    bk = float(cfg.beta_kl)
    return {
        "loss_pg": 0.0,
        "loss_kl": 0.0,
        "loss_rl_grpo": 0.0,
        "beta_kl": bk,
        "beta_kl_loss_kl": 0.0,
        "loss_orth": 0.0,
        "loss_recon": 0.0,
        "dsam_reg": 0.0,
        "R_total": 0.0,
        "mcq_acc": 0.0,
    }


# 与 ``_empty_stage2_metric_dict`` / ``extras`` 键一致，用于 epoch 聚合
_STAGE2_EXTRA_KEYS = tuple(_empty_stage2_metric_dict(DSCAVLConfig()).keys())


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
):
    """对单视频样本：多轨迹 + Qwen 生成答案 + ``reward_bundle`` + GRPO。

    返回 ``(sample_loss, sample_reward, sample_acc, extras)``；``extras`` 为各分项在有效 rollout 上的均值。
    """
    precomputed_features = F.layer_norm(precomputed_features, (precomputed_features.shape[-1],))

    rollout_outputs = []
    rollout_rewards = []
    rollout_acc = []
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

        reward_scalar = torch.nan_to_num(rb["R_total"].mean().detach(), nan=0.0, posinf=0.0, neginf=0.0)
        rollout_outputs.append(rollout_out)
        rollout_rewards.append(reward_scalar)
        rollout_acc.append(float(reward_result.reward_acc.item()))

    if not rollout_outputs:
        zero = torch.zeros((), device=device)
        return zero, 0.0, 0.0, _empty_stage2_metric_dict(cfg)

    rewards = torch.stack(rollout_rewards).to(device)
    advantages = compute_advantages(rewards.unsqueeze(0)).squeeze(0)
    advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0).clamp(-5.0, 5.0)

    total_loss = torch.zeros((), device=device)
    sum_pg = torch.zeros((), device=device)
    sum_kl = torch.zeros((), device=device)
    sum_rl_grpo = torch.zeros((), device=device)
    sum_orth = torch.zeros((), device=device)
    sum_recon = torch.zeros((), device=device)
    sum_dsam_reg = torch.zeros((), device=device)
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
        orth = _tmean(out["loss_orth"])
        recon = _tmean(out["loss_recon"])
        dsam_reg = cfg.stage2_dsam_reg_scale * (0.5 * orth + 0.5 * recon)
        dsam_reg_t = torch.as_tensor(dsam_reg, device=device, dtype=rl["loss_rl"].dtype)
        rollout_loss = rl["loss_rl"] + dsam_reg_t
        if not torch.isfinite(rollout_loss):
            continue
        total_loss = total_loss + rollout_loss
        sum_pg = sum_pg + rl["loss_pg"]
        sum_kl = sum_kl + rl["loss_kl"]
        sum_rl_grpo = sum_rl_grpo + rl["loss_rl"]
        sum_orth = sum_orth + orth
        sum_recon = sum_recon + recon
        sum_dsam_reg = sum_dsam_reg + float(dsam_reg)
        valid_rollouts += 1

    if valid_rollouts == 0:
        zero = torch.zeros((), device=device)
        ex = _empty_stage2_metric_dict(cfg)
        ex["R_total"] = float(rewards.mean().detach().item())
        ex["mcq_acc"] = sum(rollout_acc) / max(len(rollout_acc), 1)
        return zero, ex["R_total"], ex["mcq_acc"], ex

    vr = float(valid_rollouts)
    mean_pg = float((sum_pg / vr).detach().item())
    mean_kl = float((sum_kl / vr).detach().item())
    mean_rl_grpo = float((sum_rl_grpo / vr).detach().item())
    mean_orth = sum_orth / vr
    mean_recon = sum_recon / vr
    mean_dsam = sum_dsam_reg / vr
    bk = float(cfg.beta_kl)

    sample_loss = total_loss / valid_rollouts
    sample_reward = rewards.mean().detach().item()
    sample_acc = sum(rollout_acc) / max(len(rollout_acc), 1)
    extras: Dict[str, float] = {
        "loss_pg": mean_pg,
        "loss_kl": mean_kl,
        "loss_rl_grpo": mean_rl_grpo,
        "beta_kl": bk,
        "beta_kl_loss_kl": bk * mean_kl,
        "loss_orth": mean_orth,
        "loss_recon": mean_recon,
        "dsam_reg": mean_dsam,
        "R_total": float(sample_reward),
        "mcq_acc": float(sample_acc),
    }
    return sample_loss, sample_reward, sample_acc, extras


def _backward_loss(model: DSCAVL, optimizer: AdamW, loss: torch.Tensor, cfg: DSCAVLConfig) -> bool:
    """标准反传流水线；梯度非有限时清空并返回 False。"""
    if not torch.isfinite(loss):
        return False
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if not check_gradients_finite(model):
        optimizer.zero_grad(set_to_none=True)
        return False
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
    eval_mcq_only: bool = False,
    epoch_id: int = 1,
    sanity_steps: int = 0,
) -> tuple[float, float, float, Dict[str, float]]:
    """按 batch 内逐样本累加 loss 后平均反传。

    返回 ``(mean_loss, mean_reward, mean_acc, detail_metrics)``；``detail_metrics`` 为各 optimizer
    步上 batch 均值的再平均（与 loss 同级别的 epoch 均值）。
    """
    total_loss = 0.0
    total_reward = 0.0
    total_acc = 0.0
    sum_detail = {k: 0.0 for k in _STAGE2_EXTRA_KEYS}
    step_count = 0
    skipped_nonfinite_samples = 0
    skipped_empty_batches = 0
    skipped_backward = 0

    progress = tqdm(dataloader, desc=f"[Stage2-Qwen-FromStage1] Epoch {epoch_id}", leave=False)
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
        batch_reward = 0.0
        batch_acc = 0.0
        batch_detail = {k: 0.0 for k in _STAGE2_EXTRA_KEYS}
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

            sample_loss, sample_reward, sample_acc, sample_ex = _compute_sample_stage2_loss(
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
            )
            if not torch.isfinite(sample_loss):
                skipped_nonfinite_samples += 1
                continue

            batch_loss = batch_loss + sample_loss
            batch_reward += sample_reward
            batch_acc += sample_acc
            for k in _STAGE2_EXTRA_KEYS:
                batch_detail[k] += sample_ex[k]
            valid_count += 1

        if valid_count == 0:
            skipped_empty_batches += 1
            continue

        loss = batch_loss / valid_count
        if not _backward_loss(model, optimizer, loss, cfg):
            skipped_backward += 1
            continue

        total_loss += loss.detach().item()
        mean_r = batch_reward / valid_count
        mean_a = batch_acc / valid_count
        total_reward += mean_r
        total_acc += mean_a
        for k in _STAGE2_EXTRA_KEYS:
            sum_detail[k] += batch_detail[k] / valid_count
        step_count += 1
        denom = max(step_count, 1)
        progress.set_postfix(
            loss=f"{(total_loss / denom):.4f}",
            reward=f"{(total_reward / denom):.4f}",
            acc=f"{(total_acc / denom):.4f}",
            pg=f"{(sum_detail['loss_pg'] / denom):.3f}",
            kl=f"{(sum_detail['loss_kl'] / denom):.3f}",
        )
        if sanity_steps > 0 and step_count >= sanity_steps:
            break

    peak_mb = get_peak_memory_mb()
    print(
        "[Stage2-Qwen-FromStage1][diag] "
        f"steps={step_count}, "
        f"skip_nonfinite_samples={skipped_nonfinite_samples}, "
        f"skip_empty_batches={skipped_empty_batches}, "
        f"skip_backward={skipped_backward}"
        + (f", peak_mem_mb={peak_mb:.1f}" if peak_mb > 0 else "")
    )

    denom = max(step_count, 1)
    detail_out = {k: sum_detail[k] / denom for k in _STAGE2_EXTRA_KEYS}
    return (
        total_loss / denom,
        total_reward / denom,
        total_acc / denom,
        detail_out,
    )


def _atomic_torch_save(obj: Any, path: Path) -> None:
    """先写 ``*.tmp`` 再 ``os.replace``，避免磁盘满或中断时留下半截损坏的 ``.pt``。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def save_checkpoint(
    ckpt_dir: Path,
    model: DSCAVL,
    optimizer: AdamW,
    cfg: DSCAVLConfig,
    epoch: int,
    metrics: dict[str, float],
    is_best: bool,
    *,
    save_epoch_file: bool = True,
):
    """保存完整模型与独立 policy 权重，便于 Stage2 热启。

    只 ``torch.save`` 一次到 ``checkpoint-latest.pt``，再按需 ``copy2``，降低峰值写入与损坏概率。
    若仍报 ``file write failed``，多为磁盘满：``df -h``，并用 ``--checkpoint-dir`` 指到大容量盘（如 autodl 的 data 盘）。
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "policy_state_dict": model.policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg.__dict__,
        "metrics": metrics,
    }
    latest = ckpt_dir / "checkpoint-latest.pt"
    try:
        _atomic_torch_save(payload, latest)
    except RuntimeError as exc:
        raise RuntimeError(
            f"写入 checkpoint 失败（常见原因：磁盘已满或配额用尽）。"
            f"请执行 df -h 检查，并将 --checkpoint-dir 设到有足够空间的路径。"
            f" 目标目录: {ckpt_dir}"
        ) from exc
    if save_epoch_file:
        shutil.copy2(latest, ckpt_dir / f"checkpoint-epoch{epoch}.pt")
    if is_best:
        shutil.copy2(latest, ckpt_dir / "checkpoint-best.pt")


def main():
    """组装 Qwen 奖励器、加载 Stage1、训练循环与 SwanLab（可选）。"""
    args = parse_args()
    cfg = DSCAVLConfig()
    if args.train_batch_size is not None:
        cfg.train_batch_size = max(int(args.train_batch_size), 1)
    if args.group_size is not None:
        cfg.group_size = max(int(args.group_size), 1)
    if args.num_workers is not None:
        cfg.num_workers = max(int(args.num_workers), 0)

    repo_root = Path(__file__).resolve().parent
    data_root = resolve_path_from_repo(repo_root, args.data_root)
    feature_root = resolve_path_from_repo(repo_root, args.feature_root)
    stage1_ckpt = resolve_path_from_repo(repo_root, args.stage1_checkpoint)
    if not stage1_ckpt.exists():
        raise FileNotFoundError(f"Stage1 checkpoint not found: {stage1_ckpt}")
    stage1_payload = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    merge_cgrm_use_f_bg_cfg(
        cfg,
        stage1_payload if isinstance(stage1_payload, dict) else None,
        args.cgrm_use_f_bg,
        args.no_cgrm_f_bg,
    )
    print(
        "[Stage2-Qwen-FromStage1] runtime cfg: "
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

    if not ensure_frozen_visual_encoder(model):
        raise RuntimeError(
            "[Stage2-Qwen-FromStage1] Visual encoder (SigLIP) must be frozen. Use precomputed features only."
        )
    print("[Stage2-Qwen-FromStage1] Visual encoder frozen (precomputed features).")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.stage2_lr, weight_decay=cfg.weight_decay)
    rewarder = QwenAnswerRewarder(
        model_path=args.qwen_model_path,
        device=device,
        max_new_tokens=args.qwen_max_new_tokens,
        max_selected_frames=args.max_selected_frames,
    )

    ckpt_dir = resolve_train_checkpoint_dir(repo_root, args.checkpoint_dir, "stage2_qwen_grpo_from_stage1")
    print(f"[Stage2-Qwen-FromStage1] checkpoint_dir={ckpt_dir}")
    project = resolve_swanlab_project(args, cfg)
    swan_ctx = swanlab_try_init(
        disabled=args.disable_swanlab,
        project=project,
        experiment_name=args.swanlab_experiment_name,
        experiment_name_prefix="stage2-qwen-from-stage1",
        workspace=args.swanlab_workspace,
        log_prefix="Stage2-Qwen-FromStage1",
        config={
            "stage": "stage2_qwen_grpo_from_stage1",
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
            "lr": cfg.stage2_lr,
            "checkpoint_dir": str(ckpt_dir),
        },
    )

    best_reward = float("-inf")
    try:
        for epoch in range(cfg.stage2_epochs):
            epoch_id = epoch + 1
            mean_loss, mean_reward, mean_acc, detail = run_stage2_epoch(
                model=model,
                ref_policy=ref_policy,
                rewarder=rewarder,
                optimizer=optimizer,
                dataloader=dataloader,
                cfg=cfg,
                device=device,
                eval_mcq_only=args.eval_mcq_only,
                epoch_id=epoch_id,
                sanity_steps=args.sanity_steps,
            )
            print(
                f"[Stage2-Qwen-FromStage1] epoch={epoch_id}, "
                f"loss_total={mean_loss:.4f}, R_total={mean_reward:.4f}, R_acc(mcq)={mean_acc:.4f}\n"
                f"  GRPO: loss_pg={detail['loss_pg']:.4f}, loss_kl={detail['loss_kl']:.4f}, "
                f"beta_kl={detail['beta_kl']:.5f}, beta_kl*loss_kl={detail['beta_kl_loss_kl']:.4f}, "
                f"loss_rl_grpo={detail['loss_rl_grpo']:.4f}\n"
                f"  DSAM reg: loss_orth={detail['loss_orth']:.4f}, loss_recon={detail['loss_recon']:.4f}, "
                f"dsam_reg={detail['dsam_reg']:.4f}"
            )

            metrics = {"loss": mean_loss, "reward": mean_reward, "mcq_acc": mean_acc, **detail}
            log_payload: Dict[str, float | int] = {
                "stage2_qwen_from_stage1/epoch": epoch_id,
                "stage2_qwen_from_stage1/loss_total": mean_loss,
                "stage2_qwen_from_stage1/R_total": mean_reward,
                "stage2_qwen_from_stage1/R_acc_mcq": mean_acc,
            }
            for key, val in detail.items():
                log_payload[f"stage2_qwen_from_stage1/{key}"] = val
            swan_ctx.log(log_payload, log_prefix="Stage2-Qwen-FromStage1")

            is_best = mean_reward > best_reward
            if is_best:
                best_reward = mean_reward
            save_checkpoint(
                ckpt_dir=ckpt_dir,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                epoch=epoch_id,
                metrics=metrics,
                is_best=is_best,
                save_epoch_file=not args.no_checkpoint_epoch,
            )
    finally:
        swan_ctx.finish(log_prefix="Stage2-Qwen-FromStage1")


if __name__ == "__main__":
    main()
