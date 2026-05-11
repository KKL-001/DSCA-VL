"""Stage2：在冻结 SigLIP 下用 GRPO + 代理 MCQ 奖励主要更新 ``FramePolicyHead``。

每条样本内多轨迹采样；旧策略快照用于 importance sampling；参考策略算 KL。
"""
from __future__ import annotations

import argparse
import copy
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
    ensure_frozen_visual_encoder,
    check_gradients_finite,
    get_peak_memory_mb,
    resolve_train_checkpoint_dir,
)
from dscavl.swanlab_train import (
    add_swanlab_train_args,
    resolve_swanlab_project,
    swanlab_try_init,
)
from dscavl.train_utils import add_cgrm_use_f_bg_cli_args, merge_cgrm_use_f_bg_cfg
from dscavl.grpo import compute_advantages, compute_stage2_grpo_terms, grpo_objective
from dscavl.proxy_mcq import compute_mcq_exact_reward
from dscavl.rewards import reward_bundle


def build_tokenizer(cfg: DSCAVLConfig, repo_root: Path):
    """同 Stage1：加载与 query 编码一致的 tokenizer。"""
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
    """Stage2 可不加载字幕（弱监督不在本阶段使用）。"""
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


def _compute_sample_stage2_loss(
    model: DSCAVL,
    old_policy,
    ref_policy,
    cfg: DSCAVLConfig,
    query: torch.Tensor,
    query_mask: torch.Tensor | None,
    precomputed_features: torch.Tensor,
    options: list[str],
    answer: str,
    tokenizer,
    device: torch.device,
):
    """单样本：``group_size`` 次 rollout → 优势 → 逐条 GRPO 损失 + 轻量 DSAM 正则。"""
    # 按最后一维 LayerNorm，稳定特征尺度
    precomputed_features = F.layer_norm(precomputed_features, (precomputed_features.shape[-1],))

    rollout_outputs = []
    rollout_rewards = []
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

        reward_acc, _, _, _ = compute_mcq_exact_reward(
            model=model,
            out=rollout_out,
            options=options,
            answer=answer,
            tokenizer=tokenizer,
            device=device,
        )
        rb = reward_bundle(
            reward_acc=reward_acc,
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
        rollout_outputs.append((rollout_out, rb))
        rollout_rewards.append(reward_scalar)

    if not rollout_outputs:
        zero = torch.zeros((), device=device)
        return zero, 0.0

    rewards = torch.stack(rollout_rewards).to(device)
    advantages = compute_advantages(rewards.unsqueeze(0)).squeeze(0)
    advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0).clamp(-5.0, 5.0)

    total_loss = torch.zeros((), device=device)
    valid_rollouts = 0

    for ridx, (out, _rb) in enumerate(rollout_outputs):
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

        # Only include rl_loss in the backward pass; regularizers are diagnostic-only in stage2.
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
        return zero, rewards.mean().detach().item()

    sample_loss = total_loss / valid_rollouts
    sample_reward = rewards.mean().detach().item()
    return sample_loss, sample_reward


def _extract_batch_tensors(batch, device: torch.device):
    """取出并搬到 device 的 query 张量对。"""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _backward_stage2_loss(model: DSCAVL, optimizer: AdamW, loss: torch.Tensor, cfg: DSCAVLConfig) -> bool:
    """反传、梯度有限性检查、裁剪与 ``optimizer.step``；失败返回 False。"""
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


def _accumulate_stage2_batch(
    model: DSCAVL,
    ref_policy,
    tokenizer,
    batch,
    cfg: DSCAVLConfig,
    device: torch.device,
):
    """对一个 DataLoader batch 逐样本累加 loss/reward；深拷贝当前 policy 作 old_policy。"""
    features_list = batch.get("precomputed_features")
    if features_list is None:
        return None

    input_ids, attention_mask = _extract_batch_tensors(batch, device)
    options_batch = batch.get("options")
    answers_batch = batch.get("answer")
    old_policy = copy.deepcopy(model.policy).eval()

    batch_loss = torch.zeros((), device=device)
    batch_reward = 0.0
    valid_count = 0
    skipped_nonfinite_samples = 0

    for i, features in enumerate(features_list):
        if features is None:
            continue

        precomputed_features = features.unsqueeze(0).to(device)
        query = input_ids[i : i + 1]
        query_mask = attention_mask[i : i + 1] if attention_mask is not None else None

        sample_loss, sample_reward = _compute_sample_stage2_loss(
            model,
            old_policy,
            ref_policy,
            cfg,
            query,
            query_mask,
            precomputed_features,
            options=options_batch[i],
            answer=answers_batch[i],
            tokenizer=tokenizer,
            device=device,
        )
        if not torch.isfinite(sample_loss):
            skipped_nonfinite_samples += 1
            continue
        batch_loss = batch_loss + sample_loss
        batch_reward += sample_reward
        valid_count += 1

    return batch_loss, batch_reward, valid_count, skipped_nonfinite_samples


def run_stage2_epoch(
    model: DSCAVL,
    ref_policy,
    tokenizer,
    optimizer: AdamW,
    dataloader: DataLoader,
    cfg: DSCAVLConfig,
    device: torch.device,
    epoch_id: int,
    sanity_steps: int = 0,
) -> tuple[float, float, dict[str, float]]:
    """返回 epoch 平均 loss、平均总奖励，以及供 SwanLab 记录的诊断标量。"""
    total_loss = 0.0
    total_reward = 0.0
    step_count = 0
    skipped_nonfinite_samples = 0
    skipped_nonfinite_batches = 0
    skipped_nonfinite_grads = 0

    progress = tqdm(dataloader, desc=f"[Stage2] Epoch {epoch_id}", leave=False)
    for batch in progress:
        batch_result = _accumulate_stage2_batch(model, ref_policy, tokenizer, batch, cfg, device)
        if batch_result is None:
            continue
        batch_loss, batch_reward, valid_count, skipped_samples = batch_result
        skipped_nonfinite_samples += skipped_samples

        if valid_count == 0:
            continue

        loss = batch_loss / valid_count
        if not torch.isfinite(loss):
            skipped_nonfinite_batches += 1
            continue

        if not _backward_stage2_loss(model, optimizer, loss, cfg):
            skipped_nonfinite_grads += 1
            continue

        total_loss += loss.detach().item()
        total_reward += batch_reward / valid_count
        step_count += 1
        progress.set_postfix(
            loss=f"{(total_loss / max(step_count, 1)):.4f}",
            reward=f"{(total_reward / max(step_count, 1)):.4f}",
        )
        if sanity_steps > 0 and step_count >= sanity_steps:
            break

    mean_loss = total_loss / max(step_count, 1)
    mean_reward = total_reward / max(step_count, 1)
    peak_mb = get_peak_memory_mb()
    print(
        "[Stage2][diag] "
        f"steps={step_count}, "
        f"skip_samples_nonfinite={skipped_nonfinite_samples}, "
        f"skip_batches_nonfinite={skipped_nonfinite_batches}, "
        f"skip_grads_nonfinite={skipped_nonfinite_grads}"
        + (f", peak_mem_mb={peak_mb:.1f}" if peak_mb > 0 else "")
        + " (loss=rl_only, excludes detached regularizers)"
    )
    diag = {
        "stage2/diag_steps": float(step_count),
        "stage2/diag_skip_samples_nonfinite": float(skipped_nonfinite_samples),
        "stage2/diag_skip_batches_nonfinite": float(skipped_nonfinite_batches),
        "stage2/diag_skip_grads_nonfinite": float(skipped_nonfinite_grads),
        "stage2/diag_peak_memory_mb": float(peak_mb),
    }
    return mean_loss, mean_reward, diag


def _save_stage2_checkpoint(
    ckpt_dir: Path,
    model: DSCAVL,
    optimizer: AdamW,
    cfg: DSCAVLConfig,
    epoch: int,
    mean_loss: float,
    mean_reward: float,
    is_best: bool,
):
    """额外保存 ``policy_state_dict`` 便于只加载策略头。"""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"loss": mean_loss, "reward": mean_reward}
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "policy_state_dict": model.policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg.__dict__,
        "metrics": metrics,
    }
    epoch_path = ckpt_dir / f"checkpoint-epoch{epoch}.pt"
    latest_path = ckpt_dir / "checkpoint-latest.pt"
    torch.save(payload, epoch_path)
    torch.save(payload, latest_path)
    if is_best:
        torch.save(payload, ckpt_dir / "checkpoint-best.pt")


def parse_args() -> argparse.Namespace:
    """覆盖 batch/group_size/workers 等运行时参数。"""
    parser = argparse.ArgumentParser(description="Train DSCA-VL stage2.")
    parser.add_argument("--data-root", type=str, default=None, help="Override cfg.data_root")
    parser.add_argument("--feature-root", type=str, default=None, help="Override cfg.feature_root")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory. Default: output/stage2/<YYYYMMDD_HHMMSS> under repo root.",
    )
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
    add_cgrm_use_f_bg_cli_args(parser)
    return parser.parse_args()


def main():
    """仅解冻 policy 参数；构建 ref_policy 快照并开始 Stage2 训练循环。"""
    args = parse_args()
    cfg = DSCAVLConfig()
    if args.data_root:
        cfg.data_root = args.data_root
    if args.feature_root:
        cfg.feature_root = args.feature_root
    if args.train_batch_size is not None:
        cfg.train_batch_size = max(int(args.train_batch_size), 1)
    if args.group_size is not None:
        cfg.group_size = max(int(args.group_size), 1)
    if args.num_workers is not None:
        cfg.num_workers = max(int(args.num_workers), 0)
    merge_cgrm_use_f_bg_cfg(cfg, None, args.cgrm_use_f_bg, args.no_cgrm_f_bg)

    repo_root = Path(__file__).resolve().parent
    data_root = repo_root / cfg.data_root
    feature_root = repo_root / cfg.feature_root
    print(
        "[Stage2] runtime cfg: "
        f"batch_size={cfg.train_batch_size}, group_size={cfg.group_size}, num_workers={cfg.num_workers}, "
        f"cgrm_use_f_bg={cfg.cgrm_use_f_bg}"
    )

    tokenizer = build_tokenizer(cfg, repo_root)
    dataloader = build_dataloader(cfg, tokenizer, data_root, feature_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = QueryTextEncoder(vocab_size=len(tokenizer), dim=cfg.dim)
    model = DSCAVL(cfg, text_encoder=text_encoder).to(device)
    ref_policy = copy.deepcopy(model.policy).eval()

    # Stage2: update policy only for stability.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.policy.parameters():
        p.requires_grad = True

    if not ensure_frozen_visual_encoder(model):
        raise RuntimeError(
            "[Stage2] Visual encoder (SigLIP) must be frozen. Use precomputed features only."
        )
    print("[Stage2] Visual encoder frozen (precomputed features), policy-only training.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.stage2_lr, weight_decay=cfg.weight_decay)
    ckpt_dir = resolve_train_checkpoint_dir(repo_root, args.checkpoint_dir, "stage2")
    print(f"[Stage2] checkpoint_dir={ckpt_dir}")

    project = resolve_swanlab_project(args, cfg)
    swan_ctx = swanlab_try_init(
        disabled=args.disable_swanlab,
        project=project,
        experiment_name=args.swanlab_experiment_name,
        experiment_name_prefix="stage2",
        workspace=args.swanlab_workspace,
        log_prefix="Stage2",
        config={
            "stage": "stage2",
            "swanlab_project": project,
            "data_root": str(data_root),
            "feature_root": str(feature_root),
            "epochs": cfg.stage2_epochs,
            "batch_size": cfg.train_batch_size,
            "lr": cfg.stage2_lr,
            "weight_decay": cfg.weight_decay,
            "text_model_name_or_path": cfg.text_model_name_or_path,
            "group_size": cfg.group_size,
            "beta_kl": cfg.beta_kl,
            "omega_cons": cfg.omega_cons,
            "omega_sparse": cfg.omega_sparse,
            "cgrm_use_f_bg": cfg.cgrm_use_f_bg,
            "checkpoint_dir": str(ckpt_dir),
        },
    )

    best_reward = float("-inf")
    try:
        for epoch in range(cfg.stage2_epochs):
            epoch_id = epoch + 1
            mean_loss, mean_reward, diag = run_stage2_epoch(
                model,
                ref_policy,
                tokenizer,
                optimizer,
                dataloader,
                cfg,
                device,
                epoch_id,
                args.sanity_steps,
            )
            print(f"[Stage2] epoch={epoch_id}, loss={mean_loss:.4f}, R_total={mean_reward:.4f}")

            log_payload = {
                "stage2/epoch": epoch_id,
                "stage2/loss": mean_loss,
                "stage2/R_total": mean_reward,
            }
            log_payload.update(diag)
            swan_ctx.log(log_payload, log_prefix="Stage2")

            is_best = mean_reward > best_reward
            if is_best:
                best_reward = mean_reward
            _save_stage2_checkpoint(ckpt_dir, model, optimizer, cfg, epoch_id, mean_loss, mean_reward, is_best)
    finally:
        swan_ctx.finish(log_prefix="Stage2")


if __name__ == "__main__":
    main()