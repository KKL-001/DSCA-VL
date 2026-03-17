from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dscavl import DSCAVL, DSCAVLConfig, QueryTextEncoder, QuestionFeatureDataset, variable_feature_collate
from dscavl.grpo import compute_advantages, grpo_objective
from dscavl.proxy_mcq import compute_mcq_exact_reward
from dscavl.rewards import reward_bundle

try:
    import swanlab
except ImportError:  # pragma: no cover - optional dependency
    swanlab = None


def build_tokenizer(cfg: DSCAVLConfig, repo_root: Path):
    tokenizer_path = repo_root / cfg.text_model_name_or_path
    source = str(tokenizer_path) if tokenizer_path.exists() else cfg.text_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_dataloader(cfg: DSCAVLConfig, tokenizer, data_root: Path, feature_root: Path) -> DataLoader:
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
    # Normalize cached SigLIP features per frame for better optimization stability.
    precomputed_features = F.layer_norm(precomputed_features, (precomputed_features.shape[-1],))

    rollout_outputs = []
    rollout_rewards = []
    group_size = max(int(cfg.group_size), 1)

    for _ in range(group_size):
        out = model(
            None,
            query,
            mode="stage2",
            attention_mask=query_mask,
            precomputed_features=precomputed_features,
        )
        # Strict GRPO uses trajectories sampled by a frozen "old" policy snapshot.
        with torch.no_grad():
            old_out = old_policy(out["H_graph"], out["f_q"], out["S_graph"], sample=True)

        rollout_out = {
            **out,
            "actions": old_out["actions"],
            "probs_old": old_out["probs"],
            "logits_old": old_out["logits"],
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
        with torch.no_grad():
            ref_out = ref_policy(out["H_graph"], out["f_q"], out["S_graph"], sample=False)

        rl = grpo_objective(
            probs_new=out["probs"],
            probs_old=out["probs_old"],
            actions=out["actions"],
            advantages=advantages[ridx : ridx + 1],
            logits_new=out["logits"],
            logits_ref=ref_out["logits"],
            beta_kl=cfg.beta_kl,
        )

        # Only include rl_loss in the backward pass; regularizers are diagnostic-only in stage2.
        rollout_loss = rl["loss_rl"]
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
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _backward_stage2_loss(model: DSCAVL, optimizer: AdamW, loss: torch.Tensor, cfg: DSCAVLConfig) -> bool:
    if not torch.isfinite(loss):
        return False

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    grad_finite = True
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            grad_finite = False
            break
    if not grad_finite:
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
) -> tuple[float, float]:
    total_loss = 0.0
    total_reward = 0.0
    step_count = 0
    skipped_nonfinite_samples = 0
    skipped_nonfinite_batches = 0
    skipped_nonfinite_grads = 0

    for batch in dataloader:
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

    mean_loss = total_loss / max(step_count, 1)
    mean_reward = total_reward / max(step_count, 1)
    print(
        "[Stage2][diag] "
        f"steps={step_count}, "
        f"skip_samples_nonfinite={skipped_nonfinite_samples}, "
        f"skip_batches_nonfinite={skipped_nonfinite_batches}, "
        f"skip_grads_nonfinite={skipped_nonfinite_grads} "
        "(loss=rl_only, excludes detached regularizers)"
    )
    return mean_loss, mean_reward


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


def main():
    cfg = DSCAVLConfig()
    repo_root = Path(__file__).resolve().parent
    data_root = repo_root / cfg.data_root
    feature_root = repo_root / cfg.feature_root

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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.stage2_lr, weight_decay=cfg.weight_decay)
    ckpt_dir = repo_root / "output" / "stage2"

    run = None
    if swanlab is not None:
        run = swanlab.init(
            project="DSCA-VL",
            experiment_name=f"stage2-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "stage": "stage2",
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
            },
        )
    else:
        print("[Stage2] swanlab not installed, skip experiment tracking.")

    best_reward = float("-inf")
    try:
        for epoch in range(cfg.stage2_epochs):
            epoch_id = epoch + 1
            mean_loss, mean_reward = run_stage2_epoch(model, ref_policy, tokenizer, optimizer, dataloader, cfg, device)
            print(f"[Stage2] epoch={epoch_id}, loss={mean_loss:.4f}, R_total={mean_reward:.4f}")

            if run is not None:
                swanlab.log(
                    {
                        "stage2/epoch": epoch_id,
                        "stage2/loss": mean_loss,
                        "stage2/R_total": mean_reward,
                    }
                )

            is_best = mean_reward > best_reward
            if is_best:
                best_reward = mean_reward
            _save_stage2_checkpoint(ckpt_dir, model, optimizer, cfg, epoch_id, mean_loss, mean_reward, is_best)
    finally:
        if run is not None:
            swanlab.finish()


if __name__ == "__main__":
    main()