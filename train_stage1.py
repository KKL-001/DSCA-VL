from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dscavl import DSCAVL, DSCAVLConfig, QueryTextEncoder, QuestionFeatureDataset, variable_feature_collate
from dscavl.weak_supervision import build_gt_mask_from_subtitles

try:
    import swanlab
except ImportError:  # pragma: no cover - optional dependency
    swanlab = None


def stage1_loss(outputs, cfg: DSCAVLConfig):
    return (
        cfg.lambda_compact * outputs["loss_compact"]
        + cfg.lambda_orth * outputs["loss_orth"]
        + cfg.lambda_align * outputs["loss_align"]
    )


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
        include_subtitles=True,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=variable_feature_collate,
    )


def _build_stage1_gt_mask(batch, index: int, cfg: DSCAVLConfig, device: torch.device):
    subtitle_paths = batch.get("subtitle_path")
    timestamps_list = batch.get("timestamps")
    questions = batch.get("question")
    options_batch = batch.get("options")
    gt_mask = build_gt_mask_from_subtitles(
        subtitle_path=subtitle_paths[index],
        timestamps=timestamps_list[index],
        question=questions[index],
        options=options_batch[index],
        min_hits=cfg.weak_sup_min_hits,
        expand_sec=cfg.weak_sup_expand_sec,
    )
    if gt_mask is not None:
        gt_mask = gt_mask.unsqueeze(0).to(device)
    return gt_mask


def _compute_stage1_sample_loss(model: DSCAVL, cfg: DSCAVLConfig, batch, index: int, input_ids: torch.Tensor, attention_mask, device: torch.device):
    features = batch["precomputed_features"][index]
    if features is None:
        return None

    precomputed_features = features.unsqueeze(0).to(device)
    query = input_ids[index : index + 1]
    query_mask = attention_mask[index : index + 1] if attention_mask is not None else None
    gt_mask = _build_stage1_gt_mask(batch, index, cfg, device)
    if gt_mask is not None:
        gt_hit_ratio = float(gt_mask.float().mean().item())
        gt_has_supervision = 1.0
    else:
        gt_hit_ratio = 0.0
        gt_has_supervision = 0.0

    outputs = model(
        None,
        query,
        mode="stage1",
        gt_mask=gt_mask,
        attention_mask=query_mask,
        precomputed_features=precomputed_features,
    )
    return {
        "loss": stage1_loss(outputs, cfg),
        "loss_compact": outputs["loss_compact"],
        "loss_orth": outputs["loss_orth"],
        "loss_align": outputs["loss_align"],
        "gt_hit_ratio": gt_hit_ratio,
        "gt_has_supervision": gt_has_supervision,
    }


def run_stage1_epoch(model: DSCAVL, optimizer: AdamW, dataloader: DataLoader, cfg: DSCAVLConfig, device: torch.device) -> dict[str, float]:
    total_loss = 0.0
    total_compact = 0.0
    total_orth = 0.0
    total_align = 0.0
    total_gt_hit_ratio = 0.0
    total_gt_has_supervision = 0.0
    step_count = 0

    for batch in dataloader:
        features_list = batch.get("precomputed_features")
        if features_list is None:
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        batch_loss = torch.zeros((), device=device)
        batch_compact = 0.0
        batch_orth = 0.0
        batch_align = 0.0
        batch_gt_hit_ratio = 0.0
        batch_gt_has_supervision = 0.0
        valid_count = 0

        for i, _ in enumerate(features_list):
            sample_stats = _compute_stage1_sample_loss(model, cfg, batch, i, input_ids, attention_mask, device)
            if sample_stats is None:
                continue
            batch_loss = batch_loss + sample_stats["loss"]
            batch_compact += sample_stats["loss_compact"].detach().item()
            batch_orth += sample_stats["loss_orth"].detach().item()
            batch_align += sample_stats["loss_align"].detach().item()
            batch_gt_hit_ratio += sample_stats["gt_hit_ratio"]
            batch_gt_has_supervision += sample_stats["gt_has_supervision"]
            valid_count += 1

        if valid_count == 0:
            continue

        loss = batch_loss / valid_count

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
        total_compact += batch_compact / valid_count
        total_orth += batch_orth / valid_count
        total_align += batch_align / valid_count
        total_gt_hit_ratio += batch_gt_hit_ratio / valid_count
        total_gt_has_supervision += batch_gt_has_supervision / valid_count
        step_count += 1

    denom = max(step_count, 1)
    return {
        "loss": total_loss / denom,
        "loss_compact": total_compact / denom,
        "loss_orth": total_orth / denom,
        "loss_align": total_align / denom,
        "gt_coverage": total_gt_has_supervision / denom,
        "gt_hit_ratio": total_gt_hit_ratio / denom,
    }


def _save_stage1_checkpoint(
    ckpt_dir: Path,
    model: DSCAVL,
    optimizer: AdamW,
    cfg: DSCAVLConfig,
    epoch: int,
    metrics: dict[str, float],
    is_best: bool,
):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ckpt_dir = repo_root / "output" / "stage1"

    run = None
    if swanlab is not None:
        run = swanlab.init(
            project="DSCA-VL",
            experiment_name=f"stage1-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "stage": "stage1",
                "data_root": str(data_root),
                "feature_root": str(feature_root),
                "epochs": cfg.stage1_epochs,
                "batch_size": cfg.train_batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "text_model_name_or_path": cfg.text_model_name_or_path,
                "lambda_compact": cfg.lambda_compact,
                "lambda_orth": cfg.lambda_orth,
                "lambda_align": cfg.lambda_align,
            },
        )
    else:
        print("[Stage1] swanlab not installed, skip experiment tracking.")

    best_loss = float("inf")
    try:
        for epoch in range(cfg.stage1_epochs):
            epoch_id = epoch + 1
            metrics = run_stage1_epoch(model, optimizer, dataloader, cfg, device)
            print(
                f"[Stage1] epoch={epoch_id}, "
                f"loss={metrics['loss']:.4f}, "
                f"compact={metrics['loss_compact']:.4f}, "
                f"orth={metrics['loss_orth']:.4f}, "
                f"align={metrics['loss_align']:.4f}, "
                f"gt_coverage={metrics['gt_coverage']:.4f}, "
                f"gt_hit_ratio={metrics['gt_hit_ratio']:.4f}"
            )

            log_data = {
                "stage1/epoch": epoch_id,
                "stage1/loss": metrics["loss"],
                "stage1/loss_compact": metrics["loss_compact"],
                "stage1/loss_orth": metrics["loss_orth"],
                "stage1/loss_align": metrics["loss_align"],
                "stage1/gt_coverage": metrics["gt_coverage"],
                "stage1/gt_hit_ratio": metrics["gt_hit_ratio"],
            }
            if run is not None:
                swanlab.log(log_data)

            is_best = metrics["loss"] < best_loss
            if is_best:
                best_loss = metrics["loss"]
            _save_stage1_checkpoint(ckpt_dir, model, optimizer, cfg, epoch_id, metrics, is_best)
    finally:
        if run is not None:
            swanlab.finish()


if __name__ == "__main__":
    main()