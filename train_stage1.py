"""Stage1：冻结视觉、训练 DSAM/CGRM（及文本 stub），字幕弱监督对齐。

数据来自 ``QuestionFeatureDataset`` + 离线 SigLIP patch 缓存；主损失为 DSAM 四项加权和，
可选 ``lambda_cgrm_weak`` 将字幕 ``gt_mask`` 池化到事件维并与 ``S_event`` 对齐。
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from dscavl import (
    DSCAVL,
    DSCAVLConfig,
    QueryTextEncoder,
    QuestionFeatureDataset,
    list_arch_variants,
    variable_feature_collate,
    ensure_frozen_visual_encoder,
    get_peak_memory_mb,
    prepare_feature_batch,
    resolve_train_checkpoint_dir,
    Stage1LRState,
    TrainingDiagnostics,
)
from dscavl.swanlab_train import (
    add_swanlab_train_args,
    resolve_swanlab_project,
    swanlab_try_init,
    training_diagnostics_to_log,
)
from dscavl.train_utils import add_cgrm_use_f_bg_cli_args, merge_cgrm_use_f_bg_cfg
from dscavl.cgrm_weak_loss import loss_cgrm_weak
from dscavl.weak_supervision import build_gt_mask_from_subtitles


def loss_policy_salign(
    s_graph_target: torch.Tensor,
    frame_logits: torch.Tensor,
    frame_mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """将策略帧 logits 与 ``S_graph`` 在有效帧上对齐（target 为 detach 的 CGRM 图分数，仅反传到 policy 头）。"""
    t = max(float(temperature), 1e-6)
    m = (frame_mask > 0.5).to(dtype=frame_logits.dtype)
    t_logits = s_graph_target / t + (1.0 - m) * -1e4
    p_logits = frame_logits / t + (1.0 - m) * -1e4
    target = torch.softmax(t_logits, dim=-1)
    log_p = torch.log_softmax(p_logits, dim=-1)
    return -(target * log_p).sum(dim=-1).mean()


def stage1_loss(outputs, cfg: DSCAVLConfig):
    """紧凑/正交/对齐/重建四项损失的线性组合（权重来自 ``cfg``）。

    若 ``lambda_cgrm_mix_entropy > 0``，减去 ``mix_lambdas`` 的 Shannon 熵（batch 均值），
    等价于在最小化总损失时 **增大** 语义/因果混合权重熵，缓解单支塌缩。
    """
    base = (
        cfg.lambda_compact * outputs["loss_compact"]
        + cfg.lambda_orth * outputs["loss_orth"]
        + cfg.lambda_align * outputs["loss_align"]
        + cfg.lambda_recon * outputs["loss_recon"]
    )
    if cfg.lambda_cgrm_mix_entropy <= 0.0:
        return base
    lam = outputs.get("mix_lambdas")
    if not isinstance(lam, torch.Tensor) or lam.numel() == 0:
        return base
    eps = 1e-8
    p = lam.clamp(min=eps, max=1.0 - eps)
    ent = -(p * p.log()).sum(dim=-1).mean()
    return base - cfg.lambda_cgrm_mix_entropy * ent


def build_tokenizer(cfg: DSCAVLConfig, repo_root: Path):
    """解析本地或 Hub 上的 tokenizer 路径；补全 pad_token。"""
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
    """Stage1 打开 ``include_subtitles`` 以便构造 ``gt_mask``。"""
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
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=variable_feature_collate,
    )


@lru_cache(maxsize=131072)
def _cached_gt_mask_bits(
    subtitle_path: str | None,
    ts_key: tuple[float, ...],
    question: str,
    options_key: tuple[str, ...],
    min_hits: int,
    expand_sec: float,
) -> tuple[int, ...]:
    """LRU 缓存的字幕弱监督位图，避免同一视频重复解析 SRT。"""
    if subtitle_path is None or not ts_key:
        return ()
    timestamps = torch.tensor(ts_key, dtype=torch.float32)
    gt_mask = build_gt_mask_from_subtitles(
        subtitle_path=subtitle_path,
        timestamps=timestamps,
        question=question,
        options=list(options_key),
        min_hits=min_hits,
        expand_sec=expand_sec,
    )
    if gt_mask is None:
        return ()
    return tuple(int(x) for x in gt_mask.to(torch.int8).tolist())


def _prepare_stage1_batch(
    batch,
    cfg: DSCAVLConfig,
    device: torch.device,
) -> dict | None:
    """padding 特征、构造 ``gt_mask``/``frame_mask``；无有效特征返回 None。"""
    features_list = batch.get("precomputed_features")
    if features_list is None:
        return None

    valid_indices = [i for i, feat in enumerate(features_list) if feat is not None]
    if not valid_indices:
        return None

    feats_to_pad = [features_list[i] for i in valid_indices]
    padded_features, frame_mask, batch_info = prepare_feature_batch(
        feats_to_pad, padding_value=0.0, device=device
    )
    max_len = padded_features.shape[1]

    input_ids = batch["input_ids"][valid_indices].to(device, non_blocking=True)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask[valid_indices].to(device, non_blocking=True)

    # 与 prepare_feature_batch 返回的 frame_mask 同设备，避免 gt_mask &= frame_mask 时 CPU/CUDA 混用
    gt_mask = torch.zeros((len(valid_indices), max_len), dtype=torch.bool, device=device)
    gt_has_supervision = 0.0
    gt_hit_ratio_sum = 0.0

    subtitle_paths = batch.get("subtitle_path")
    timestamps_list = batch.get("timestamps")
    questions = batch.get("question")
    options_batch = batch.get("options")

    for row, idx in enumerate(valid_indices):
        ts = timestamps_list[idx]
        if ts is None or not isinstance(ts, torch.Tensor):
            continue
        ts_key = tuple(float(x) for x in ts.tolist())
        bits = _cached_gt_mask_bits(
            subtitle_path=subtitle_paths[idx],
            ts_key=ts_key,
            question=questions[idx],
            options_key=tuple(options_batch[idx]),
            min_hits=cfg.weak_sup_min_hits,
            expand_sec=cfg.weak_sup_expand_sec,
        )
        if not bits:
            continue
        sample_mask = torch.tensor(bits, dtype=torch.bool, device=device)
        copy_len = min(sample_mask.numel(), max_len)
        if copy_len == 0:
            continue
        gt_mask[row, :copy_len] = sample_mask[:copy_len]
        gt_mask[row] &= frame_mask[row]
        gt_has_supervision += 1.0
        gt_hit_ratio_sum += float(gt_mask[row].float().mean().item())

    if gt_has_supervision == 0.0:
        gt_mask = None

    return {
        "precomputed_features": padded_features,
        "batch_info": batch_info,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "frame_mask": frame_mask,
        "gt_mask": gt_mask,
        "gt_has_supervision": gt_has_supervision / len(valid_indices),
        "gt_hit_ratio": gt_hit_ratio_sum / len(valid_indices),
    }


def run_stage1_epoch(
    model: DSCAVL,
    optimizer: AdamW,
    dataloader: DataLoader,
    cfg: DSCAVLConfig,
    device: torch.device,
    epoch_id: int,
    sanity_steps: int = 0,
    lr_state: Stage1LRState | None = None,
) -> dict[str, float]:
    """单轮训练循环；含 NaN/梯度检查、可选 ``Stage1LRState`` 与 ``TrainingDiagnostics``。"""
    total_loss = 0.0
    total_compact = 0.0
    total_orth = 0.0
    total_align = 0.0
    total_recon = 0.0
    total_gt_hit_ratio = 0.0
    total_gt_has_supervision = 0.0
    total_cgrm_weak = 0.0
    cgrm_weak_batches = 0
    total_policy_salign = 0.0
    policy_salign_steps = 0
    step_count = 0
    diag = TrainingDiagnostics()

    progress = tqdm(dataloader, desc=f"[Stage1] Epoch {epoch_id}", leave=False)
    for batch in progress:
        cgrm_weak_lw_item: float | None = None
        prepared = _prepare_stage1_batch(batch, cfg, device)
        if prepared is None:
            diag.skipped_empty_batches += 1
            continue

        outputs = model(
            None,
            prepared["input_ids"],
            mode="stage1",
            gt_mask=prepared["gt_mask"],
            frame_mask=prepared["frame_mask"],
            attention_mask=prepared["attention_mask"],
            precomputed_features=prepared["precomputed_features"],
        )
        loss = stage1_loss(outputs, cfg)
        if cfg.lambda_policy_salign > 0.0:
            s_det = outputs["S_graph"].detach()
            h_det = outputs["H_graph"].detach()
            f_det = outputs["f_q"].detach()
            h_ev = outputs.get("H_event")
            h_ev = h_ev.detach() if h_ev is not None else None
            s_ev = outputs.get("S_event")
            s_ev = s_ev.detach() if s_ev is not None else None
            mem = outputs.get("membership")
            mem = mem.detach() if mem is not None else None
            fpa = outputs.get("frame_patch_stats")
            fpa = fpa.detach() if fpa is not None else None
            eva = outputs.get("event_patch_stats")
            eva = eva.detach() if eva is not None else None
            po = model.policy(
                h_det,
                f_det,
                s_det,
                sample=False,
                intra_event_topk=True,
                h_event=h_ev,
                s_event=s_ev,
                membership=mem,
                frame_aux=fpa,
                event_aux=eva,
            )
            l_ps = loss_policy_salign(
                s_det,
                po["frame_logits"],
                prepared["frame_mask"],
                temperature=cfg.policy_salign_temperature,
            )
            if torch.isfinite(l_ps).all():
                loss = loss + cfg.lambda_policy_salign * l_ps
                total_policy_salign += float(l_ps.detach().item())
                policy_salign_steps += 1
        if cfg.lambda_cgrm_weak > 0.0 and prepared["gt_mask"] is not None:
            if cfg.cgrm_weak_detach_inputs:
                fp_stats = outputs.get("frame_patch_stats")
                if fp_stats is not None:
                    fp_stats = fp_stats.detach()
                f_bg_w = None
                if cfg.cgrm_use_f_bg:
                    f_bg_w = outputs.get("F_bg")
                    if f_bg_w is not None:
                        f_bg_w = f_bg_w.detach()
                cg_w = model.cgrm(
                    outputs["F_fg"].detach(),
                    outputs["S_sem"].detach(),
                    outputs["f_q"].detach(),
                    frame_patch_stats=fp_stats,
                    f_bg=f_bg_w,
                )
                s_event_w, membership_w = cg_w["S_event"], cg_w["membership"]
            else:
                s_event_w, membership_w = outputs["S_event"], outputs["membership"]
            lw = loss_cgrm_weak(
                s_event_w,
                membership_w,
                prepared["gt_mask"],
                frame_mask=prepared["frame_mask"],
                event_pool=cfg.cgrm_weak_event_pool,
                loss_type=cfg.cgrm_weak_loss,
                tv_weight=cfg.cgrm_weak_tv_weight,
            )
            if torch.isfinite(lw).all():
                loss = loss + cfg.lambda_cgrm_weak * lw
                cgrm_weak_lw_item = float(lw.detach().item())

        if not torch.isfinite(loss):
            diag.loss_nan_count += 1
            diag.skipped_nonfinite_batches += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_ok = all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in model.parameters() if p.requires_grad
        )
        if not grad_ok:
            diag.skipped_nonfinite_grads += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        if lr_state is not None:
            lr_state.apply_before_step(optimizer)
        optimizer.step()
        if lr_state is not None:
            lr_state.after_step(loss.detach().item())

        total_loss += loss.detach().item()
        total_compact += outputs["loss_compact"].detach().item()
        total_orth += outputs["loss_orth"].detach().item()
        total_align += outputs["loss_align"].detach().item()
        total_recon += outputs["loss_recon"].detach().item()
        total_gt_hit_ratio += prepared["gt_hit_ratio"]
        total_gt_has_supervision += prepared["gt_has_supervision"]
        step_count += 1
        if cgrm_weak_lw_item is not None:
            total_cgrm_weak += cgrm_weak_lw_item
            cgrm_weak_batches += 1
        if prepared.get("batch_info"):
            diag.feature_shapes.append(prepared["batch_info"].shape)
        postfix: dict[str, str] = {
            "loss": f"{(total_loss / max(step_count, 1)):.4f}",
            "gt_hit": f"{(total_gt_hit_ratio / max(step_count, 1)):.4f}",
        }
        if cfg.lambda_cgrm_weak > 0.0 and cgrm_weak_batches > 0:
            postfix["cgrm_w"] = f"{(total_cgrm_weak / cgrm_weak_batches):.4f}"
        if lr_state is not None:
            postfix["lr"] = f"{lr_state.last_lr:.2e}"
        progress.set_postfix(**postfix)
        if sanity_steps > 0 and step_count >= sanity_steps:
            break

    diag.step_count = step_count
    diag.peak_memory_mb = get_peak_memory_mb()
    denom = max(step_count, 1)
    lr_epoch = (
        float(lr_state.last_lr)
        if lr_state is not None and step_count > 0
        else float(optimizer.param_groups[0].get("lr", cfg.lr))
    )
    cgrm_denom = max(cgrm_weak_batches, 1)
    loss_cgrm_weak_avg = (total_cgrm_weak / cgrm_denom) if cfg.lambda_cgrm_weak > 0.0 else 0.0
    ps_denom = max(policy_salign_steps, 1)
    loss_ps_avg = (total_policy_salign / ps_denom) if cfg.lambda_policy_salign > 0.0 else 0.0
    return {
        "loss": total_loss / denom,
        "loss_compact": total_compact / denom,
        "loss_orth": total_orth / denom,
        "loss_align": total_align / denom,
        "loss_recon": total_recon / denom,
        "gt_coverage": total_gt_has_supervision / denom,
        "gt_hit_ratio": total_gt_hit_ratio / denom,
        "loss_cgrm_weak": loss_cgrm_weak_avg,
        "cgrm_weak_batches": float(cgrm_weak_batches),
        "loss_policy_salign": loss_ps_avg,
        "lr": lr_epoch,
        "_diag": diag,
    }


def _save_stage1_checkpoint(
    ckpt_dir: Path,
    model: DSCAVL,
    optimizer: AdamW,
    cfg: DSCAVLConfig,
    epoch: int,
    metrics: dict[str, float],
    is_best: bool,
    save_epoch_file: bool = True,
):
    """写入 checkpoint（含 ``cfg.__dict__``）：可选省略按 epoch 编号的副本以节省磁盘。"""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_metrics = {k: v for k, v in metrics.items() if k != "_diag"}
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg.__dict__,
        "metrics": save_metrics,
    }
    epoch_path = ckpt_dir / f"checkpoint-epoch{epoch}.pt"
    latest_path = ckpt_dir / "checkpoint-latest.pt"
    if save_epoch_file:
        torch.save(payload, epoch_path)
    torch.save(payload, latest_path)
    if is_best:
        torch.save(payload, ckpt_dir / "checkpoint-best.pt")


def parse_args() -> argparse.Namespace:
    """命令行入口：数据根、特征根、sanity 步数、SwanLab 开关。"""
    parser = argparse.ArgumentParser(description="Train DSCA-VL stage1.")
    parser.add_argument("--data-root", type=str, default=None, help="Override cfg.data_root")
    parser.add_argument("--feature-root", type=str, default=None, help="Override cfg.feature_root")
    parser.add_argument(
        "--arch-variant",
        type=str,
        default=None,
        choices=list_arch_variants(),
        help="Architecture variant for module ablation. Default: cfg.arch_variant.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory. Default: output/stage1/<YYYYMMDD_HHMMSS> under repo root.",
    )
    parser.add_argument(
        "--sanity-steps",
        type=int,
        default=0,
        help="Run only N optimizer steps per epoch for quick sanity validation (<=0 means full epoch).",
    )
    add_swanlab_train_args(parser)
    parser.add_argument(
        "--no-stage1-lr-schedule",
        action="store_true",
        help="Disable warmup+cosine LR schedule; keep constant cfg.lr.",
    )
    parser.add_argument(
        "--stage1-lr-peak",
        type=float,
        default=None,
        help="Peak LR after warmup (sets cfg.lr for this run). Default: cfg.lr.",
    )
    parser.add_argument(
        "--stage1-lr-min",
        type=float,
        default=None,
        help="Cosine floor LR. Default: cfg.stage1_lr_min.",
    )
    parser.add_argument(
        "--stage1-warmup-ratio",
        type=float,
        default=None,
        help="Warmup steps = ratio * (epochs * steps_per_epoch). Default: cfg.stage1_warmup_ratio.",
    )
    parser.add_argument(
        "--stage1-lr-loss-reweight",
        type=float,
        default=None,
        help="Blend factor for loss-EMA-based LR scaling (0=off). Default: cfg.stage1_lr_loss_reweight.",
    )
    add_cgrm_use_f_bg_cli_args(parser)
    parser.add_argument(
        "--lambda-cgrm-weak",
        type=float,
        default=None,
        help="Weight for CGRM weak supervision (gt_mask → events vs S_event). Default: cfg.lambda_cgrm_weak.",
    )
    parser.add_argument(
        "--cgrm-weak-event-pool",
        type=str,
        default=None,
        choices=["mean", "max"],
        help="Pool frames to event targets: mean (match EventPooler) or max. Default: cfg.cgrm_weak_event_pool.",
    )
    parser.add_argument(
        "--cgrm-weak-loss",
        type=str,
        default=None,
        choices=["bce_logits", "mse"],
        help="Weak loss: BCEWithLogits on S_event vs pooled targets, or MSE after per-sample min-max on S_event.",
    )
    parser.add_argument(
        "--cgrm-weak-tv-weight",
        type=float,
        default=None,
        help="Optional L1 smoothness on S_event across adjacent events. Default: cfg.cgrm_weak_tv_weight.",
    )
    parser.add_argument(
        "--cgrm-weak-e2e",
        action="store_true",
        help="Weak loss 也经 CGRM 回传到 DSAM/查询编码（默认关闭：detach 输入、弱监督仅更新 CGRM，更稳）。",
    )
    parser.add_argument(
        "--cgrm-mix-temperature",
        type=float,
        default=None,
        help="CGRM semantic/causal mix softmax temperature (>1 softens). Default: cfg.cgrm_mix_temperature.",
    )
    parser.add_argument(
        "--lambda-cgrm-mix-entropy",
        type=float,
        default=None,
        help="Stage1: weight for maximizing entropy of mix_lambdas (0=off). Default: cfg.lambda_cgrm_mix_entropy.",
    )
    parser.add_argument(
        "--lambda-policy-salign",
        type=float,
        default=None,
        help="Stage1: distill policy logits toward CGRM S_graph (only policy grads). Default: cfg.lambda_policy_salign.",
    )
    parser.add_argument(
        "--policy-salign-temperature",
        type=float,
        default=None,
        help="Softmax temperature for S_graph vs frame_logits alignment. Default: cfg.policy_salign_temperature.",
    )
    # 可验收计划 §6 infer / §5 CGRM 桥接：落盘到 cfg，供 checkpoint 与 compare 前向一致
    parser.add_argument(
        "--bridge-weight",
        type=float,
        default=None,
        help="CGRM bridge_weight（DSAM↔图）。Default: cfg.bridge_weight。",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=None,
        help="CGRM S_graph 中语义项权重。Default: cfg.beta1。",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=None,
        help="CGRM S_graph 中中心性项权重。Default: cfg.beta2。",
    )
    parser.add_argument(
        "--keep-ratio-low",
        type=float,
        default=None,
        help="infer：每视频最少选中比例下限。Default: cfg.keep_ratio_low。",
    )
    parser.add_argument(
        "--keep-ratio-high",
        type=float,
        default=None,
        help="infer：每视频选中 budget 上界（与 T 相乘）。Default: cfg.keep_ratio_high。",
    )
    parser.add_argument(
        "--min-event-ratio",
        type=float,
        default=None,
        help="infer：至少选中的事件数占事件总数比例。Default: cfg.min_event_ratio。",
    )
    parser.add_argument(
        "--max-per-event",
        type=int,
        default=None,
        help="infer：每事件内最多取帧数。Default: cfg.max_per_event。",
    )
    parser.add_argument(
        "--lambda-orth",
        type=float,
        default=None,
        help="DSAM loss_orth 权重。Default: cfg.lambda_orth。",
    )
    parser.add_argument(
        "--lambda-compact",
        type=float,
        default=None,
        help="DSAM loss_compact 权重。Default: cfg.lambda_compact。",
    )
    parser.add_argument(
        "--lambda-align",
        type=float,
        default=None,
        help="DSAM loss_align 权重。Default: cfg.lambda_align。",
    )
    parser.add_argument(
        "--lambda-recon",
        type=float,
        default=None,
        help="DSAM loss_recon 权重。Default: cfg.lambda_recon。",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=None,
        help="覆盖 cfg.stage1_epochs（默认同 DSCAVLConfig，通常为 30）。",
    )
    parser.add_argument(
        "--no-epoch-checkpoints",
        action="store_true",
        help="不写入 checkpoint-epoch{k}.pt，仅保留 checkpoint-latest.pt 与（若 is_best）checkpoint-best.pt，省磁盘。",
    )
    return parser.parse_args()


def main():
    """构建模型与 DataLoader，多 epoch 训练并记录 SwanLab（可选）。"""
    args = parse_args()
    cfg = DSCAVLConfig()
    if args.data_root:
        cfg.data_root = args.data_root
    if args.feature_root:
        cfg.feature_root = args.feature_root
    if args.arch_variant is not None:
        cfg.arch_variant = args.arch_variant
    if args.stage1_lr_peak is not None:
        cfg.lr = float(args.stage1_lr_peak)
    if args.stage1_lr_min is not None:
        cfg.stage1_lr_min = float(args.stage1_lr_min)
    if args.stage1_warmup_ratio is not None:
        cfg.stage1_warmup_ratio = float(args.stage1_warmup_ratio)
    if args.stage1_lr_loss_reweight is not None:
        cfg.stage1_lr_loss_reweight = float(args.stage1_lr_loss_reweight)
    merge_cgrm_use_f_bg_cfg(cfg, None, args.cgrm_use_f_bg, args.no_cgrm_f_bg)
    if args.lambda_cgrm_weak is not None:
        cfg.lambda_cgrm_weak = float(args.lambda_cgrm_weak)
    if args.cgrm_weak_event_pool is not None:
        cfg.cgrm_weak_event_pool = args.cgrm_weak_event_pool
    if args.cgrm_weak_loss is not None:
        cfg.cgrm_weak_loss = args.cgrm_weak_loss
    if args.cgrm_weak_tv_weight is not None:
        cfg.cgrm_weak_tv_weight = float(args.cgrm_weak_tv_weight)
    if args.cgrm_weak_e2e:
        cfg.cgrm_weak_detach_inputs = False
    if args.cgrm_mix_temperature is not None:
        cfg.cgrm_mix_temperature = float(args.cgrm_mix_temperature)
    if args.lambda_cgrm_mix_entropy is not None:
        cfg.lambda_cgrm_mix_entropy = float(args.lambda_cgrm_mix_entropy)
    if args.lambda_policy_salign is not None:
        cfg.lambda_policy_salign = float(args.lambda_policy_salign)
    if args.policy_salign_temperature is not None:
        cfg.policy_salign_temperature = float(args.policy_salign_temperature)
    if args.bridge_weight is not None:
        cfg.bridge_weight = float(args.bridge_weight)
    if args.beta1 is not None:
        cfg.beta1 = float(args.beta1)
    if args.beta2 is not None:
        cfg.beta2 = float(args.beta2)
    if args.keep_ratio_low is not None:
        cfg.keep_ratio_low = float(args.keep_ratio_low)
    if args.keep_ratio_high is not None:
        cfg.keep_ratio_high = float(args.keep_ratio_high)
    if args.min_event_ratio is not None:
        cfg.min_event_ratio = float(args.min_event_ratio)
    if args.max_per_event is not None:
        cfg.max_per_event = int(args.max_per_event)
    if args.lambda_orth is not None:
        cfg.lambda_orth = float(args.lambda_orth)
    if args.lambda_compact is not None:
        cfg.lambda_compact = float(args.lambda_compact)
    if args.lambda_align is not None:
        cfg.lambda_align = float(args.lambda_align)
    if args.lambda_recon is not None:
        cfg.lambda_recon = float(args.lambda_recon)
    if args.stage1_epochs is not None:
        cfg.stage1_epochs = int(args.stage1_epochs)

    repo_root = Path(__file__).resolve().parent
    data_root = repo_root / cfg.data_root
    feature_root = repo_root / cfg.feature_root

    tokenizer = build_tokenizer(cfg, repo_root)
    dataloader = build_dataloader(cfg, tokenizer, data_root, feature_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = QueryTextEncoder(vocab_size=len(tokenizer), dim=cfg.dim)
    model = DSCAVL(cfg, text_encoder=text_encoder).to(device)

    if not ensure_frozen_visual_encoder(model):
        raise RuntimeError(
            "[Stage1] Visual encoder (SigLIP) must be frozen for stable training. "
            "Use precomputed features only."
        )
    print("[Stage1] Visual encoder frozen (precomputed features).")
    print(
        f"[Stage1] arch_variant={cfg.arch_variant} "
        f"isoclip={cfg.use_isoclip_debias} slerp={cfg.use_riemann_slerp} "
        f"soft_mask={cfg.use_soft_mask} sbp={cfg.use_sbp}"
    )
    print(f"[Stage1] cgrm_use_f_bg={cfg.cgrm_use_f_bg}")
    print(
        f"[Stage1] bridge_weight={cfg.bridge_weight}, beta1={cfg.beta1}, beta2={cfg.beta2}, "
        f"infer keep=({cfg.keep_ratio_low},{cfg.keep_ratio_high}) "
        f"min_event_ratio={cfg.min_event_ratio} max_per_event={cfg.max_per_event}\n"
        f"[Stage1] lambda_policy_salign={cfg.lambda_policy_salign}, "
        f"policy_salign_temperature={cfg.policy_salign_temperature}"
    )
    print(
        f"[Stage1] lambda_cgrm_weak={cfg.lambda_cgrm_weak}, "
        f"cgrm_weak_event_pool={cfg.cgrm_weak_event_pool}, "
        f"cgrm_weak_loss={cfg.cgrm_weak_loss}, "
        f"cgrm_weak_tv_weight={cfg.cgrm_weak_tv_weight}, "
        f"cgrm_weak_detach_inputs={cfg.cgrm_weak_detach_inputs}"
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    use_lr_schedule = cfg.stage1_cosine_lr_schedule and not args.no_stage1_lr_schedule
    total_opt_steps = max(1, cfg.stage1_epochs * len(dataloader))
    warmup_steps = int(cfg.stage1_warmup_ratio * total_opt_steps)
    warmup_steps = min(max(0, warmup_steps), max(0, total_opt_steps - 1))
    lr_state: Stage1LRState | None = None
    if use_lr_schedule:
        lr_state = Stage1LRState(
            lr_peak=float(cfg.lr),
            lr_min=float(cfg.stage1_lr_min),
            warmup_steps=warmup_steps,
            total_steps=total_opt_steps,
            loss_reweight=float(cfg.stage1_lr_loss_reweight),
            loss_ema_beta=float(cfg.stage1_lr_loss_ema_beta),
            loss_blend_floor=float(cfg.stage1_lr_loss_blend_floor),
        )
        print(
            "[Stage1] LR schedule: linear warmup + cosine decay + optional loss EMA scale | "
            f"total_steps={total_opt_steps}, warmup_steps={warmup_steps}, "
            f"peak={cfg.lr:.3e}, min={cfg.stage1_lr_min:.3e}, "
            f"loss_reweight={cfg.stage1_lr_loss_reweight}"
        )
    else:
        print(f"[Stage1] LR schedule: off (constant lr={cfg.lr:.3e})")

    optimizer = AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ckpt_dir = resolve_train_checkpoint_dir(repo_root, args.checkpoint_dir, "stage1")
    print(f"[Stage1] checkpoint_dir={ckpt_dir}")

    project = resolve_swanlab_project(args, cfg)
    swan_ctx = swanlab_try_init(
        disabled=args.disable_swanlab,
        project=project,
        experiment_name=args.swanlab_experiment_name,
        experiment_name_prefix="stage1",
        workspace=args.swanlab_workspace,
        log_prefix="Stage1",
        config={
            "stage": "stage1",
            "swanlab_project": project,
            "data_root": str(data_root),
            "feature_root": str(feature_root),
            "epochs": cfg.stage1_epochs,
            "batch_size": cfg.train_batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "stage1_cosine_lr_schedule": use_lr_schedule,
            "stage1_lr_min": cfg.stage1_lr_min,
            "stage1_warmup_ratio": cfg.stage1_warmup_ratio,
            "stage1_lr_loss_reweight": cfg.stage1_lr_loss_reweight,
            "stage1_total_schedule_steps": total_opt_steps,
            "text_model_name_or_path": cfg.text_model_name_or_path,
            "lambda_compact": cfg.lambda_compact,
            "lambda_orth": cfg.lambda_orth,
            "lambda_align": cfg.lambda_align,
            "lambda_recon": cfg.lambda_recon,
            "lambda_cgrm_weak": cfg.lambda_cgrm_weak,
            "cgrm_weak_event_pool": cfg.cgrm_weak_event_pool,
            "cgrm_weak_loss": cfg.cgrm_weak_loss,
            "cgrm_weak_tv_weight": cfg.cgrm_weak_tv_weight,
            "cgrm_weak_detach_inputs": cfg.cgrm_weak_detach_inputs,
            "cgrm_use_f_bg": cfg.cgrm_use_f_bg,
            "lambda_policy_salign": cfg.lambda_policy_salign,
            "policy_salign_temperature": cfg.policy_salign_temperature,
            "checkpoint_dir": str(ckpt_dir),
        },
    )

    best_loss = float("inf")
    try:
        for epoch in range(cfg.stage1_epochs):
            epoch_id = epoch + 1
            metrics = run_stage1_epoch(
                model,
                optimizer,
                dataloader,
                cfg,
                device,
                epoch_id,
                args.sanity_steps,
                lr_state=lr_state,
            )
            print(
                f"[Stage1] epoch={epoch_id}, "
                f"loss={metrics['loss']:.4f}, "
                f"lr={metrics['lr']:.3e}, "
                f"compact={metrics['loss_compact']:.4f}, "
                f"orth={metrics['loss_orth']:.4f}, "
                f"align={metrics['loss_align']:.4f}, "
                f"recon={metrics['loss_recon']:.4f}, "
                f"gt_coverage={metrics['gt_coverage']:.4f}, "
                f"gt_hit_ratio={metrics['gt_hit_ratio']:.4f}, "
                f"loss_cgrm_weak={metrics['loss_cgrm_weak']:.4f}, "
                f"cgrm_weak_batches={int(metrics['cgrm_weak_batches'])}, "
                f"loss_policy_salign={metrics['loss_policy_salign']:.4f}"
            )
            diag = metrics.get("_diag")
            if diag is not None and (
                diag.step_count > 0
                or diag.skipped_empty_batches > 0
                or diag.skipped_nonfinite_batches > 0
                or diag.skipped_nonfinite_grads > 0
                or diag.skipped_nonfinite_samples > 0
            ):
                print(f"[Stage1][diag] {diag.to_log_str()}")

            log_data = {
                "stage1/epoch": epoch_id,
                "stage1/loss": metrics["loss"],
                "stage1/lr": metrics["lr"],
                "stage1/loss_compact": metrics["loss_compact"],
                "stage1/loss_orth": metrics["loss_orth"],
                "stage1/loss_align": metrics["loss_align"],
                "stage1/loss_recon": metrics["loss_recon"],
                "stage1/gt_coverage": metrics["gt_coverage"],
                "stage1/gt_hit_ratio": metrics["gt_hit_ratio"],
                "stage1/loss_cgrm_weak": metrics["loss_cgrm_weak"],
                "stage1/cgrm_weak_batches": metrics["cgrm_weak_batches"],
                "stage1/lambda_cgrm_weak": cfg.lambda_cgrm_weak,
                "stage1/cgrm_weak_detach_inputs": float(cfg.cgrm_weak_detach_inputs),
                "stage1/loss_policy_salign": metrics["loss_policy_salign"],
            }
            log_data.update(training_diagnostics_to_log(metrics.get("_diag"), "stage1"))
            swan_ctx.log(log_data, log_prefix="Stage1")

            is_best = metrics["loss"] < best_loss
            if is_best:
                best_loss = metrics["loss"]
            _save_stage1_checkpoint(
                ckpt_dir,
                model,
                optimizer,
                cfg,
                epoch_id,
                metrics,
                is_best,
                save_epoch_file=not args.no_epoch_checkpoints,
            )
    finally:
        swan_ctx.finish(log_prefix="Stage1")


if __name__ == "__main__":
    main()