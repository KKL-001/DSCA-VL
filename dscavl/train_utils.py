"""Stage1/Stage2 共用：变长特征 padding、梯度/显存诊断、视觉编码器冻结检查、checkpoint 目录解析。

在 SigLIP 冻结、主用离线 ``[T,P,D]`` 特征时保证 batch 维与 mask 一致。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from .config import DSCAVLConfig


@dataclass
class FeatureBatchInfo:
    """``prepare_feature_batch`` 返回的 shape/layout 元信息，便于日志与断言。"""

    shape: Tuple[int, ...]
    layout: str  # "T,D" or "T,P,D"
    num_frames: int
    patch_tokens_per_frame: Optional[int] = None
    batch_size: int = 0


@dataclass
class TrainingDiagnostics:
    """训练过程累计：跳过 batch/梯度次数、NaN 计数、特征 shape 样本、峰值显存。"""

    step_count: int = 0
    skipped_nonfinite_samples: int = 0
    skipped_nonfinite_batches: int = 0
    skipped_nonfinite_grads: int = 0
    skipped_empty_batches: int = 0
    loss_nan_count: int = 0
    grad_nan_count: int = 0
    feature_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    peak_memory_mb: float = 0.0

    def to_log_str(self) -> str:
        """拼成单行人类可读诊断字符串。"""
        parts = [
            f"steps={self.step_count}",
            f"skip_nonfinite_samples={self.skipped_nonfinite_samples}",
            f"skip_nonfinite_batches={self.skipped_nonfinite_batches}",
            f"skip_nonfinite_grads={self.skipped_nonfinite_grads}",
            f"skip_empty_batches={self.skipped_empty_batches}",
        ]
        if self.loss_nan_count > 0:
            parts.append(f"loss_nan={self.loss_nan_count}")
        if self.grad_nan_count > 0:
            parts.append(f"grad_nan={self.grad_nan_count}")
        if self.peak_memory_mb > 0:
            parts.append(f"peak_mem_mb={self.peak_memory_mb:.1f}")
        return ", ".join(parts)


@dataclass
class Stage1LRState:
    """Stage1：线性预热到 ``lr_peak``，再按余弦退火至 ``lr_min``；可选按 loss EMA 温和缩放。

    在每次成功 ``backward`` 之后、``optimizer.step()`` 之前调用 ``apply_before_step``，
    在 ``optimizer.step()`` 之后用当前 batch 标量 loss 调用 ``after_step``。
    """

    lr_peak: float
    lr_min: float
    warmup_steps: int
    total_steps: int
    loss_reweight: float = 0.0
    loss_ema_beta: float = 0.98
    loss_blend_floor: float = 0.12
    global_step: int = 0
    loss_ref: Optional[float] = None
    loss_ema: Optional[float] = None
    last_lr: float = 0.0

    def __post_init__(self) -> None:
        self.lr_peak = float(max(self.lr_peak, self.lr_min))
        self.lr_min = float(max(self.lr_min, 0.0))
        self.total_steps = max(1, int(self.total_steps))
        ws = max(0, int(self.warmup_steps))
        self.warmup_steps = min(ws, max(0, self.total_steps - 1))
        self.loss_reweight = max(0.0, min(1.0, float(self.loss_reweight)))
        self.loss_ema_beta = max(0.5, min(0.9999, float(self.loss_ema_beta)))
        self.loss_blend_floor = max(1e-3, min(1.0, float(self.loss_blend_floor)))

    def _cosine_base_lr(self) -> float:
        s = self.global_step
        ws = self.warmup_steps
        if ws > 0 and s < ws:
            return self.lr_peak * float(s + 1) / float(ws)
        cos_step = s - ws
        cos_total = max(1, self.total_steps - ws)
        progress = min(1.0, cos_step / float(cos_total))
        return self.lr_min + (self.lr_peak - self.lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def apply_before_step(self, optimizer: torch.optim.Optimizer) -> float:
        lr = self._cosine_base_lr()
        if self.loss_reweight > 0.0 and self.loss_ema is not None and self.loss_ref is not None:
            ratio = self.loss_ema / max(self.loss_ref, 1e-12)
            blend = max(self.loss_blend_floor, min(1.0, ratio))
            lr *= 1.0 - self.loss_reweight + self.loss_reweight * blend
        lr = max(self.lr_min, float(lr))
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        self.last_lr = lr
        return lr

    def after_step(self, loss_value: float) -> None:
        lv = float(loss_value)
        if not math.isfinite(lv):
            self.global_step += 1
            return
        if self.loss_ema is None:
            self.loss_ref = lv
            self.loss_ema = lv
        else:
            b = self.loss_ema_beta
            self.loss_ema = b * self.loss_ema + (1.0 - b) * lv
        self.global_step += 1


def prepare_feature_batch(
    features_list: List[torch.Tensor],
    padding_value: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, FeatureBatchInfo]:
    """将多条变长 ``[T,...]`` 右填充为 ``[B,T,...]``，并生成帧有效 ``frame_mask``。"""
    valid = [i for i, f in enumerate(features_list) if f is not None and f.numel() > 0]
    if not valid:
        raise ValueError("No valid features in batch")

    feats = [features_list[i].float() for i in valid]
    lengths = torch.tensor([x.shape[0] for x in feats], dtype=torch.long)
    padded = pad_sequence(feats, batch_first=True, padding_value=padding_value)
    max_len = padded.shape[1]
    frame_mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

    layout = "T,D" if padded.ndim == 3 else "T,P,D"
    patch_tokens = int(padded.shape[2]) if padded.ndim == 4 else None
    info = FeatureBatchInfo(
        shape=tuple(padded.shape),
        layout=layout,
        num_frames=int(max_len),
        patch_tokens_per_frame=patch_tokens,
        batch_size=len(valid),
    )

    if device is not None:
        padded = padded.to(device, non_blocking=True)
        frame_mask = frame_mask.to(device, non_blocking=True)

    return padded, frame_mask, info


def prepare_single_feature(
    features: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, FeatureBatchInfo]:
    """单样本加 batch 维 ``[1,T,...]``，供 Stage2 逐条 rollout。"""
    if features is None or features.numel() == 0:
        raise ValueError("Empty or None features")
    feats = features.float().unsqueeze(0)
    layout = "T,D" if feats.ndim == 3 else "T,P,D"
    patch_tokens = int(feats.shape[2]) if feats.ndim == 4 else None
    info = FeatureBatchInfo(
        shape=tuple(feats.shape),
        layout=layout,
        num_frames=int(feats.shape[1]),
        patch_tokens_per_frame=patch_tokens,
        batch_size=1,
    )
    if device is not None:
        feats = feats.to(device, non_blocking=True)
    return feats, info


def ensure_frozen_visual_encoder(model: torch.nn.Module) -> bool:
    """若存在 ``vis_encoder``，要求全部 ``requires_grad=False``。"""
    vis = getattr(model, "vis_encoder", None)
    if vis is None:
        return True
    for p in vis.parameters():
        if p.requires_grad:
            return False
    return True


def check_gradients_finite(model: torch.nn.Module) -> bool:
    """遍历已有 ``.grad`` 的张量，全部有限则 True。"""
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


def get_peak_memory_mb() -> float:
    """CUDA 下返回 ``max_memory_allocated`` 的 MB 数；非 CUDA 返回 0。"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def log_batch_shape_info(
    batch_info: Optional[FeatureBatchInfo],
    prefix: str = "[train]",
) -> None:
    """打印当前 batch 特征 shape 与 layout（可选 P）。"""
    if batch_info is None:
        return
    msg = f"{prefix} features shape={batch_info.shape} layout={batch_info.layout}"
    if batch_info.patch_tokens_per_frame is not None:
        msg += f" P={batch_info.patch_tokens_per_frame}"
    print(msg)


def resolve_train_checkpoint_dir(
    repo_root: Path,
    checkpoint_dir: str | None,
    stage_leaf: str,
) -> Path:
    """解析训练 checkpoint 根目录。

    - 若 ``checkpoint_dir`` 非空：相对 ``repo_root`` 拼接后 resolve；已是绝对路径则不变。
    - 否则：``{repo_root}/output/{stage_leaf}/{YYYYMMDD_HHMMSS}``（每次运行新建时间子目录）。
    """
    if checkpoint_dir is not None and str(checkpoint_dir).strip():
        p = Path(checkpoint_dir)
        out = p.resolve() if p.is_absolute() else (repo_root / p).resolve()
        return out
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (repo_root / "output" / stage_leaf / ts).resolve()


def add_cgrm_use_f_bg_cli_args(parser: Any) -> None:
    """互斥开关：``--cgrm-use-f-bg`` / ``--no-cgrm-f-bg``（均未指定时由 checkpoint cfg 或默认决定）。"""
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--cgrm-use-f-bg",
        action="store_true",
        help="CGRM 使用 fg_bg_fuse(cat(F_fg,F_bg))；需与含该权重的 Stage1 checkpoint 一致。",
    )
    g.add_argument(
        "--no-cgrm-f-bg",
        action="store_true",
        help="CGRM 仅用 F_fg（与旧实验 / 未训练 fg_bg_fuse 的 checkpoint 对齐）。",
    )


def merge_cgrm_use_f_bg_cfg(
    cfg: "DSCAVLConfig",
    stage1_payload: Optional[dict],
    cli_use_f_bg: bool,
    cli_no_f_bg: bool,
) -> None:
    """CLI 优先；否则从 Stage1 checkpoint 的 ``cfg`` 字典恢复 ``cgrm_use_f_bg``。"""
    if cli_use_f_bg:
        cfg.cgrm_use_f_bg = True
    elif cli_no_f_bg:
        cfg.cgrm_use_f_bg = False
    elif isinstance(stage1_payload, dict):
        inner = stage1_payload.get("cfg")
        if isinstance(inner, dict) and "cgrm_use_f_bg" in inner:
            cfg.cgrm_use_f_bg = bool(inner["cgrm_use_f_bg"])


def model_state_dict_from_stage1_payload(payload: Any) -> Dict[str, torch.Tensor]:
    """从 Stage1/Stage2 保存的整包 dict 中取出 ``model_state_dict``；已是 state_dict 则原样返回。"""
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dict")
    if "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload  # raw state_dict


def merge_saved_config_dict(cfg: "DSCAVLConfig", payload: Any) -> None:
    """将 checkpoint 中 ``cfg`` 字典合并到 ``DSCAVLConfig``（只覆盖同名字段，避免训练/推理口径不一致）。

    ``train_stage1`` / ``train_stage2*`` 保存的 payload 含 ``cfg`` 与 ``model_state_dict``。
    评测脚本若仍用类默认值，会出现 ``cgrm_mix_temperature``、``keep_ratio_*`` 等与训练时前向不一致，
    选帧会偏离训练期行为。
    """
    if not isinstance(payload, dict):
        return
    inner = payload.get("cfg")
    if not isinstance(inner, dict):
        return
    fields = getattr(type(cfg), "__dataclass_fields__", None) or {}
    for k, v in inner.items():
        if k not in fields:
            continue
        try:
            setattr(cfg, k, v)
        except (TypeError, ValueError):
            pass
