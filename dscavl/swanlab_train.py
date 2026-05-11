"""SwanLab 可选接入：自定义训练循环统一 init / log / finish。

未安装 ``swanlab`` 或 ``--disable-swanlab`` 时全程无操作；init/log 失败会降级为打印提示并尽量 ``finish``。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    import argparse

    from .train_utils import TrainingDiagnostics

try:
    import swanlab as _sw
except ImportError:  # pragma: no cover - optional dependency
    _sw = None


@dataclass
class SwanLabTrainContext:
    """``init`` 成功则需在结束时 ``finish``；单次 ``log`` 失败会关闭后续 log，仍会 ``finish``。"""

    _need_finish: bool
    _log_enabled: bool = True

    @property
    def active(self) -> bool:
        return self._need_finish

    def log(self, data: Dict[str, Any], *, log_prefix: str = "SwanLab") -> None:
        if not self._need_finish or not self._log_enabled or _sw is None:
            return
        try:
            _sw.log(data)
        except Exception as exc:  # pragma: no cover - runtime / network
            print(f"[{log_prefix}] swanlab log failed, disable further logging: {exc}")
            self._log_enabled = False

    def finish(self, *, log_prefix: str = "SwanLab") -> None:
        if not self._need_finish or _sw is None:
            self._need_finish = False
            return
        try:
            _sw.finish()
        except KeyboardInterrupt:
            print(
                f"[{log_prefix}] swanlab.finish() interrupted; cloud sync may be incomplete. "
                "Tip: --disable-swanlab for Ctrl+C-friendly local runs."
            )
        except Exception as exc:  # pragma: no cover
            print(f"[{log_prefix}] swanlab finish failed (ignored): {exc}")
        finally:
            self._need_finish = False
            self._log_enabled = False


def _swanlab_safe_init(init_kw: Dict[str, Any], log_prefix: str) -> bool:
    if _sw is None:
        return False
    try:
        _sw.init(**init_kw)
        return True
    except TypeError:
        if "workspace" not in init_kw:
            print(f"[{log_prefix}] swanlab init failed: incompatible init arguments")
            return False
        kw2 = {k: v for k, v in init_kw.items() if k != "workspace"}
        try:
            _sw.init(**kw2)
            return True
        except Exception as exc:
            print(f"[{log_prefix}] swanlab init failed, continue without tracking: {exc}")
            return False
    except Exception as exc:
        print(f"[{log_prefix}] swanlab init failed, continue without tracking: {exc}")
        return False


def swanlab_try_init(
    *,
    disabled: bool,
    project: str,
    experiment_name: Optional[str],
    experiment_name_prefix: str,
    config: Optional[Dict[str, Any]],
    workspace: Optional[str] = None,
    log_prefix: str = "SwanLab",
) -> SwanLabTrainContext:
    if disabled:
        return SwanLabTrainContext(_need_finish=False)
    if _sw is None:
        print(f"[{log_prefix}] swanlab not installed, skip experiment tracking.")
        return SwanLabTrainContext(_need_finish=False)
    name = experiment_name or f"{experiment_name_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    init_kw: Dict[str, Any] = {"project": project, "experiment_name": name, "config": config or {}}
    if workspace:
        init_kw["workspace"] = workspace
    ok = _swanlab_safe_init(init_kw, log_prefix)
    return SwanLabTrainContext(_need_finish=ok)


def resolve_swanlab_project(args: Any, cfg: Any) -> str:
    """优先级：命令行 ``--swanlab-project`` > 环境变量 ``SWANLAB_PROJECT`` > ``cfg.swanlab_project``。"""
    cli = getattr(args, "swanlab_project", None)
    if cli:
        return str(cli)
    env = os.environ.get("SWANLAB_PROJECT")
    if env:
        return env
    return str(getattr(cfg, "swanlab_project", "DSCA-VL"))


def add_swanlab_train_args(parser: "argparse.ArgumentParser") -> "argparse.ArgumentParser":
    import argparse as _argparse

    if not isinstance(parser, _argparse.ArgumentParser):
        raise TypeError("parser must be argparse.ArgumentParser")
    parser.add_argument(
        "--disable-swanlab",
        action="store_true",
        help="Disable SwanLab logging even if swanlab is installed.",
    )
    parser.add_argument(
        "--swanlab-project",
        type=str,
        default=None,
        help="SwanLab project name (overrides env SWANLAB_PROJECT and cfg.swanlab_project).",
    )
    parser.add_argument(
        "--swanlab-experiment-name",
        type=str,
        default=None,
        help="SwanLab experiment/run display name; default: <prefix>-YYYYMMDD-HHMMSS.",
    )
    parser.add_argument(
        "--swanlab-workspace",
        type=str,
        default=None,
        help="Optional SwanLab workspace id (team); ignored if the installed API does not support it.",
    )
    return parser


def training_diagnostics_to_log(diag: "TrainingDiagnostics", namespace: str) -> Dict[str, Any]:
    """将 ``TrainingDiagnostics`` 转为 SwanLab 可记录的标量（仅数值，避免 str 导致图表类型错误）。"""
    from .train_utils import TrainingDiagnostics as _TD

    if diag is None or not isinstance(diag, _TD):
        return {}
    ns = namespace.rstrip("/")
    out: Dict[str, Any] = {
        f"{ns}/diag_steps": float(diag.step_count),
        f"{ns}/diag_skip_nonfinite_samples": float(diag.skipped_nonfinite_samples),
        f"{ns}/diag_skip_nonfinite_batches": float(diag.skipped_nonfinite_batches),
        f"{ns}/diag_skip_nonfinite_grads": float(diag.skipped_nonfinite_grads),
        f"{ns}/diag_skip_empty_batches": float(diag.skipped_empty_batches),
        f"{ns}/diag_loss_nan": float(diag.loss_nan_count),
        f"{ns}/diag_grad_nan": float(diag.grad_nan_count),
        f"{ns}/diag_peak_memory_mb": float(diag.peak_memory_mb),
    }
    if diag.feature_shapes:
        out[f"{ns}/diag_feature_shape_samples"] = float(len(diag.feature_shapes))
        last = diag.feature_shapes[-1]
        out[f"{ns}/diag_last_shape_rank"] = float(len(last))
        for i, dim in enumerate(last[:6]):
            out[f"{ns}/diag_last_shape_d{i}"] = float(dim)
    return out
