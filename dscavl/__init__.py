"""dscavl：DSCA-VL 模型、数据集、编码器与训练辅助的公共导出入口。

外部脚本应优先 ``from dscavl import ...``，避免深层路径耦合。
"""
from .config import DSCAVLConfig
from .arch_registry import ArchVariant, apply_arch_variant, get_arch_variant, list_arch_variants
from .train_utils import (
    FeatureBatchInfo,
    Stage1LRState,
    TrainingDiagnostics,
    check_gradients_finite,
    ensure_frozen_visual_encoder,
    get_peak_memory_mb,
    prepare_feature_batch,
    prepare_single_feature,
    resolve_train_checkpoint_dir,
)
from .swanlab_train import (
    SwanLabTrainContext,
    add_swanlab_train_args,
    resolve_swanlab_project,
    swanlab_try_init,
    training_diagnostics_to_log,
)
from .data import (
    QuestionFeatureDataset,
    QuestionSample,
    VideoSample,
    build_video_index,
    flatten_questions,
    single_sample_collate,
    variable_feature_collate,
)
from .encoders import FrozenVisualEncoder, QueryTextEncoder
from .dsam import DSAM
from .cgrm import CGRM
from .policy import FramePolicyHead
from .model import DSCAVL
from .proxy_mcq import compute_mcq_exact_reward, extract_answer_label, normalize_option_text, score_mcq_options
from .cgrm_weak_loss import loss_cgrm_weak, pool_gt_mask_to_events
from .weak_supervision import SubtitleSegment, build_gt_mask_from_subtitles, extract_keywords, match_subtitle_segments, parse_srt_segments

__all__ = [
    "DSCAVLConfig",
    "ArchVariant",
    "apply_arch_variant",
    "get_arch_variant",
    "list_arch_variants",
    "SwanLabTrainContext",
    "add_swanlab_train_args",
    "resolve_swanlab_project",
    "swanlab_try_init",
    "training_diagnostics_to_log",
    "FeatureBatchInfo",
    "Stage1LRState",
    "TrainingDiagnostics",
    "check_gradients_finite",
    "ensure_frozen_visual_encoder",
    "get_peak_memory_mb",
    "prepare_feature_batch",
    "prepare_single_feature",
    "resolve_train_checkpoint_dir",
    "QuestionFeatureDataset",
    "QuestionSample",
    "VideoSample",
    "build_video_index",
    "flatten_questions",
    "single_sample_collate",
    "variable_feature_collate",
    "FrozenVisualEncoder",
    "QueryTextEncoder",
    "DSAM",
    "CGRM",
    "FramePolicyHead",
    "DSCAVL",
    "compute_mcq_exact_reward",
    "extract_answer_label",
    "normalize_option_text",
    "score_mcq_options",
    "SubtitleSegment",
    "build_gt_mask_from_subtitles",
    "extract_keywords",
    "match_subtitle_segments",
    "parse_srt_segments",
    "loss_cgrm_weak",
    "pool_gt_mask_to_events",
]
