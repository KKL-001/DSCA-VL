from .config import DSCAVLConfig
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
from .weak_supervision import SubtitleSegment, build_gt_mask_from_subtitles, extract_keywords, match_subtitle_segments, parse_srt_segments

__all__ = [
    "DSCAVLConfig",
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
]
