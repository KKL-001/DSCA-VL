"""Composable R18+ variant modules for DSAM/CGRM experiments."""

from .modality import ModalitySeparationAdapter
from .postprocess import temporal_nms_1d
from .routing import adaptive_budget, bandit_update
from .selectors import event_anchor_select

__all__ = [
    "ModalitySeparationAdapter",
    "adaptive_budget",
    "bandit_update",
    "event_anchor_select",
    "temporal_nms_1d",
]
