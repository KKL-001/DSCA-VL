"""Architecture variant registry for DSCA-VL module ablations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import DSCAVLConfig


@dataclass(frozen=True)
class ArchVariant:
    """Static metadata and config overrides for one architecture variant."""

    name: str
    label: str
    description: str
    overrides: dict[str, Any]


_VARIANT_BOOL_FIELDS = (
    "use_isoclip_debias",
    "use_riemann_slerp",
    "use_soft_mask",
    "use_sbp",
    "use_ums_mae",
    "use_pretrained_feature_adapter",
    "use_everest_pruning",
    "use_adaptive_budget",
    "use_hornet_policy",
    "use_efs_selection",
    "use_dynamic_logit_threshold",
    "use_hroute",
    "use_st_nms",
    "use_tta_mab",
)

_R17_FUSION = {
    "use_isoclip_debias": True,
    "use_riemann_slerp": True,
    "use_soft_mask": True,
    "use_sbp": True,
}


_VARIANTS: dict[str, ArchVariant] = {
    "baseline": ArchVariant(
        name="baseline",
        label="Baseline",
        description="Current DSCA-VL structure without new decoupling modules.",
        overrides={
            "use_isoclip_debias": False,
            "use_riemann_slerp": False,
            "use_soft_mask": False,
            "use_sbp": False,
        },
    ),
    "isoclip": ArchVariant(
        name="isoclip",
        label="IsoCLIP模块更迭",
        description="Spectral-style anisotropy debiasing before DSAM decomposition.",
        overrides={
            "use_isoclip_debias": True,
            "use_riemann_slerp": False,
            "use_soft_mask": False,
            "use_sbp": False,
        },
    ),
    "riemann_slerp": ArchVariant(
        name="riemann_slerp",
        label="RiemannSLERP模块更迭",
        description="EMA background anchor with spherical interpolation foreground separation.",
        overrides={
            "use_isoclip_debias": False,
            "use_riemann_slerp": True,
            "use_soft_mask": False,
            "use_sbp": False,
        },
    ),
    "soft_mask": ArchVariant(
        name="soft_mask",
        label="SoftMask模块更迭",
        description="Hadamard soft channel masking instead of hard residual subtraction.",
        overrides={
            "use_isoclip_debias": False,
            "use_riemann_slerp": False,
            "use_soft_mask": True,
            "use_sbp": False,
        },
    ),
    "sbp": ArchVariant(
        name="sbp",
        label="SBP模块更迭",
        description="Signed graph negative-edge propagation in CGRM.",
        overrides={
            "use_isoclip_debias": False,
            "use_riemann_slerp": False,
            "use_soft_mask": False,
            "use_sbp": True,
        },
    ),
    "fusion_all": ArchVariant(
        name="fusion_all",
        label="融合模块更迭",
        description="IsoCLIP debiasing, Riemann SLERP, SoftMask, and SBP enabled together.",
        overrides=_R17_FUSION,
    ),
    "r18_ums_mae": ArchVariant(
        name="r18_ums_mae",
        label="UMS_MaE模块更迭+R18",
        description="R17 fusion with UMS-style LAC/VAC separation and modality-adaptive ensemble.",
        overrides={**_R17_FUSION, "use_ums_mae": True},
    ),
    "r19_pretrained_features": ArchVariant(
        name="r19_pretrained_features",
        label="轻量预训练特征模块更迭+R19",
        description="R17 fusion with an optional lightweight pretrained feature adapter.",
        overrides={**_R17_FUSION, "use_pretrained_feature_adapter": True},
    ),
    "r20_everest_pruning": ArchVariant(
        name="r20_everest_pruning",
        label="EVEREST冗余修剪模块更迭+R20",
        description="R17 fusion with EVEREST-style feature disparity pruning.",
        overrides={**_R17_FUSION, "use_everest_pruning": True},
    ),
    "r21_qts_budget": ArchVariant(
        name="r21_qts_budget",
        label="QTSplus预算模块更迭+R21",
        description="R17 fusion with query-aware adaptive frame budget.",
        overrides={**_R17_FUSION, "use_adaptive_budget": True},
    ),
    "r22_hornet_policy": ArchVariant(
        name="r22_hornet_policy",
        label="HORNet策略模块更迭+R22",
        description="R17 fusion with HORNet-inspired keep-probability policy blending.",
        overrides={**_R17_FUSION, "use_hornet_policy": True},
    ),
    "r23_efs": ArchVariant(
        name="r23_efs",
        label="EFS事件锚定模块更迭+R23",
        description="R17 fusion with event-anchored frame selection and MMR refinement.",
        overrides={**_R17_FUSION, "use_efs_selection": True},
    ),
    "r24_dynamic_logit": ArchVariant(
        name="r24_dynamic_logit",
        label="Logit温度动态阈值模块更迭+R24",
        description="R17 fusion with entropy-aware logit temperature and dynamic thresholds.",
        overrides={**_R17_FUSION, "use_dynamic_logit_threshold": True},
    ),
    "r25_hroute": ArchVariant(
        name="r25_hroute",
        label="HRoute路由模块更迭+R25",
        description="R17 fusion with query-complexity routing for graph compute allocation.",
        overrides={**_R17_FUSION, "use_hroute": True},
    ),
    "r26_stnms": ArchVariant(
        name="r26_stnms",
        label="STNMS去冗余模块更迭+R26",
        description="R17 fusion with temporal non-maximum suppression for selected frames.",
        overrides={**_R17_FUSION, "use_st_nms": True},
    ),
    "r27_tta_mab": ArchVariant(
        name="r27_tta_mab",
        label="TTAVid_MAB模块更迭+R27",
        description="R17 fusion with optional test-time bandit sampling updates.",
        overrides={**_R17_FUSION, "use_tta_mab": True},
    ),
    "r28_ums_efs": ArchVariant(
        name="r28_ums_efs",
        label="UMS_EFS组合模块更迭+R28",
        description="Combine modality separation with event-anchored selection.",
        overrides={**_R17_FUSION, "use_ums_mae": True, "use_efs_selection": True},
    ),
    "r29_ums_stnms": ArchVariant(
        name="r29_ums_stnms",
        label="UMS_STNMS组合模块更迭+R29",
        description="Combine modality separation with temporal redundancy suppression.",
        overrides={**_R17_FUSION, "use_ums_mae": True, "use_st_nms": True},
    ),
    "r30_feature_everest": ArchVariant(
        name="r30_feature_everest",
        label="Feature_EVEREST组合模块更迭+R30",
        description="Combine pretrained feature adapter with EVEREST-style pruning.",
        overrides={**_R17_FUSION, "use_pretrained_feature_adapter": True, "use_everest_pruning": True},
    ),
    "r31_budget_hroute": ArchVariant(
        name="r31_budget_hroute",
        label="Budget_HRoute组合模块更迭+R31",
        description="Combine adaptive budget with query-complexity graph routing.",
        overrides={**_R17_FUSION, "use_adaptive_budget": True, "use_hroute": True},
    ),
    "r32_hornet_stnms": ArchVariant(
        name="r32_hornet_stnms",
        label="HORNet_STNMS组合模块更迭+R32",
        description="Combine keep-probability policy with temporal NMS.",
        overrides={**_R17_FUSION, "use_hornet_policy": True, "use_st_nms": True},
    ),
    "r33_efs_mab": ArchVariant(
        name="r33_efs_mab",
        label="EFS_MAB组合模块更迭+R33",
        description="Combine event anchors with test-time bandit sampling updates.",
        overrides={**_R17_FUSION, "use_efs_selection": True, "use_tta_mab": True},
    ),
    "r34_cgrm_fullfix": ArchVariant(
        name="r34_cgrm_fullfix",
        label="CGRM全修复模块更迭+R34",
        description="Enable all CGRM-side fixes for sparse, diverse, query-aware selection.",
        overrides={
            **_R17_FUSION,
            "use_adaptive_budget": True,
            "use_hornet_policy": True,
            "use_efs_selection": True,
            "use_dynamic_logit_threshold": True,
            "use_hroute": True,
            "use_st_nms": True,
            "use_tta_mab": True,
        },
    ),
    "r35_dsam_fullfix": ArchVariant(
        name="r35_dsam_fullfix",
        label="DSAM全修复模块更迭+R35",
        description="Enable all DSAM/feature-side fixes.",
        overrides={
            **_R17_FUSION,
            "use_ums_mae": True,
            "use_pretrained_feature_adapter": True,
            "use_everest_pruning": True,
        },
    ),
    "r36_dsam_cgrm_fusion": ArchVariant(
        name="r36_dsam_cgrm_fusion",
        label="DSAM_CGRM融合模块更迭+R36",
        description="Fuse DSAM-side and CGRM-side fixes before final full coverage.",
        overrides={
            **_R17_FUSION,
            "use_ums_mae": True,
            "use_pretrained_feature_adapter": True,
            "use_everest_pruning": True,
            "use_adaptive_budget": True,
            "use_efs_selection": True,
            "use_dynamic_logit_threshold": True,
            "use_hroute": True,
            "use_st_nms": True,
        },
    ),
    "r37_fusion_final": ArchVariant(
        name="r37_fusion_final",
        label="全覆盖融合模块更迭+R37",
        description="R17 plus all R18+ modules for final coverage.",
        overrides={
            **_R17_FUSION,
            "use_ums_mae": True,
            "use_pretrained_feature_adapter": True,
            "use_everest_pruning": True,
            "use_adaptive_budget": True,
            "use_hornet_policy": True,
            "use_efs_selection": True,
            "use_dynamic_logit_threshold": True,
            "use_hroute": True,
            "use_st_nms": True,
            "use_tta_mab": True,
        },
    ),
}


def list_arch_variants() -> tuple[str, ...]:
    """Return supported architecture variant names in execution order."""

    return tuple(_VARIANTS.keys())


def get_arch_variant(name: str) -> ArchVariant:
    """Return one variant or raise a clear error for invalid CLI/checkpoint values."""

    key = (name or "baseline").strip()
    try:
        return _VARIANTS[key]
    except KeyError as exc:
        valid = ", ".join(list_arch_variants())
        raise ValueError(f"Unknown arch_variant {name!r}; expected one of: {valid}") from exc


def apply_arch_variant(cfg: DSCAVLConfig, name: str | None = None) -> DSCAVLConfig:
    """Apply a variant's module toggles in-place and return ``cfg`` for chaining."""

    variant = get_arch_variant(name or cfg.arch_variant)
    cfg.arch_variant = variant.name
    for field in _VARIANT_BOOL_FIELDS:
        if hasattr(cfg, field):
            setattr(cfg, field, False)
    for field, value in variant.overrides.items():
        setattr(cfg, field, value)
    return cfg
