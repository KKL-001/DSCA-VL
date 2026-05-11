"""DSCA-VL 全局默认配置（数据路径、SigLIP、DSAM/CGRM、RL 与训练超参）。

所有训练入口与 ``DSCAVL`` 模块共享 ``DSCAVLConfig``；改动时需与特征缓存
（``siglip_cache_version``、``patch_tokens_per_frame``）及数据管线一致。

**环境与依赖**：在仓库根目录执行 ``pip install -r requirements.txt`` 安装核心依赖
（含 SigLIP/transformers、视频 I/O、以及 Qwen2-VL 用的 ``qwen-vl-utils``）。
训练脚本（``train_stage*.py``）在已安装 ``swanlab`` 时默认上报实验；也可用 ``--disable-swanlab`` 关闭。
根目录 ``train.py``（ModelScope、PEFT）再按需安装 ``modelscope``、``peft``，见 ``requirements.txt``。
"""
from dataclasses import dataclass


@dataclass
class DSCAVLConfig:
    """可序列化的训练/推理超参集合，字段按子系统分组便于对照论文与脚本。"""
    data_root: str = "data"
    feature_root: str = "features"
    siglip_model_name: str = "google/siglip-base-patch16-224"
    siglip_cache_version: str = "siglip_patch_v2"
    text_model_name_or_path: str = "/root/autodl-tmp/Qwen/Qwen2-VL-2B-Instruct"
    frame_sample_fps: float = 1.0
    siglip_batch_size: int = 16
    patch_tokens_per_frame: int = 64
    query_max_length: int = 128
    train_batch_size: int = 16
    num_workers: int = 2

    dim: int = 768
    bg_prototypes: int = 128
    temp_window: int = 8
    temp_stride: int = 4
    sigma_temp: float = 3.0

    # DSAM（可验收迭代 R2：略提高 orth/recon 动态，利 BASELINE #4–#6）
    # R12：Phase2 时间维 sim 去均值后略降 τ，与缩放后的 sim 方差/标度匹配
    alpha_init: float = 1.0
    tau: float = 0.065
    lambda_orth: float = 0.12
    lambda_compact: float = 1.0
    lambda_align: float = 1.0
    # R12：锐化+稀疏图后略降 recon 相对权重，避免与 align/weak 争梯度
    lambda_recon: float = 0.52

    # CGRM（R2：§8 F_bg 开；§6/§2.x 略抬 bridge、β 微调；#7–#9 门控用 mix 温度+熵）
    beta1: float = 0.48
    beta2: float = 0.52
    # When True, CGRM frame features use fg_bg_fuse(cat(F_fg, F_bg)); when False, F_fg only (legacy).
    cgrm_use_f_bg: bool = True
    # Causal graph calibration
    causal_decay_tau: float = 6.0
    bridge_weight: float = 0.38
    # Softmax temperature on mix(f_q) logits; lower values sharpen semantic vs causal mixture.
    cgrm_mix_temperature: float = 1.0
    # Stage1: maximize entropy of mix weights to encourage using both semantic and causal adjacency (0 = off).
    # R12：图去饱和后略抬 mix 熵项，防 λ_sem/λ_cau 单支塌缩
    lambda_cgrm_mix_entropy: float = 0.022

    # Architecture variants for module ablations from the feature-decoupling reference.
    # ``baseline`` preserves the current model; named variants only toggle module structure,
    # so R13-R17 can keep the same optimizer/loss/validation settings.
    arch_variant: str = "baseline"
    use_isoclip_debias: bool = False
    use_riemann_slerp: bool = False
    use_soft_mask: bool = False
    use_sbp: bool = False
    isoclip_debias_strength: float = 0.35
    riemann_slerp_strength: float = 0.35
    soft_mask_strength: float = 0.35
    sbp_negative_threshold: float = 0.18
    sbp_negative_weight: float = 0.25

    # R18+ full-coverage module switches. Defaults keep old checkpoints unchanged.
    use_ums_mae: bool = False
    use_pretrained_feature_adapter: bool = False
    use_everest_pruning: bool = False
    use_adaptive_budget: bool = False
    use_hornet_policy: bool = False
    use_efs_selection: bool = False
    use_dynamic_logit_threshold: bool = False
    use_hroute: bool = False
    use_st_nms: bool = False
    use_tta_mab: bool = False
    ums_bottleneck_dim: int = 256
    lambda_ums_orth: float = 0.05
    pretrained_feature_root: str = ""
    pretrained_feature_blend: float = 0.35
    everest_keep_ratio: float = 0.72
    adaptive_budget_low: int = 4
    adaptive_budget_high: int = 48
    hornet_policy_weight: float = 0.35
    efs_diversity_weight: float = 0.30
    dynamic_logit_temperature_min: float = 0.65
    dynamic_logit_temperature_max: float = 2.50
    hroute_complexity_threshold: float = 0.35
    st_nms_min_gap: int = 2
    tta_mab_lr: float = 0.10

    # Stage1: align FramePolicyHead logits to frozen CGRM S_graph（infer 不训练 policy 时 MLP 近似随机，对比 uniform 会系统性吃亏）
    lambda_policy_salign: float = 0.12
    policy_salign_temperature: float = 0.22

    # CMOS
    gamma: float = 1.0
    intra_top_ratio: float = 0.5

    # RL / GRPO
    group_size: int = 4
    omega_cons: float = 0.3
    omega_sparse: float = 0.3
    omega_coverage: float = 0.2
    beta_kl: float = 0.02

    # Weak supervision from subtitles
    weak_sup_min_hits: int = 1
    weak_sup_expand_sec: float = 1.5

    # CGRM weak loss: subtitle gt_mask pooled to events vs S_event (Stage1 auxiliary)
    lambda_cgrm_weak: float = 0.12
    cgrm_weak_event_pool: str = "max"  # "mean" | "max"
    cgrm_weak_loss: str = "bce_logits"  # "bce_logits" | "mse"
    cgrm_weak_tv_weight: float = 0.0
    # True：弱监督只对 CGRM 反传（对 F_fg/S_sem/f_q detach 后再算一遍 cgrm），避免与 stage1 主损失在共享子图上叠加出 NaN 梯度。
    cgrm_weak_detach_inputs: bool = True

    # -------------------------------------------------------------------------
    # Phase 2（预留，当前训练逻辑未使用）：MECD-Benchmark 等事件级 relation 强监督
    # -------------------------------------------------------------------------
    # 与 MECD+ / VGCM 思路一致：显式 relation 标签 + 轻量 RelationHead。字段名与默认值
    # 先固定，便于日后实现时直接读写 ``cfg`` 并写入 checkpoint（``cfg.__dict__``），避免
    # 再改 dataclass 字段名导致新旧 ckpt 语义漂移；旧 checkpoint 不含下列键时，由本类默认
    # 值补齐即可。
    # MECD（或兼容格式）训练标注 JSON；空字符串 = 关闭强监督分支。
    mecd_train_json: str = ""
    # 可选验证集 JSON；空 = 未配置。
    mecd_val_json: str = ""
    # 事件级 relation 损失权重；0 = 不加入总损失（与 Phase 2 未实现时一致）。
    lambda_mecd_relation: float = 0.0
    # Relation 类别数；0 = 未启用或由 Phase 2 DataLoader 从标签解析后覆盖。
    mecd_num_relation_classes: int = 0
    # 标注内视频为相对路径时的根目录；空 = 后续可用 ``data_root`` 等默认。
    mecd_video_root: str = ""

    # SwanLab（训练脚本 ``resolve_swanlab_project``：CLI > SWANLAB_PROJECT > 本字段）
    swanlab_project: str = "DSCA-VL"

    # Train
    # R12：显式低于 5e-3 峰值，配合 Phase2 特征/图锐化后首训稳定
    lr: float = 4.3e-3
    # Stage1 LR：``lr`` 为预热结束后的峰值；余弦退火至 ``stage1_lr_min``（见 ``Stage1LRState``）
    stage1_cosine_lr_schedule: bool = True
    stage1_lr_min: float = 5e-5
    stage1_warmup_ratio: float = 0.2
    stage1_lr_loss_reweight: float = 0.15
    stage1_lr_loss_ema_beta: float = 0.98
    stage1_lr_loss_blend_floor: float = 0.12
    stage2_lr: float = 1e-5
    stage2_dsam_reg_scale: float = 0.1
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    stage1_epochs: int = 30
    stage2_epochs: int = 5

    # Infer（R2：压紧 k≈8/T 量纲、提高事件覆盖，抑单峰；与 compare 同 k 时仍经 Qwen top-k）
    keep_ratio_low: float = 0.06
    keep_ratio_high: float = 0.095
    # §2.x：略提高事件覆盖，减轻单条视频单峰选帧
    min_event_ratio: float = 0.42
    max_per_event: int = 1
