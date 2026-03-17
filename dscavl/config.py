from dataclasses import dataclass


@dataclass
class DSCAVLConfig:
    data_root: str = "data"
    feature_root: str = "features"
    siglip_model_name: str = "google/siglip-base-patch16-224"
    text_model_name_or_path: str = "Qwen2-VL-2B-Instruct"
    frame_sample_fps: float = 1.0
    siglip_batch_size: int = 16
    query_max_length: int = 128
    train_batch_size: int = 4
    num_workers: int = 0

    dim: int = 768
    bg_prototypes: int = 8
    temp_window: int = 8
    temp_stride: int = 4
    sigma_temp: float = 3.0

    # DSAM
    alpha_init: float = 1.0
    tau: float = 0.07
    lambda_orth: float = 0.1
    lambda_compact: float = 1.0
    lambda_align: float = 1.0

    # CGRM
    beta1: float = 0.5
    beta2: float = 0.5

    # CMOS
    gamma: float = 1.0

    # RL / GRPO
    group_size: int = 4
    omega_cons: float = 0.5
    omega_sparse: float = 0.3
    beta_kl: float = 0.02

    # Weak supervision from subtitles
    weak_sup_min_hits: int = 1
    weak_sup_expand_sec: float = 1.5

    # Train
    lr: float = 2e-5
    stage2_lr: float = 1e-5
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    stage1_epochs: int = 10
    stage2_epochs: int = 5

    # Infer
    keep_ratio_low: float = 0.08
    keep_ratio_high: float = 0.12
