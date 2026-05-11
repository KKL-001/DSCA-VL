"""视觉与查询文本编码适配层：默认占位实现；生产环境替换为 SigLIP + Qwen 文本嵌入。"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class FrozenVisualEncoder(nn.Module):
    """冻结的视觉编码占位：在线路径将帧卷积下采样到方形 patch 网格；离线路径透传缓存。"""

    def __init__(
        self,
        input_channels: int = 3,
        dim: int = 768,
        patch_tokens_per_frame: int = 64,
        return_patch_tokens: bool = False,
    ):
        """``patch_tokens_per_frame`` 须为完全平方数（如 64→8×8）。"""
        super().__init__()
        self.dim = dim
        self.patch_tokens_per_frame = max(1, int(patch_tokens_per_frame))
        self.return_patch_tokens = return_patch_tokens
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(64, dim)

        # Freeze by default to match paper setting.
        for p in self.parameters():
            p.requires_grad = False

    def _target_grid_size(self) -> tuple[int, int]:
        """由 P 推导池化空间边长 (H,W)。"""
        side = int(round(self.patch_tokens_per_frame ** 0.5))
        if side * side != self.patch_tokens_per_frame:
            raise ValueError(
                f"patch_tokens_per_frame must be a perfect square, got {self.patch_tokens_per_frame}"
            )
        return side, side

    def _format_feature_output(
        self,
        features: torch.Tensor,
        return_patch_tokens: bool,
    ) -> torch.Tensor:
        """按 ``return_patch_tokens`` 在 ``[B,T,P,D]`` 与 ``[B,T,D]``（帧均值）间切换。"""
        if features.ndim == 4:
            return features if return_patch_tokens else features.mean(dim=2)
        if features.ndim == 3:
            return features if not return_patch_tokens else features.unsqueeze(2)
        raise ValueError(f"Unsupported feature tensor shape: {tuple(features.shape)}")

    def _encode_patch_tokens(self, video_frames: torch.Tensor) -> torch.Tensor:
        """简易 CNN + adaptive pool 得到每帧 P 个 patch 向量。"""
        b, t, c, h, w = video_frames.shape
        x = video_frames.reshape(b * t, c, h, w)
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, self._target_grid_size())
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x.reshape(b, t, self.patch_tokens_per_frame, self.dim)

    def forward(
        self,
        video_frames: Optional[torch.Tensor] = None,
        precomputed_features: Optional[torch.Tensor] = None,
        return_patch_tokens: Optional[bool] = None,
    ) -> torch.Tensor:
        """二选一输入：预计算特征或原始视频帧张量 ``[B,T,C,H,W]``。"""
        if return_patch_tokens is None:
            return_patch_tokens = self.return_patch_tokens
        if precomputed_features is not None:
            if precomputed_features.ndim == 2:
                precomputed_features = precomputed_features.unsqueeze(0)
            return self._format_feature_output(precomputed_features, return_patch_tokens)
        if video_frames is None:
            raise ValueError("Either video_frames or precomputed_features must be provided.")

        patch_tokens = self._encode_patch_tokens(video_frames)
        return self._format_feature_output(patch_tokens, return_patch_tokens)


class QueryTextEncoder(nn.Module):
    """最小查询编码：Embedding 池化 + LayerNorm；可替换为真实 LM 的最后一层句向量。"""

    def __init__(self, vocab_size: int = 32000, dim: int = 768):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """mask 加权平均得到句向量 ``[B,D]``；无 mask 时简单均值。"""
        # token_ids: [B, L]
        x = self.embed(token_ids)
        if attention_mask is not None:
            weights = attention_mask.unsqueeze(-1).to(x.dtype)
            x = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            x = x.mean(dim=1)
        return self.norm(x)
