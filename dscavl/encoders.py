from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class FrozenVisualEncoder(nn.Module):
    """Frozen visual encoder adapter.

    Real reproduction should swap this with SigLIP frame encoder or precomputed features.
    """

    def __init__(self, input_channels: int = 3, dim: int = 768):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, dim)

        # Freeze by default to match paper setting.
        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self,
        video_frames: Optional[torch.Tensor] = None,
        precomputed_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if precomputed_features is not None:
            return precomputed_features
        if video_frames is None:
            raise ValueError("Either video_frames or precomputed_features must be provided.")

        b, t, c, h, w = video_frames.shape
        x = video_frames.view(b * t, c, h, w)
        x = self.backbone(x).flatten(1)
        x = self.proj(x)
        return x.view(b, t, -1)


class QueryTextEncoder(nn.Module):
    """Minimal query encoder stub.

    Replace this module with Qwen2-VL text embedding extraction in production.
    """

    def __init__(self, vocab_size: int = 32000, dim: int = 768):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # token_ids: [B, L]
        x = self.embed(token_ids)
        if attention_mask is not None:
            weights = attention_mask.unsqueeze(-1).to(x.dtype)
            x = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            x = x.mean(dim=1)
        return self.norm(x)
