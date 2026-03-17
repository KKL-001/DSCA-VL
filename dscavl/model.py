from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .cgrm import CGRM
from .config import DSCAVLConfig
from .dsam import DSAM
from .encoders import FrozenVisualEncoder, QueryTextEncoder
from .policy import FramePolicyHead


class DSCAVL(nn.Module):
    def __init__(
        self,
        cfg: DSCAVLConfig,
        vis_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        llm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = cfg

        self.vis_encoder = vis_encoder or FrozenVisualEncoder(dim=cfg.dim)
        self.text_encoder = text_encoder or QueryTextEncoder(dim=cfg.dim)
        self.dsam = DSAM(
            dim=cfg.dim,
            bg_prototypes=cfg.bg_prototypes,
            tau=cfg.tau,
            alpha_init=cfg.alpha_init,
        )
        self.cgrm = CGRM(
            dim=cfg.dim,
            window=cfg.temp_window,
            stride=cfg.temp_stride,
            sigma_temp=cfg.sigma_temp,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
        )
        self.policy = FramePolicyHead(dim=cfg.dim, gamma=cfg.gamma)

        # External MLLM adapter, for example Qwen2-VL pipeline.
        self.llm = llm

    def _select_actions_infer(self, probs: torch.Tensor) -> torch.Tensor:
        b, t = probs.shape
        k_low = max(1, int(t * self.cfg.keep_ratio_low))
        k_high = max(k_low, int(t * self.cfg.keep_ratio_high))

        actions = torch.zeros_like(probs)
        for i in range(b):
            k = min(k_high, t)
            idx = torch.topk(probs[i], k=k, dim=-1).indices
            actions[i, idx] = 1.0

            # Ensure lower bound.
            if actions[i].sum() < k_low:
                need = int(k_low - actions[i].sum().item())
                extra = torch.topk(probs[i] * (1 - actions[i]), k=need, dim=-1).indices
                actions[i, extra] = 1.0
        return actions

    def encode(
        self,
        video_frames: Optional[torch.Tensor],
        query_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        precomputed_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        f_raw = self.vis_encoder(video_frames=video_frames, precomputed_features=precomputed_features)
        f_q = self.text_encoder(query_tokens, attention_mask=attention_mask)
        return {
            "F_raw": f_raw,
            "f_q": f_q,
        }

    def forward(
        self,
        video_frames: Optional[torch.Tensor],
        query_tokens: torch.Tensor,
        mode: str = "stage1",
        gt_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        precomputed_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc = self.encode(
            video_frames,
            query_tokens,
            attention_mask=attention_mask,
            precomputed_features=precomputed_features,
        )
        f_raw, f_q = enc["F_raw"], enc["f_q"]

        ds = self.dsam(f_raw, f_q, gt_mask=gt_mask)
        cg = self.cgrm(ds["F_fg"], ds["S_sem"], f_q)

        if mode == "stage1":
            return {
                **enc,
                **ds,
                **cg,
            }

        sample = mode == "stage2"
        po = self.policy(cg["H_graph"], f_q, cg["S_graph"], sample=sample)

        if mode == "infer":
            po["actions"] = self._select_actions_infer(po["probs"])

        return {
            **enc,
            **ds,
            **cg,
            **po,
        }
