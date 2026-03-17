from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


class DSAM(nn.Module):
    def __init__(
        self,
        dim: int,
        bg_prototypes: int = 8,
        tau: float = 0.07,
        alpha_init: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.bg_prototypes = bg_prototypes
        self.tau = tau
        self.eps = eps

        self.q_bg = nn.Parameter(torch.randn(bg_prototypes, dim) * 0.02)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.ln = nn.LayerNorm(dim)

    def _compute_background_prototypes(self, f_raw: torch.Tensor) -> torch.Tensor:
        # f_raw: [B, T, D], q_bg: [K, D]
        b, _, d = f_raw.shape
        q = self.q_bg.unsqueeze(0).expand(b, -1, -1)  # [B, K, D]
        attn = torch.matmul(q, f_raw.transpose(1, 2)) / (d ** 0.5)  # [B, K, T]
        attn = torch.softmax(attn, dim=-1)
        p_bg = torch.matmul(attn, f_raw)  # [B, K, D]
        return p_bg

    def _compactness_loss(self, f_raw: torch.Tensor, p_bg: torch.Tensor) -> torch.Tensor:
        # Normalize to unit sphere so distances are scale-invariant across feature dims.
        f_n = F.normalize(f_raw, dim=-1)          # [B, T, D]
        p_n = F.normalize(p_bg, dim=-1)           # [B, K, D]
        # cosine distance = 1 - cosine_similarity; range [0, 2]
        dot = torch.matmul(f_n, p_n.transpose(1, 2))  # [B, T, K]
        dist = 1.0 - dot                               # [B, T, K]
        min_dist = dist.min(dim=-1).values             # [B, T]
        return min_dist.mean()

    def _project_to_bg(self, f_raw: torch.Tensor, p_bg: torch.Tensor) -> torch.Tensor:
        # Formula-faithful projection used in paper.
        numer = (f_raw.unsqueeze(2) * p_bg.unsqueeze(1)).sum(dim=-1)  # [B, T, K]
        denom = p_bg.unsqueeze(1).pow(2).sum(dim=-1) + self.eps  # [B, 1, K]
        coef = numer / denom  # [B, T, K]
        f_bg = (coef.unsqueeze(-1) * p_bg.unsqueeze(1)).sum(dim=2)  # [B, T, D]
        return f_bg

    def _orthogonality_loss(self, f_fg: torch.Tensor, f_bg: torch.Tensor) -> torch.Tensor:
        # Use Gram-matrix off-diagonal penalty on normalized background prototypes.
        # This gives non-zero gradients even when prototypes are nearly orthogonal.
        # f_fg and f_bg are shaped [B, T, D]; we penalise cross-frame bg similarity.
        p_n = F.normalize(f_bg, dim=-1)  # [B, T, D]
        gram = torch.matmul(p_n, p_n.transpose(1, 2))  # [B, T, T]
        # Zero out diagonal (self-similarity = 1 always)
        eye = torch.eye(gram.size(1), device=gram.device, dtype=gram.dtype).unsqueeze(0)
        off_diag = gram * (1.0 - eye)
        return off_diag.pow(2).mean()

    def _semantic_scores(self, f_fg: torch.Tensor, f_q: torch.Tensor) -> torch.Tensor:
        f_q = F.normalize(f_q, dim=-1).unsqueeze(1)  # [B, 1, D]
        f_fg = F.normalize(f_fg, dim=-1)  # [B, T, D]
        sim = (f_fg * f_q).sum(dim=-1)  # [B, T]
        return torch.sigmoid(sim / self.tau)

    def _alignment_loss(self, s_sem: torch.Tensor, gt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # gt_mask is expected as bool [B, T], where True marks positive frames.
        if gt_mask is None:
            return torch.zeros((), device=s_sem.device, dtype=s_sem.dtype)
        positive = s_sem[gt_mask]
        if positive.numel() == 0:
            return torch.zeros((), device=s_sem.device, dtype=s_sem.dtype)
        return -(positive.clamp_min(self.eps).log()).mean()

    def forward(self, f_raw: torch.Tensor, f_q: torch.Tensor, gt_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        p_bg = self._compute_background_prototypes(f_raw)
        f_bg = self._project_to_bg(f_raw, p_bg)
        f_fg = self.ln(f_raw - self.alpha * f_bg)
        s_sem = self._semantic_scores(f_fg, f_q)

        loss_compact = self._compactness_loss(f_raw, p_bg)
        loss_orth = self._orthogonality_loss(f_fg, f_bg)
        loss_align = self._alignment_loss(s_sem, gt_mask)

        return {
            "F_fg": f_fg,
            "F_bg": f_bg,
            "P_bg": p_bg,
            "S_sem": s_sem,
            "loss_compact": loss_compact,
            "loss_orth": loss_orth,
            "loss_align": loss_align,
        }
