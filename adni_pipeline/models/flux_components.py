"""FLUX-specific components: ConfidenceHead and confidence-tempering loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceHead(nn.Module):
    """Per-modality head: logits and log-variance for confidence tempering."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.logits_head = nn.Linear(hidden_dim // 2, num_classes)
        self.logvar_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor):
        h = self.fe(x)
        logits = self.logits_head(h)
        log_var = self.logvar_head(h).expand_as(logits)
        return logits, log_var


def _safe_exp(x: torch.Tensor, cap: float = 40.0) -> torch.Tensor:
    return torch.exp(torch.clamp(x, -cap, cap))


def confidence_tempering_loss(
    mode: str,
    logits: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    curr_mod_mask: torch.Tensor,
    min_temp: float = 1.0,
    eps: float = 1e-8,
):
    """Temperature-scaled cross-entropy (confidence tempering).

    Args:
        mode: 'unimodal' or 'multimodal'.
        logits: (B, num_classes) for both modes.
        log_var: (B, num_classes) for unimodal; (M, B, num_classes) for multimodal.
        target: (B,) class indices.
        curr_mod_mask: (B,) for unimodal; (B, M) for multimodal.
    """
    if mode == "unimodal":
        uni_meta = {"uni_mean_sigma": None, "uni_mean_T": None}
        keep_idx = torch.where(curr_mod_mask == 1)[0]
        if keep_idx.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True), uni_meta

        logits = logits[keep_idx]
        log_var = log_var[keep_idx]
        target = target[keep_idx]
        sigma = _safe_exp(0.5 * log_var)
        temperature = sigma + 1e-6
        scaled_logits = logits / temperature

        uni_meta["uni_mean_sigma"] = float(sigma.mean().item())
        uni_meta["uni_mean_T"] = float(temperature.mean().item())
        return F.cross_entropy(scaled_logits, target), uni_meta

    elif mode == "multimodal":
        avail_mask = curr_mod_mask.transpose(0, 1).float()
        sigma2_m = _safe_exp(log_var).mean(dim=-1)
        precision_m = avail_mask / (sigma2_m + eps)
        precision_sum = precision_m.sum(dim=0)

        sigma2_eff = torch.where(
            precision_sum > 0,
            1.0 / (precision_sum + eps),
            torch.full_like(precision_sum, min_temp ** 2),
        )
        T = torch.sqrt(sigma2_eff) + 1e-6
        scaled_logits = logits / T.unsqueeze(dim=-1)

        mm_meta = {
            "mm_mean_sigma2_eff": float(sigma2_eff.mean().item()),
            "mm_mean_T": float(T.mean().item()),
        }
        return F.cross_entropy(scaled_logits, target), mm_meta

    else:
        raise ValueError(f"mode must be 'unimodal' or 'multimodal', got {mode}")
