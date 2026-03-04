from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def resize_latent_to_target(latent: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    将 latent 时间长度统一到 target_len。

    Args:
        latent: (B, C, T)
        target_len: 目标时间长度
    Returns:
        resized: (B, C, target_len)
    """
    if latent.dim() != 3:
        raise ValueError(f"latent must be 3D (B,C,T), got {tuple(latent.shape)}")
    if latent.size(-1) == target_len:
        return latent
    return F.interpolate(latent, size=target_len, mode="linear", align_corners=False)


def resize_latent_per_sample(latent: torch.Tensor, target_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    按样本目标长度重采样，并 pad 到 batch 内最大长度。

    Args:
        latent: (B, C, T)
        target_lens: (B,) long
    Returns:
        latent_out: (B, C, T_max)
        latent_mask: (B, T_max) bool
    """
    if latent.dim() != 3:
        raise ValueError(f"latent must be 3D (B,C,T), got {tuple(latent.shape)}")
    if target_lens.dim() != 1 or target_lens.size(0) != latent.size(0):
        raise ValueError("target_lens must be shape (B,)")

    bsz, channels, _ = latent.shape
    t_max = int(target_lens.max().item())

    out = torch.zeros((bsz, channels, t_max), device=latent.device, dtype=latent.dtype)
    mask = torch.zeros((bsz, t_max), device=latent.device, dtype=torch.bool)

    for i in range(bsz):
        t_i = int(target_lens[i].item())
        z_i = resize_latent_to_target(latent[i : i + 1], t_i).squeeze(0)
        out[i, :, :t_i] = z_i
        mask[i, :t_i] = True

    return out, mask


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, latent_mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (B, C, T)
    latent_mask: (B, T) bool
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")
    if latent_mask.dim() != 2 or latent_mask.size(0) != pred.size(0) or latent_mask.size(1) != pred.size(2):
        raise ValueError(
            f"latent_mask shape mismatch: expected (B,T)=({pred.size(0)},{pred.size(2)}), got {tuple(latent_mask.shape)}"
        )

    m = latent_mask.to(dtype=pred.dtype).unsqueeze(1)  # (B,1,T)
    diff2 = (pred - target).pow(2) * m
    denom = m.sum().clamp_min(1.0)
    return diff2.sum() / denom
