"""
三尺度条件模型：在 LatentSpaceConditionalModel 基础上加入结构条件与多尺度弱对齐融合。

核心改造：
1) 三尺度条件编码（high/mid/low）
2) low 尺度默认走弱对齐（token pooling）
3) 支持 condition dropout + scale dropout
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..latent_conditional_model import LatentSpaceConditionalModel


SCALE_TO_ID = {"high": 0, "mid": 1, "low": 2}
ID_TO_SCALE = {0: "high", 1: "mid", 2: "low"}


class StructuralConditionHead(nn.Module):
    def __init__(self, in_dim: int = 4, hidden_dim: int = 128, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, structural_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            structural_features: (B, 4)
        Returns:
            structural_token: (B, 1, out_dim)
        """
        x = self.net(structural_features)
        return x.unsqueeze(1)


class MultiScaleConditionFusion(nn.Module):
    """
    将基础 MIDI 编码特征按尺度做轻量投影，并输出给 UNet 的三尺度条件字典。
    - high/mid: 保留序列特征
    - low: 默认 weak alignment（token pooling -> 1 global token）
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.high_proj = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.SiLU(), nn.Dropout(dropout))
        self.mid_proj = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.SiLU(), nn.Dropout(dropout))
        self.low_proj = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.SiLU(), nn.Dropout(dropout))
        self.scale_embed = nn.Embedding(3, embedding_dim)

        # 分支类型 embedding：让 UNet 在拼接条件时能区分 high / mid / low 来源
        self.branch_embed = nn.Embedding(3, embedding_dim)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        if mask.dtype != torch.bool:
            mask = mask > 0
        w = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = w.sum(dim=1).clamp_min(1e-6)
        return (x * w).sum(dim=1) / denom

    def forward(
        self,
        midi_features: torch.Tensor,
        scale_id: torch.Tensor,
        midi_mask: Optional[torch.Tensor] = None,
        weak_low_alignment: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Optional[torch.Tensor]]]:
        """
        Args:
            midi_features: (B, S, D)
            scale_id: (B,), 0/1/2 => high/mid/low
        Returns:
            cond_dict: {"high": (B,S,D), "mid": (B,S,D), "low": (B,S_or_1,D)}
            mask_dict: 对应 mask 字典
        """
        bsz, seq_len, _ = midi_features.shape
        device = midi_features.device

        out = midi_features.clone()

        high_idx = (scale_id == SCALE_TO_ID["high"]).nonzero(as_tuple=False).flatten()
        mid_idx = (scale_id == SCALE_TO_ID["mid"]).nonzero(as_tuple=False).flatten()
        low_idx = (scale_id == SCALE_TO_ID["low"]).nonzero(as_tuple=False).flatten()

        if high_idx.numel() > 0:
            out[high_idx] = self.high_proj(out[high_idx])
        if mid_idx.numel() > 0:
            out[mid_idx] = self.mid_proj(out[mid_idx])
        if low_idx.numel() > 0:
            out[low_idx] = self.low_proj(out[low_idx])

        out = out + self.scale_embed(scale_id).unsqueeze(1)

        seq_mask = None
        if midi_mask is not None:
            seq_mask = midi_mask if midi_mask.dtype == torch.bool else (midi_mask > 0)

        # 为不同分支注入类型 embedding
        high_type = self.branch_embed(torch.full((bsz,), SCALE_TO_ID["high"], device=device, dtype=torch.long)).unsqueeze(1)
        mid_type = self.branch_embed(torch.full((bsz,), SCALE_TO_ID["mid"], device=device, dtype=torch.long)).unsqueeze(1)
        low_type = self.branch_embed(torch.full((bsz,), SCALE_TO_ID["low"], device=device, dtype=torch.long)).unsqueeze(1)

        high_cond = out + high_type
        mid_cond = out + mid_type

        if weak_low_alignment:
            low_vec = self._masked_mean(out, seq_mask)  # (B,D)
            low_cond = low_vec.unsqueeze(1) + low_type  # (B,1,D)
            low_mask = torch.ones((bsz, 1), device=device, dtype=torch.bool)
        else:
            low_cond = out + low_type
            low_mask = seq_mask

        cond_dict = {
            "high": high_cond,
            "mid": mid_cond,
            "low": low_cond,
        }
        mask_dict = {
            "high": seq_mask,
            "mid": seq_mask,
            "low": low_mask,
        }
        return cond_dict, mask_dict


class MultiScaleLatentSpaceConditionalModel(LatentSpaceConditionalModel):
    def __init__(self, *args, structural_feature_dim: int = 4, structural_hidden_dim: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        self.structural_head = StructuralConditionHead(
            in_dim=structural_feature_dim,
            hidden_dim=structural_hidden_dim,
            out_dim=self.embedding_dim,
            dropout=kwargs.get("dropout", 0.1),
        )
        self.multiscale_fusion = MultiScaleConditionFusion(
            embedding_dim=self.embedding_dim,
            dropout=kwargs.get("dropout", 0.1),
        )

    @staticmethod
    def _append_structural_condition(
        cond_dict: Dict[str, torch.Tensor],
        mask_dict: Dict[str, Optional[torch.Tensor]],
        structural_token: torch.Tensor,
    ):
        out_cond: Dict[str, torch.Tensor] = {}
        out_mask: Dict[str, Optional[torch.Tensor]] = {}

        for k, feat in cond_dict.items():
            mk = mask_dict.get(k)
            feat_k = torch.cat([feat, structural_token], dim=1)

            if mk is None:
                mask_k = None
            else:
                if mk.dtype != torch.bool:
                    mk = mk > 0
                ones = torch.ones((mk.size(0), 1), device=mk.device, dtype=torch.bool)
                mask_k = torch.cat([mk, ones], dim=1)

            out_cond[k] = feat_k
            out_mask[k] = mask_k

        return out_cond, out_mask

    @staticmethod
    def _random_token_shift(
        cond_features: torch.Tensor,
        cond_mask: Optional[torch.Tensor],
        max_shift: int = 4,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if max_shift <= 0:
            return cond_features, cond_mask

        bsz = cond_features.shape[0]
        out = cond_features.clone()
        out_mask = None if cond_mask is None else cond_mask.clone()

        for i in range(bsz):
            shift = int(torch.randint(low=-max_shift, high=max_shift + 1, size=(1,), device=cond_features.device).item())
            if shift == 0:
                continue
            out[i] = torch.roll(out[i], shifts=shift, dims=0)
            if out_mask is not None:
                out_mask[i] = torch.roll(out_mask[i], shifts=shift, dims=0)

        return out, out_mask

    @classmethod
    def _random_token_shift_dict(
        cls,
        cond_dict: Dict[str, torch.Tensor],
        mask_dict: Dict[str, Optional[torch.Tensor]],
        max_shift: int = 2,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Optional[torch.Tensor]]]:
        out_c: Dict[str, torch.Tensor] = {}
        out_m: Dict[str, Optional[torch.Tensor]] = {}

        for k, feat in cond_dict.items():
            if k == "low":
                out_c[k] = feat
                out_m[k] = mask_dict.get(k)
                continue
            f, m = cls._random_token_shift(feat, mask_dict.get(k), max_shift=max_shift)
            out_c[k] = f
            out_m[k] = m

        return out_c, out_m

    def forward_with_latent_and_structure(
        self,
        audio_latent: torch.Tensor,
        midi_tokens: torch.Tensor,
        structural_features: torch.Tensor,
        scale_id: torch.Tensor,
        midi_mask: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        condition_dropout_rate: float = 0.0,
        scale_dropout_rate: float = 0.0,
        weak_low_alignment: bool = True,
        token_shift_for_weak_alignment: int = 2,
    ):
        """
        阶段二训练入口：预计算 latent + midi token + 结构特征 + 尺度标签。
        """
        latent = audio_latent

        midi_features, _ = self.midi_encoder(midi_tokens, mask=midi_mask)
        cond_dict, mask_dict = self.multiscale_fusion(
            midi_features=midi_features,
            scale_id=scale_id,
            midi_mask=midi_mask,
            weak_low_alignment=weak_low_alignment,
        )

        if token_shift_for_weak_alignment > 0 and self.training:
            cond_dict, mask_dict = self._random_token_shift_dict(
                cond_dict,
                mask_dict,
                max_shift=token_shift_for_weak_alignment,
            )

        structural_token = self.structural_head(structural_features.to(midi_features.device, dtype=midi_features.dtype))
        cond_dict, mask_dict = self._append_structural_condition(cond_dict, mask_dict, structural_token)

        if scale_dropout_rate > 0.0 and self.training:
            keep_scale = torch.rand(latent.size(0), device=latent.device) > scale_dropout_rate
            if not keep_scale.all():
                for k in list(cond_dict.keys()):
                    uncond = self._make_uncond_features(cond_dict[k])
                    cond_dict[k] = torch.where(keep_scale[:, None, None], cond_dict[k], uncond)

        if condition_dropout_rate > 0.0 and self.training:
            keep = torch.rand(latent.size(0), device=latent.device) > condition_dropout_rate
            if not keep.all():
                for k in list(cond_dict.keys()):
                    uncond = self._make_uncond_features(cond_dict[k])
                    cond_dict[k] = torch.where(keep[:, None, None], cond_dict[k], uncond)

        if timesteps is None:
            timesteps = self.noise_schedule.sample_timesteps(latent.size(0), latent.device)

        noise = torch.randn_like(latent)
        noisy_latent = self.noise_schedule.add_noise(latent, noise, timesteps)

        predicted_noise = self.unet(noisy_latent, timesteps, cond_dict, condition_mask=mask_dict)
        return predicted_noise, noise
