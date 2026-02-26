"""
潜在空间条件生成模型（方案三方式 3.2：潜在空间扩散模型）
参考 Tango 项目的扩散模型实现

使用扩散模型在潜在空间中进行条件生成
"""

import math
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from audio.latent_encoder import LatentAudioEncoder, LatentAudioDecoder
from midi.custom.layers import Encoder
from .unet import ConditionalUNet
from .noise_schedule import NoiseSchedule


def _normalize_valid_mask(mask: Optional[torch.Tensor], seq_len: int) -> Optional[torch.Tensor]:
    """
    将不同形状/类型的 mask 归一化为 (B, S) 的 bool valid_mask（True=有效token）。
    支持：
    - (B, S)
    - (B, 1, 1, S) / (B, 1, S) 等
    - int/float/bool
    """
    if mask is None:
        return None

    while mask.dim() > 2:
        mask = mask.squeeze(1)

    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    if mask.size(-1) != seq_len:
        if mask.size(-1) > seq_len:
            mask = mask[..., :seq_len]
        else:
            pad = seq_len - mask.size(-1)
            mask = nn.functional.pad(mask, (0, pad), value=0)

    if mask.dtype == torch.bool:
        valid = mask
    else:
        valid = mask > 0
    return valid


class LatentSpaceConditionalModel(nn.Module):
    """
    潜在空间条件生成模型（方式 3.2：扩散模型）

    修复点（P0/P1）：
    - 将 midi_mask 贯穿传入 UNet cross-attention（忽略 padding token）。
    - CFG 的“无条件”不再使用全零，改为 learnable null embedding（更贴近训练分布，效果更稳）。
    - generate() 默认 shape 推断使用 midi_mask 的有效长度（若提供），避免被 padding 长度误导。
    """

    def __init__(
        self,
        # VAE 参数
        n_mel_channels: int = 80,
        latent_dim: int = 32,
        compression_factor: int = 4,
        # MIDI 编码器参数
        midi_vocab_size: int = 390,
        # 扩散模型参数
        embedding_dim: int = 256,
        num_layers: int = 6,
        max_seq: int = 2048,
        dropout: float = 0.2,
        # VAE 编码器/解码器参数
        n_filters: int = 128,
        n_residual_layers: int = 3,
        # UNet 参数
        base_channels: int = 64,
        channel_multipliers: list = None,
        num_res_blocks: int = 2,
        time_emb_dim: int = 512,
        # Cross-Attention 参数
        use_cross_attention: bool = True,
        cross_attention_heads: int = 8,
        cross_attention_layers: list = None,
        # 扩散模型参数
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.latent_dim = latent_dim
        self.compression_factor = compression_factor
        self.embedding_dim = embedding_dim
        self.midi_vocab_size = midi_vocab_size
        self.max_seq = max_seq
        self.num_timesteps = num_timesteps

        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 8]

        self.vae_encoder = LatentAudioEncoder(
            n_mel_channels=n_mel_channels,
            latent_dim=latent_dim,
            n_filters=n_filters,
            n_residual_layers=n_residual_layers,
            compression_factor=compression_factor,
        )

        self.vae_decoder = LatentAudioDecoder(
            latent_dim=latent_dim,
            n_mel_channels=n_mel_channels,
            n_filters=n_filters,
            n_residual_layers=n_residual_layers,
            compression_factor=compression_factor,
        )

        self.midi_encoder = Encoder(
            num_layers=num_layers,
            d_model=embedding_dim,
            input_vocab_size=midi_vocab_size,
            rate=dropout,
            max_len=max_seq,
        )

        # learnable null embedding（CFG 用）
        self.null_midi_emb = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.unet = ConditionalUNet(
            in_channels=latent_dim,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            time_emb_dim=time_emb_dim,
            condition_dim=embedding_dim,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            use_cross_attention=use_cross_attention,
            cross_attention_heads=cross_attention_heads,
            cross_attention_layers=cross_attention_layers,
        )

        self.noise_schedule = NoiseSchedule(num_timesteps=num_timesteps, schedule_type=schedule_type)

    def _make_uncond_features(self, cond: torch.Tensor) -> torch.Tensor:
        """
        cond: (B, S, D) -> uncond: (B, S, D)
        """
        return self.null_midi_emb.expand(cond.size(0), cond.size(1), cond.size(2)).to(cond.dtype).to(cond.device)

    def forward(
        self,
        audio_mel_spec: torch.Tensor,
        midi_tokens: torch.Tensor,
        midi_mask: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        condition_dropout_rate: float = 0.0,
    ):
        """
        训练阶段 forward（支持 CFG 训练）
        """
        latent = self.vae_encoder(audio_mel_spec)  # (B, latent_dim, T_compressed)

        midi_features, _ = self.midi_encoder(midi_tokens, mask=midi_mask)  # (B, S, D)

        if condition_dropout_rate > 0.0 and self.training:
            batch_size = midi_features.size(0)
            keep = torch.rand(batch_size, device=midi_features.device) > condition_dropout_rate
            if not keep.all():
                uncond = self._make_uncond_features(midi_features)
                midi_features = torch.where(keep[:, None, None], midi_features, uncond)

        if timesteps is None:
            timesteps = self.noise_schedule.sample_timesteps(latent.size(0), latent.device)

        noise = torch.randn_like(latent)
        noisy_latent = self.noise_schedule.add_noise(latent, noise, timesteps)

        # 关键：把 midi_mask 传进 UNet，让 cross-attn 忽略 padding
        valid_mask = _normalize_valid_mask(midi_mask, midi_features.size(1)) if midi_mask is not None else None
        predicted_noise = self.unet(noisy_latent, timesteps, midi_features, condition_mask=valid_mask)

        return predicted_noise, noise

    def forward_with_latent(
        self,
        audio_latent: torch.Tensor,
        midi_tokens: torch.Tensor,
        midi_mask: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        condition_dropout_rate: float = 0.0,
    ):
        """
        使用预计算 latent 的 forward（阶段二训练）
        """
        latent = audio_latent

        midi_features, _ = self.midi_encoder(midi_tokens, mask=midi_mask)

        if condition_dropout_rate > 0.0 and self.training:
            batch_size = midi_features.size(0)
            keep = torch.rand(batch_size, device=midi_features.device) > condition_dropout_rate
            if not keep.all():
                uncond = self._make_uncond_features(midi_features)
                midi_features = torch.where(keep[:, None, None], midi_features, uncond)

        if timesteps is None:
            timesteps = self.noise_schedule.sample_timesteps(latent.size(0), latent.device)

        noise = torch.randn_like(latent)
        noisy_latent = self.noise_schedule.add_noise(latent, noise, timesteps)

        valid_mask = _normalize_valid_mask(midi_mask, midi_features.size(1)) if midi_mask is not None else None
        predicted_noise = self.unet(noisy_latent, timesteps, midi_features, condition_mask=valid_mask)

        return predicted_noise, noise

    def encode_midi(self, midi_tokens: torch.Tensor, mask: torch.Tensor = None):
        midi_features, _ = self.midi_encoder(midi_tokens, mask=mask)
        return midi_features

    def generate(
        self,
        midi_tokens: torch.Tensor,
        shape: tuple = None,
        midi_mask: torch.Tensor = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
    ):
        """
        生成阶段：MIDI 条件扩散生成（支持 CFG）
        """
        self.eval()

        if midi_tokens.dim() == 1:
            midi_tokens = midi_tokens.unsqueeze(0)
        batch_size = midi_tokens.size(0)
        device = midi_tokens.device

        with torch.no_grad():
            midi_features = self.encode_midi(midi_tokens, mask=midi_mask)

        valid_mask = _normalize_valid_mask(midi_mask, midi_features.size(1)) if midi_mask is not None else None

        if shape is None:
            # 默认：使用“有效长度”推断，而不是 padding 后的长度
            if valid_mask is not None:
                # 取 batch 内最大有效长度，保证 batch 形状一致
                midi_len = int(valid_mask.sum(dim=1).max().item())
            else:
                midi_len = midi_tokens.size(1)

            T_compressed = max(1, (midi_len + self.compression_factor - 1) // self.compression_factor)
            shape = (self.latent_dim, T_compressed)

        latent_dim, T_compressed = shape
        latent = torch.randn(batch_size, latent_dim, T_compressed, device=device)

        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, device=device).long()

        use_cfg = guidance_scale > 1.0
        if use_cfg:
            uncond_features = self._make_uncond_features(midi_features)

        for t in timesteps:
            t_batch = t.expand(batch_size)
            with torch.no_grad():
                if use_cfg:
                    combined_features = torch.cat([midi_features, uncond_features], dim=0)
                    combined_latent = latent.repeat(2, 1, 1)
                    combined_t = t_batch.repeat(2)

                    # 注意：mask 也要拼接两份
                    if valid_mask is not None:
                        combined_mask = torch.cat([valid_mask, valid_mask], dim=0)
                    else:
                        combined_mask = None

                    combined_pred = self.unet(combined_latent, combined_t, combined_features, condition_mask=combined_mask)
                    pred_cond = combined_pred[:batch_size]
                    pred_uncond = combined_pred[batch_size:]
                    noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                else:
                    noise_pred = self.unet(latent, t_batch, midi_features, condition_mask=valid_mask)

            latent = self.noise_schedule.sample(noise_pred, latent, t_batch)

        with torch.no_grad():
            generated_mel = self.vae_decoder(latent)

        return latent, generated_mel
