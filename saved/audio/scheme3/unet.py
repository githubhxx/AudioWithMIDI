"""
UNet 模型用于潜在空间扩散模型
参考 Tango 项目的扩散模型实现

用于在潜在空间中进行噪声预测
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 调试开关：用于打印 encoder / decoder 每一层的形状与通道数
# 调试时可设为 True，正常训练时建议改回 False 以避免日志过多
DEBUG_UNET = False


class TimestepEmbedding(nn.Module):
    """
    时间步嵌入（用于扩散模型）

    参考Tango项目的实现，使用正弦位置编码 + MLP投影
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim

        # 正弦位置编码
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer("emb", emb)

        # MLP投影层（参考Tango项目，使用SiLU激活）
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: 时间步 (batch_size,) 或标量

        Returns:
            timestep_emb: 时间步嵌入 (batch_size, embedding_dim)
        """
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)

        # 正弦位置编码
        emb = timesteps.float()[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        # MLP投影（增强表达能力）
        emb = self.mlp(emb)

        return emb


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

    # squeeze 到最多 2 维
    while mask.dim() > 2:
        mask = mask.squeeze(1)

    if mask.dim() == 1:
        # (S,) -> (1,S)
        mask = mask.unsqueeze(0)

    if mask.size(-1) != seq_len:
        # 尝试自动对齐（极少见）：截断或padding
        if mask.size(-1) > seq_len:
            mask = mask[..., :seq_len]
        else:
            pad = seq_len - mask.size(-1)
            mask = F.pad(mask, (0, pad), value=0)

    if mask.dtype == torch.bool:
        valid = mask
    else:
        valid = mask > 0

    return valid


def _masked_mean(x: torch.Tensor, valid_mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    """
    x: (..., S, D) or (B,S,D)
    valid_mask: (B,S) True=valid
    """
    if valid_mask is None:
        return x.mean(dim=dim)

    # broadcast
    m = valid_mask.to(dtype=x.dtype)
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    denom = m.sum(dim=dim).clamp_min(1e-6)
    return (x * m).sum(dim=dim) / denom


class ResBlock(nn.Module):
    """
    残差块（用于 UNet）

    修复点：
    - 兼容 condition 为序列 (B, T_cond, D) 的情况（会自动池化为 (B, D)，支持 mask）。
    - 使用 FiLM(scale-shift) 注入 time / condition，比简单加法更稳定。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        condition_dim: int = 0,
        dropout: float = 0.1,
        num_groups: int = 8,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.condition_dim = condition_dim

        # 时间步投影 -> (scale, shift)
        self.time_proj = nn.Linear(time_emb_dim, out_channels * 2) if time_emb_dim > 0 else None

        # 条件投影 -> (scale, shift)
        self.condition_proj = nn.Linear(condition_dim, out_channels * 2) if condition_dim > 0 else None

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # 归一化
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 残差连接（如果通道数不同）
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        )

    def _pool_condition(
        self,
        condition: torch.Tensor,
        condition_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        将 condition 统一为 (B, D)
        支持：
        - (B, D)
        - (B, D, 1)
        - (B, T, D)
        """
        if condition.dim() == 2:
            return condition
        if condition.dim() == 3 and condition.size(-1) == 1:
            return condition.squeeze(-1)

        if condition.dim() == 3:
            # (B, T, D)
            valid = _normalize_valid_mask(condition_mask, condition.size(1))
            return _masked_mean(condition, valid, dim=1)

        raise ValueError(f"Unsupported condition shape: {tuple(condition.shape)}")

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (B, in_channels, T)
            time_emb: 时间步嵌入 (B, time_emb_dim)
            condition: 条件特征
                - (B, condition_dim)
                - (B, condition_dim, 1)
                - (B, T_cond, condition_dim)
            condition_mask: 条件序列 mask（可选，(B, T_cond) 或可 squeeze 的形状），True/1=有效

        Returns:
            output: 输出特征 (B, out_channels, T)
        """
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)

        # FiLM 注入：x = x * (1 + scale) + shift
        scale = shift = None

        if self.time_proj is not None and time_emb is not None:
            ss = self.time_proj(time_emb)  # (B, 2C)
            s, sh = ss.chunk(2, dim=-1)
            scale = s if scale is None else (scale + s)
            shift = sh if shift is None else (shift + sh)

        if self.condition_proj is not None and condition is not None:
            cond_vec = self._pool_condition(condition, condition_mask)  # (B, D)
            ss = self.condition_proj(cond_vec)  # (B, 2C)
            s, sh = ss.chunk(2, dim=-1)
            scale = s if scale is None else (scale + s)
            shift = sh if shift is None else (shift + sh)

        if scale is not None and shift is not None:
            x = x * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

        x = F.silu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)

        return x + residual


class CrossAttentionResBlock(nn.Module):
    """
    带 Cross-Attention 的残差块

    修复点：
    - 支持传入 condition_mask，并映射到 MultiheadAttention 的 key_padding_mask（padding 不参与注意力）。
    - 当某些层关闭 cross-attn（cross_attention_layers），依然能安全处理序列条件（会池化）。
    - time 注入改为 FiLM(scale-shift)，更稳。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        condition_dim: int = 0,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
        num_groups: int = 8,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.condition_dim = condition_dim
        self.use_cross_attention = use_cross_attention

        # 时间步投影 -> (scale, shift)
        self.time_proj = nn.Linear(time_emb_dim, out_channels * 2) if time_emb_dim > 0 else None

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # 归一化
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

        # CrossAttention 后用于卷积特征的归一化
        self.norm_cross_attn = nn.GroupNorm(num_groups, out_channels) if use_cross_attention else None

        # Dropout
        self.dropout = nn.Dropout(dropout)

        if use_cross_attention and condition_dim > 0:
            assert out_channels % num_heads == 0, f"out_channels ({out_channels}) must be divisible by num_heads ({num_heads})"
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=out_channels,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

            # MIDI 条件投影到 out_channels（序列 -> 序列）
            self.condition_proj = nn.Sequential(
                nn.Linear(condition_dim, out_channels * 2),
                nn.SiLU(),
                nn.Linear(out_channels * 2, out_channels),
            )

            self.norm_attn = nn.LayerNorm(out_channels)
            self.fallback_condition_proj = None
        else:
            self.cross_attention = None
            self.norm_attn = None
            # 关闭 cross-attn 时：对 condition 做池化，再 FiLM 注入
            self.fallback_condition_proj = nn.Linear(condition_dim, out_channels * 2) if condition_dim > 0 else None
            self.condition_proj = None

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        )

    def _pool_condition(
        self,
        condition: torch.Tensor,
        condition_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if condition.dim() == 2:
            return condition
        if condition.dim() == 3 and condition.size(-1) == 1:
            return condition.squeeze(-1)
        if condition.dim() == 3:
            valid = _normalize_valid_mask(condition_mask, condition.size(1))
            return _masked_mean(condition, valid, dim=1)
        raise ValueError(f"Unsupported condition shape: {tuple(condition.shape)}")

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, T)
            time_emb: (B, time_emb_dim)
            condition:
                - cross-attn 开启时推荐 (B, T_cond, condition_dim)
                - 也支持 (B, condition_dim)，会自动扩展为 (B, 1, D)
            condition_mask: (B, T_cond) True/1=有效token，padding 会被忽略

        Returns:
            (B, out_channels, T)
        """
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)

        # time FiLM
        if self.time_proj is not None and time_emb is not None:
            ss = self.time_proj(time_emb)  # (B, 2C)
            scale, shift = ss.chunk(2, dim=-1)
            x = x * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

        x = F.silu(x)
        x = self.dropout(x)

        # Cross-Attention
        if self.use_cross_attention and self.cross_attention is not None and condition is not None:
            if condition.dim() == 2:
                condition = condition.unsqueeze(1)  # (B,1,D)

            # (B,C,T) -> (B,T,C)
            x_seq = x.transpose(1, 2)  # (B, T, C)

            condition_proj = self.condition_proj(condition)  # (B, S, C)

            # key_padding_mask: True = ignore (padding)
            valid = _normalize_valid_mask(condition_mask, condition_proj.size(1))
            key_padding_mask = None if valid is None else (~valid)

            attended, _ = self.cross_attention(
                query=x_seq,
                key=condition_proj,
                value=condition_proj,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )

            x_seq = x_seq + attended
            if self.norm_attn is not None:
                x_seq = self.norm_attn(x_seq)

            x = x_seq.transpose(1, 2)  # (B,C,T)
            if self.norm_cross_attn is not None:
                x = self.norm_cross_attn(x)

        # fallback: no cross-attn, but still condition-aware
        elif self.fallback_condition_proj is not None and condition is not None:
            cond_vec = self._pool_condition(condition, condition_mask)  # (B, D)
            ss = self.fallback_condition_proj(cond_vec)  # (B, 2C)
            scale, shift = ss.chunk(2, dim=-1)
            x = x * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)

        return x + residual


class UNetEncoder(nn.Module):
    """UNet 编码器（下采样）"""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        channel_multipliers: list = (1, 2, 4, 8),
        time_emb_dim: int = 512,
        condition_dim: int = 0,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        use_cross_attention: bool = False,
        cross_attention_heads: int = 8,
        cross_attention_layers: Optional[list] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.use_cross_attention = use_cross_attention

        self.conv_in = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        channels = base_channels
        block_idx = 0

        for mult in channel_multipliers:
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                use_ca = use_cross_attention
                if cross_attention_layers is not None:
                    use_ca = block_idx in cross_attention_layers

                if use_ca:
                    self.down_blocks.append(
                        CrossAttentionResBlock(
                            channels,
                            out_channels,
                            time_emb_dim=time_emb_dim,
                            condition_dim=condition_dim,
                            num_heads=cross_attention_heads,
                            dropout=dropout,
                            use_cross_attention=True,
                        )
                    )
                else:
                    self.down_blocks.append(
                        ResBlock(
                            channels,
                            out_channels,
                            time_emb_dim=time_emb_dim,
                            condition_dim=condition_dim,
                            dropout=dropout,
                        )
                    )
                channels = out_channels
                block_idx += 1

            if mult != channel_multipliers[-1]:
                self.down_blocks.append(nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1))

        # mid blocks：通常对齐最关键
        if use_cross_attention:
            self.mid_block1 = CrossAttentionResBlock(
                channels,
                channels,
                time_emb_dim=time_emb_dim,
                condition_dim=condition_dim,
                num_heads=cross_attention_heads,
                dropout=dropout,
                use_cross_attention=True,
            )
            self.mid_block2 = CrossAttentionResBlock(
                channels,
                channels,
                time_emb_dim=time_emb_dim,
                condition_dim=condition_dim,
                num_heads=cross_attention_heads,
                dropout=dropout,
                use_cross_attention=True,
            )
        else:
            self.mid_block1 = ResBlock(
                channels,
                channels,
                time_emb_dim=time_emb_dim,
                condition_dim=condition_dim,
                dropout=dropout,
            )
            self.mid_block2 = ResBlock(
                channels,
                channels,
                time_emb_dim=time_emb_dim,
                condition_dim=condition_dim,
                dropout=dropout,
            )

        self.out_channels = channels

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Returns:
            x: mid 输出 (B, C, T)
            features: skip 用特征（按浅->深顺序）
        """
        x = self.conv_in(x)
        features = [x]

        if DEBUG_UNET:
            print(f"[ENC] conv_in: shape={tuple(x.shape)}")

        enc_block_idx = 0
        for block in self.down_blocks:
            if isinstance(block, (ResBlock, CrossAttentionResBlock)):
                x = block(x, time_emb=time_emb, condition=condition, condition_mask=condition_mask)
                features.append(x)

                if DEBUG_UNET:
                    print(
                        f"[ENC] feat_idx={len(features)-1}, "
                        f"block_idx={enc_block_idx}, "
                        f"type={block.__class__.__name__}, "
                        f"shape={tuple(x.shape)}"
                    )
                enc_block_idx += 1
            else:
                x = block(x)
                if DEBUG_UNET:
                    print(f"[ENC] downsample: type={block.__class__.__name__}, shape={tuple(x.shape)}")

        x = self.mid_block1(x, time_emb=time_emb, condition=condition, condition_mask=condition_mask)
        x = self.mid_block2(x, time_emb=time_emb, condition=condition, condition_mask=condition_mask)

        return x, features


class UNetDecoder(nn.Module):
    """UNet 解码器（上采样）"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        channel_multipliers: list = (1, 2, 4, 8),
        time_emb_dim: int = 512,
        condition_dim: int = 0,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        use_cross_attention: bool = False,
        cross_attention_heads: int = 8,
        cross_attention_layers: Optional[list] = None,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.use_cross_attention = use_cross_attention

        self.up_blocks = nn.ModuleList()
        channels = in_channels
        block_idx = 0

        # 关键修改：每个 resblock 都融合 skip（而不是每个分辨率只融合一次）
        for mult in reversed(channel_multipliers):
            stage_out = base_channels * mult

            for _ in range(num_res_blocks):
                in_ch = channels + stage_out  # concat skip
                use_ca = use_cross_attention
                if cross_attention_layers is not None:
                    use_ca = block_idx in cross_attention_layers

                if use_ca:
                    self.up_blocks.append(
                        CrossAttentionResBlock(
                            in_ch,
                            stage_out,
                            time_emb_dim=time_emb_dim,
                            condition_dim=condition_dim,
                            num_heads=cross_attention_heads,
                            dropout=dropout,
                            use_cross_attention=True,
                        )
                    )
                else:
                    self.up_blocks.append(
                        ResBlock(
                            in_ch,
                            stage_out,
                            time_emb_dim=time_emb_dim,
                            condition_dim=condition_dim,
                            dropout=dropout,
                        )
                    )

                channels = stage_out
                block_idx += 1

            if mult != channel_multipliers[0]:
                self.up_blocks.append(
                    nn.ConvTranspose1d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                )

        # 输出直接回到 latent_dim（避免 64->512->32 这种跳变）
        self.conv_out = nn.Conv1d(channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        skip_connections: list,
        time_emb: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
            skip_connections: encoder 输出的 skip（浅->深顺序），内部从末尾开始使用
        """
        skip_idx = len(skip_connections) - 1

        if DEBUG_UNET:
            print("==== UNetDecoder forward ====")
            for i, s in enumerate(skip_connections):
                print(f"[DEC] skip[{i}]: shape={tuple(s.shape)}")
            print(f"[DEC] initial x: shape={tuple(x.shape)}")

        for i, block in enumerate(self.up_blocks):
            if isinstance(block, (ResBlock, CrossAttentionResBlock)):
                if skip_idx < 0:
                    raise RuntimeError("Skip connections exhausted. Please check encoder/decoder resblock counts.")

                skip = skip_connections[skip_idx]
                skip_idx -= 1

                if skip.size(-1) != x.size(-1):
                    skip = F.interpolate(skip, size=x.size(-1), mode="nearest")

                x = torch.cat([x, skip], dim=1)
                x = block(x, time_emb=time_emb, condition=condition, condition_mask=condition_mask)

                if DEBUG_UNET:
                    print(f"[DEC] block i={i} type={block.__class__.__name__} x_shape={tuple(x.shape)}")
            else:
                x = block(x)
                if DEBUG_UNET:
                    print(f"[DEC] upsample i={i} type={block.__class__.__name__} x_shape={tuple(x.shape)}")

        x = self.conv_out(x)
        return x


class ConditionalUNet(nn.Module):
    """
    条件 UNet（用于潜在空间扩散模型）

    参考 Tango 项目的 UNet 架构，支持时间步嵌入和 MIDI 条件注入
    支持 Cross-Attention 机制，实现精确的 MIDI-音频对齐

    修复点：
    - skip 连接不再按 T 聚合“只保留一条”，而是保留每个 resblock 的输出（对应 decoder 每个 resblock 都融合 skip）。
    - decoder 输出直接回到 latent_dim，移除不必要的 out_proj。
    - 支持 condition_mask，在 cross-attn 中忽略 padding。
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        channel_multipliers: list = (1, 2, 4, 8),
        time_emb_dim: int = 512,
        condition_dim: int = 256,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
        cross_attention_heads: int = 8,
        cross_attention_layers: Optional[list] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_emb_dim = time_emb_dim
        self.condition_dim = condition_dim
        self.use_cross_attention = use_cross_attention

        self.time_embedding = TimestepEmbedding(time_emb_dim)

        # 非 cross-attn 模式下的条件投影（可选）
        self.condition_proj = None
        if not use_cross_attention and condition_dim > 0:
            self.condition_proj = nn.Sequential(
                nn.Linear(condition_dim, condition_dim),
                nn.SiLU(),
                nn.Linear(condition_dim, condition_dim),
            )

        # 解析 cross_attention_layers
        encoder_ca_layers = None
        decoder_ca_layers = None
        if cross_attention_layers is not None:
            if isinstance(cross_attention_layers, list) and len(cross_attention_layers) == 2:
                encoder_ca_layers, decoder_ca_layers = cross_attention_layers
            elif isinstance(cross_attention_layers, list):
                encoder_ca_layers = decoder_ca_layers = cross_attention_layers

        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=list(channel_multipliers),
            time_emb_dim=time_emb_dim,
            condition_dim=condition_dim,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            use_cross_attention=use_cross_attention,
            cross_attention_heads=cross_attention_heads,
            cross_attention_layers=encoder_ca_layers,
        )

        self.decoder = UNetDecoder(
            in_channels=self.encoder.out_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=list(channel_multipliers),
            time_emb_dim=time_emb_dim,
            condition_dim=condition_dim,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            use_cross_attention=use_cross_attention,
            cross_attention_heads=cross_attention_heads,
            cross_attention_layers=decoder_ca_layers,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, T)
            timestep: (B,) 或标量
            condition:
                - use_cross_attention=True: (B, T_cond, condition_dim)（推荐）
                - use_cross_attention=False: (B, condition_dim) 或 (B, T_cond, condition_dim)
            condition_mask: (B, T_cond) True/1=有效token（用于 cross-attn 忽略 padding）

        Returns:
            predicted_noise: (B, in_channels, T)
        """
        time_emb = self.time_embedding(timestep)  # (B, time_emb_dim)

        if condition is not None:
            if self.use_cross_attention:
                if condition.dim() == 2:
                    condition = condition.unsqueeze(1)  # (B,1,D)
            else:
                # 非 cross-attn：池化到 (B,D)，再投影
                if condition.dim() == 3:
                    valid = _normalize_valid_mask(condition_mask, condition.size(1))
                    condition = _masked_mean(condition, valid, dim=1)
                if self.condition_proj is not None:
                    condition = self.condition_proj(condition)

        x, features = self.encoder(x, time_emb=time_emb, condition=condition, condition_mask=condition_mask)

        # 关键修改：skip 取每个 resblock 输出（排除 conv_in），按浅->深；decoder 内部从末尾开始 pop
        skip_connections = features[1:]

        if DEBUG_UNET:
            print(f"[UNet] Skip connections: {len(skip_connections)} features")
            for i, skip in enumerate(skip_connections):
                print(f"  skip[{i}]: shape={tuple(skip.shape)}")

        x = self.decoder(
            x,
            skip_connections,
            time_emb=time_emb,
            condition=condition,
            condition_mask=condition_mask,
        )
        return x
