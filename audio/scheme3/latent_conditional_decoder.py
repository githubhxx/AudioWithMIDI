"""
潜在空间条件解码器
方案三方式 3.1：潜在空间 Transformer + Cross-Attention

在潜在空间中使用 Transformer Decoder with Cross-Attention 进行条件生成
"""

import math
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from midi.custom.layers import DecoderLayer
from midi.custom.layers import DynamicPositionEmbedding


def _normalize_mask_2d(mask: Optional[torch.Tensor], seq_len: int) -> Optional[torch.Tensor]:
    """
    归一化 mask 为 (B, S)（保持原语义，不强制 bool；交给下游层处理）。
    支持：(B,S)、(B,1,1,S)、(B,1,S) 等
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
            mask = nn.functional.pad(mask, (0, seq_len - mask.size(-1)), value=0)
    return mask


class LatentSpaceConditionalDecoder(nn.Module):
    """
    潜在空间条件解码器

    修复点：
    - head 数不再使用 embedding_dim//64 这种脆弱写法，改为显式 num_heads 并做整除校验。
    - max_seq 默认对齐到主模型（建议 2048），并在输入序列超长时自动扩展位置编码（如果可行）。
    - 条件维度不匹配时可通过 midi_condition_dim + 线性投影对齐。
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        latent_dim: int = 32,
        num_layers: int = 6,
        max_seq: int = 2048,
        dropout: float = 0.2,
        num_heads: int = 8,
        midi_condition_dim: Optional[int] = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.max_seq = max_seq

        if midi_condition_dim is None:
            midi_condition_dim = embedding_dim
        self.midi_condition_dim = midi_condition_dim

        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")

        # 将潜在特征投影到 embedding 空间
        self.latent_projection = nn.Linear(latent_dim, embedding_dim)

        # MIDI 条件投影（如果维度不一致）
        self.midi_projection = nn.Identity()
        if midi_condition_dim != embedding_dim:
            self.midi_projection = nn.Linear(midi_condition_dim, embedding_dim)

        # 位置编码
        self.pos_encoding = DynamicPositionEmbedding(embedding_dim, max_seq=max_seq)

        # 解码器层（使用 cross-attention）
        self.dec_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=embedding_dim,
                    rate=dropout,
                    h=num_heads,
                    additional=False,
                    max_seq=max_seq,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        # 输出层（映射回潜在空间）
        self.output_projection = nn.Linear(embedding_dim, latent_dim)

    
def _maybe_extend_pos(self, needed_len: int) -> None:
    """
    如果输入序列长度超过 max_seq，则扩展位置编码的覆盖范围。

    说明：DynamicPositionEmbedding 的实现细节未知；为避免破坏 DecoderLayer 的可学习参数，
    这里**只**重建位置编码模块，不重建 decoder layers。
    训练时建议直接把 max_seq 设得足够大（例如与主模型一致 2048/4096）。
    """
    if needed_len <= self.max_seq:
        return
    new_max = 1
    while new_max < needed_len:
        new_max *= 2
    self.max_seq = new_max
    self.pos_encoding = DynamicPositionEmbedding(self.embedding_dim, max_seq=self.max_seq)

def forward(
        self,
        latent_features: torch.Tensor,
        midi_condition: torch.Tensor,
        mask: torch.Tensor = None,
        lookup_mask: torch.Tensor = None,
    ):
        """
        Args:
            latent_features: (B, latent_dim, T) 或 (B, T, latent_dim)
            midi_condition: (B, midi_seq_len, midi_condition_dim)
            mask: MIDI 条件的 mask（可选，形状可 squeeze 到 (B, midi_seq_len)）
            lookup_mask: 潜在特征序列的 look-ahead mask（可选）

        Returns:
            output: (B, T, latent_dim)
        """
        # 处理输入维度
        if latent_features.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {latent_features.dim()}D")

        if latent_features.size(1) == self.latent_dim:
            latent_seq = latent_features.transpose(1, 2)  # (B, T, latent_dim)
        else:
            latent_seq = latent_features  # (B, T, latent_dim)

        T = latent_seq.size(1)
        self._maybe_extend_pos(T)

        # 条件投影
        midi_condition = self.midi_projection(midi_condition)

        # 归一化 mask 形状
        mask = _normalize_mask_2d(mask, midi_condition.size(1))

        x = self.latent_projection(latent_seq)  # (B, T, embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.dec_layers:
            x = layer(x, encode_out=midi_condition, mask=mask, lookup_mask=lookup_mask)

        output = self.output_projection(x)  # (B, T, latent_dim)
        return output
