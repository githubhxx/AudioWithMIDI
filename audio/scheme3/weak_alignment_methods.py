"""
扩散模型中不需要强对齐 MIDI 和音频的方法实现

包含：
1. 对比学习对齐
2. 多尺度条件注入
3. 统计特征条件注入
4. 自监督对齐学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


# ==================== 方法 1: 对比学习对齐 ====================

class ContrastiveAlignment(nn.Module):
    """
    对比学习对齐模块
    通过对比学习让模型学习 MIDI 和音频的语义对应关系
    """
    
    def __init__(self, embedding_dim=256, temperature=0.07):
        """
        Args:
            embedding_dim: 特征维度
            temperature: 对比学习温度参数
        """
        super().__init__()
        self.temperature = temperature
        self.midi_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, midi_features, audio_features):
        """
        计算对比学习损失
        
        Args:
            midi_features: MIDI 特征 (B, T_midi, D) 或 (B, D)
            audio_features: 音频特征 (B, T_audio, D) 或 (B, D)
        
        Returns:
            loss: 对比学习损失
            similarity: 相似度矩阵 (B, B)
        """
        # 投影到统一空间
        midi_emb = self.midi_proj(midi_features)
        audio_emb = self.audio_proj(audio_features)
        
        # 如果有多时间步，进行池化
        if midi_emb.dim() == 3:
            midi_emb = midi_emb.mean(dim=1)  # (B, D)
        if audio_emb.dim() == 3:
            audio_emb = audio_emb.mean(dim=1)  # (B, D)
        
        # 归一化
        midi_emb = F.normalize(midi_emb, dim=-1)
        audio_emb = F.normalize(audio_emb, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(midi_emb, audio_emb.T) / self.temperature
        
        # InfoNCE 损失
        labels = torch.arange(similarity.size(0), device=similarity.device)
        loss = F.cross_entropy(similarity, labels)
        
        return loss, similarity


def contrastive_loss(midi_features, audio_features, temperature=0.07):
    """
    对比学习损失函数
    
    Args:
        midi_features: MIDI 特征 (B, T_midi, D) 或 (B, D)
        audio_features: 音频特征 (B, T_audio, D) 或 (B, D)
        temperature: 温度参数
    
    Returns:
        loss: 对比学习损失
    """
    # 归一化特征
    if midi_features.dim() == 3:
        midi_features = midi_features.mean(dim=1)  # (B, D)
    if audio_features.dim() == 3:
        audio_features = audio_features.mean(dim=1)  # (B, D)
    
    midi_norm = F.normalize(midi_features, dim=-1)
    audio_norm = F.normalize(audio_features, dim=-1)
    
    # 计算相似度矩阵
    similarity = torch.matmul(midi_norm, audio_norm.T) / temperature
    
    # InfoNCE 损失
    labels = torch.arange(similarity.size(0), device=similarity.device)
    loss = F.cross_entropy(similarity, labels)
    
    return loss


# ==================== 方法 2: 多尺度条件注入 ====================

class MultiScaleConditionalProjection(nn.Module):
    """
    多尺度条件投影
    在不同尺度上提取 MIDI 条件特征
    """
    
    def __init__(self, condition_dim=256, num_scales=3):
        """
        Args:
            condition_dim: 条件特征维度
            num_scales: 尺度数量
        """
        super().__init__()
        self.num_scales = num_scales
        
        # 全局条件投影
        self.global_proj = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim)
        )
        
        # 局部条件投影（每个尺度）
        self.local_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(condition_dim, condition_dim),
                nn.SiLU(),
                nn.Linear(condition_dim, condition_dim)
            )
            for _ in range(num_scales)
        ])
    
    def forward(self, midi_features):
        """
        提取多尺度条件特征
        
        Args:
            midi_features: MIDI 特征 (B, T_midi, D)
        
        Returns:
            global_cond: 全局条件 (B, D)
            local_conds: 局部条件列表 [(B, D), ...]
        """
        B, T_midi, D = midi_features.shape
        
        # 全局条件（整个序列）
        global_cond = midi_features.mean(dim=1)  # (B, D)
        global_cond = self.global_proj(global_cond)
        
        # 局部条件（不同片段）
        local_conds = []
        window_size = T_midi // self.num_scales
        
        for i in range(self.num_scales):
            start = i * window_size
            end = start + window_size if i < self.num_scales - 1 else T_midi
            local_midi = midi_features[:, start:end].mean(dim=1)  # (B, D)
            local_cond = self.local_projs[i](local_midi)
            local_conds.append(local_cond)
        
        return global_cond, local_conds


# ==================== 方法 3: 统计特征条件注入 ====================

class MIDIStatisticalFeatures(nn.Module):
    """
    MIDI 统计特征提取器
    提取 MIDI 的统计特征（音高分布、节奏特征等）
    """
    
    def __init__(self, vocab_size=390, feature_dim=128):
        """
        Args:
            vocab_size: MIDI 词汇表大小
            feature_dim: 输出特征维度
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(self._get_stats_dim(), 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
    
    def _get_stats_dim(self):
        """计算统计特征的维度"""
        # 音高分布: 128 (MIDI 音高范围)
        # 速度分布: 128 (MIDI 速度范围)
        # 时间特征: 10 (平均间隔、标准差等)
        # 和声特征: 12 (12 个调性)
        return 128 + 128 + 10 + 12
    
    def _extract_pitch_distribution(self, midi_tokens):
        """
        提取音高分布特征
        
        Args:
            midi_tokens: MIDI token 序列 (B, T)
        
        Returns:
            pitch_dist: 音高分布 (B, 128)
        """
        B, T = midi_tokens.shape
        pitch_dist = torch.zeros(B, 128, device=midi_tokens.device)
        
        # 简化：假设某些 token 范围对应音高
        # 实际实现需要根据 MIDI 编码方式提取音高信息
        for b in range(B):
            tokens = midi_tokens[b]
            # 这里需要根据实际的 MIDI 编码方式提取音高
            # 示例：假设 token 值在某个范围对应音高
            pitch_values = tokens % 128  # 简化处理
            pitch_dist[b] = torch.histc(
                pitch_values.float(), bins=128, min=0, max=127
            )
            pitch_dist[b] = pitch_dist[b] / (pitch_dist[b].sum() + 1e-8)  # 归一化
        
        return pitch_dist
    
    def _extract_rhythm_features(self, midi_tokens):
        """
        提取节奏特征
        
        Args:
            midi_tokens: MIDI token 序列 (B, T)
        
        Returns:
            rhythm_features: 节奏特征 (B, 10)
        """
        B, T = midi_tokens.shape
        rhythm_features = torch.zeros(B, 10, device=midi_tokens.device)
        
        # 简化：提取一些基本的节奏统计特征
        for b in range(B):
            tokens = midi_tokens[b]
            # 计算 token 间隔的统计特征
            # 这里需要根据实际的 MIDI 编码方式提取时间信息
            # 示例：假设某些 token 表示时间间隔
            rhythm_features[b, 0] = tokens.float().mean()
            rhythm_features[b, 1] = tokens.float().std()
            # ... 更多统计特征
        
        return rhythm_features
    
    def _extract_harmony_features(self, midi_tokens):
        """
        提取和声特征
        
        Args:
            midi_tokens: MIDI token 序列 (B, T)
        
        Returns:
            harmony_features: 和声特征 (B, 12)
        """
        B, T = midi_tokens.shape
        harmony_features = torch.zeros(B, 12, device=midi_tokens.device)
        
        # 简化：提取调性相关的特征
        # 实际实现需要分析 MIDI 的和声结构
        for b in range(B):
            tokens = midi_tokens[b]
            # 计算不同调性的分布
            # 这里需要根据实际的 MIDI 编码方式提取和声信息
            harmony_features[b] = torch.rand(12, device=tokens.device)  # 占位符
        
        return harmony_features
    
    def forward(self, midi_tokens):
        """
        提取 MIDI 统计特征
        
        Args:
            midi_tokens: MIDI token 序列 (B, T)
        
        Returns:
            stats: 统计特征 (B, feature_dim)
        """
        # 提取各种统计特征
        pitch_dist = self._extract_pitch_distribution(midi_tokens)  # (B, 128)
        rhythm_features = self._extract_rhythm_features(midi_tokens)  # (B, 10)
        harmony_features = self._extract_harmony_features(midi_tokens)  # (B, 12)
        
        # 组合特征
        stats = torch.cat([pitch_dist, rhythm_features, harmony_features], dim=-1)  # (B, 278)
        
        # 投影到目标维度
        stats = self.feature_extractor(stats)  # (B, feature_dim)
        
        return stats


# ==================== 方法 4: 自监督对齐学习 ====================

class SelfSupervisedAlignmentHead(nn.Module):
    """
    自监督对齐学习头
    预测 MIDI 和音频是否对应
    """
    
    def __init__(self, embedding_dim=256):
        """
        Args:
            embedding_dim: 特征维度
        """
        super().__init__()
        self.alignment_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, midi_features, audio_features):
        """
        预测 MIDI 和音频是否对应
        
        Args:
            midi_features: MIDI 特征 (B, T_midi, D) 或 (B, D)
            audio_features: 音频特征 (B, T_audio, D) 或 (B, D)
        
        Returns:
            alignment_score: 对齐分数 (B, 1)
        """
        # 池化到全局特征
        if midi_features.dim() == 3:
            midi_global = midi_features.mean(dim=1)  # (B, D)
        else:
            midi_global = midi_features
        
        if audio_features.dim() == 3:
            audio_global = audio_features.mean(dim=1)  # (B, D)
        else:
            audio_global = audio_features
        
        # 拼接
        combined = torch.cat([midi_global, audio_global], dim=-1)  # (B, 2D)
        
        # 预测对应关系
        alignment_score = self.alignment_head(combined)  # (B, 1)
        
        return alignment_score


# ==================== 方法 5: 弱对齐数据增强 ====================

def create_weak_alignment_pairs(midi_tokens, audio_mel, alignment_strategy='shift', max_shift=100):
    """
    创建弱对齐的 MIDI-音频对
    
    Args:
        midi_tokens: MIDI token 序列 (B, T_midi)
        audio_mel: 音频 Mel 频谱 (B, C, T_audio)
        alignment_strategy: 对齐策略 ('shift', 'crop', 'mix')
        max_shift: 最大偏移量
    
    Returns:
        midi_tokens_aug: 增强后的 MIDI tokens
        audio_mel_aug: 增强后的音频 Mel 频谱
    """
    B = midi_tokens.size(0)
    
    if alignment_strategy == 'shift':
        # 随机时间偏移
        shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=midi_tokens.device)
        midi_tokens_aug = midi_tokens.clone()
        
        for b in range(B):
            shift = shifts[b].item()
            if shift > 0:
                # 向右偏移：在前面填充
                pad = torch.zeros(shift, dtype=midi_tokens.dtype, device=midi_tokens.device)
                midi_tokens_aug[b] = torch.cat([pad, midi_tokens[b][:-shift]], dim=0)
            elif shift < 0:
                # 向左偏移：在后面填充
                pad = torch.zeros(-shift, dtype=midi_tokens.dtype, device=midi_tokens.device)
                midi_tokens_aug[b] = torch.cat([midi_tokens[b][-shift:], pad], dim=0)
        
        audio_mel_aug = audio_mel  # 音频保持原样
        
    elif alignment_strategy == 'crop':
        # 随机裁剪
        midi_tokens_aug = midi_tokens.clone()
        audio_mel_aug = audio_mel.clone()
        
        for b in range(B):
            # 随机裁剪 MIDI 或音频
            if random.random() < 0.5:
                # 裁剪 MIDI
                crop_ratio = random.uniform(0.7, 1.0)
                crop_len = int(midi_tokens.size(1) * crop_ratio)
                start = random.randint(0, midi_tokens.size(1) - crop_len)
                midi_tokens_aug[b] = midi_tokens[b, start:start+crop_len]
                # 填充到原始长度
                pad_len = midi_tokens.size(1) - crop_len
                pad = torch.zeros(pad_len, dtype=midi_tokens.dtype, device=midi_tokens.device)
                midi_tokens_aug[b] = torch.cat([midi_tokens_aug[b], pad], dim=0)
            else:
                # 裁剪音频
                crop_ratio = random.uniform(0.7, 1.0)
                crop_len = int(audio_mel.size(2) * crop_ratio)
                start = random.randint(0, audio_mel.size(2) - crop_len)
                audio_mel_aug[b] = audio_mel[b, :, start:start+crop_len]
                # 填充到原始长度
                pad_len = audio_mel.size(2) - crop_len
                pad = torch.zeros(audio_mel.size(1), pad_len, device=audio_mel.device)
                audio_mel_aug[b] = torch.cat([audio_mel_aug[b], pad], dim=1)
    
    elif alignment_strategy == 'mix':
        # 混合不同样本（但保持语义相关）
        # 这里简化处理：随机打乱批次内的配对
        indices = torch.randperm(B, device=midi_tokens.device)
        midi_tokens_aug = midi_tokens[indices]
        audio_mel_aug = audio_mel  # 音频保持原样
    
    else:
        midi_tokens_aug = midi_tokens
        audio_mel_aug = audio_mel
    
    return midi_tokens_aug, audio_mel_aug


# ==================== 综合使用示例 ====================

class WeakAlignmentTrainingHelper:
    """
    弱对齐训练辅助类
    整合各种弱对齐方法
    """
    
    def __init__(
        self,
        use_contrastive=True,
        use_multiscale=False,
        use_stats=False,
        use_self_supervised=False,
        use_weak_aug=False,
        contrastive_weight=0.1,
        temperature=0.07
    ):
        """
        Args:
            use_contrastive: 是否使用对比学习
            use_multiscale: 是否使用多尺度条件
            use_stats: 是否使用统计特征
            use_self_supervised: 是否使用自监督对齐
            use_weak_aug: 是否使用弱对齐数据增强
            contrastive_weight: 对比学习损失权重
            temperature: 对比学习温度
        """
        self.use_contrastive = use_contrastive
        self.use_multiscale = use_multiscale
        self.use_stats = use_stats
        self.use_self_supervised = use_self_supervised
        self.use_weak_aug = use_weak_aug
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
        if use_contrastive:
            self.contrastive_module = ContrastiveAlignment(temperature=temperature)
        
        if use_multiscale:
            self.multiscale_proj = MultiScaleConditionalProjection()
        
        if use_stats:
            self.stats_extractor = MIDIStatisticalFeatures()
        
        if use_self_supervised:
            self.alignment_head = SelfSupervisedAlignmentHead()
    
    def compute_alignment_loss(self, midi_features, audio_features):
        """
        计算对齐损失
        
        Args:
            midi_features: MIDI 特征 (B, T_midi, D)
            audio_features: 音频特征 (B, C, T_audio) 或 (B, T_audio, D)
        
        Returns:
            total_loss: 总的对齐损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 对比学习损失
        if self.use_contrastive:
            # 将音频特征转换为合适的格式
            if audio_features.dim() == 3 and audio_features.size(1) != midi_features.size(-1):
                # 卷积格式 (B, C, T) -> 全局池化
                audio_global = audio_features.mean(dim=-1)  # (B, C)
                # 投影到 MIDI 特征空间
                audio_global = F.linear(audio_global, 
                                       torch.randn(midi_features.size(-1), audio_features.size(1), 
                                                  device=audio_features.device))
            else:
                audio_global = audio_features
            
            contrastive_loss_val, _ = self.contrastive_module(midi_features, audio_global)
            loss_dict['contrastive'] = contrastive_loss_val
            total_loss += self.contrastive_weight * contrastive_loss_val
        
        # 自监督对齐损失
        if self.use_self_supervised:
            # 正样本
            pos_score = self.alignment_head(midi_features, audio_global)
            pos_loss = F.binary_cross_entropy(pos_score, torch.ones_like(pos_score))
            
            # 负样本（随机打乱）
            neg_indices = torch.randperm(midi_features.size(0), device=midi_features.device)
            neg_midi = midi_features[neg_indices]
            neg_score = self.alignment_head(neg_midi, audio_global)
            neg_loss = F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))
            
            alignment_loss = pos_loss + neg_loss
            loss_dict['self_supervised'] = alignment_loss
            total_loss += 0.1 * alignment_loss
        
        return total_loss, loss_dict
    
    def augment_data(self, midi_tokens, audio_mel, prob=0.3):
        """
        数据增强
        
        Args:
            midi_tokens: MIDI tokens (B, T)
            audio_mel: 音频 Mel 频谱 (B, C, T)
            prob: 增强概率
        
        Returns:
            midi_tokens_aug: 增强后的 MIDI tokens
            audio_mel_aug: 增强后的音频 Mel 频谱
        """
        if self.use_weak_aug and random.random() < prob:
            return create_weak_alignment_pairs(midi_tokens, audio_mel, alignment_strategy='shift')
        else:
            return midi_tokens, audio_mel

