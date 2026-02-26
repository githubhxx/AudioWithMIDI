"""
方案三训练脚本（方式 3.2：潜在空间扩散模型，阶段二）
参考 Tango 项目的扩散模型训练方式

训练策略（严格两阶段）：
1. 阶段一：使用 train_scheme3_vae.py 预训练 VAE 编码器/解码器
2. 阶段二（本文件）：在预计算的潜在空间上训练扩散模型，VAE 完全冻结

本脚本只负责阶段二：
- 直接读取 latent_preprocess.py 预先生成的音频潜在特征（.pickle）
- 使用 MIDI token 序列作为条件，在潜在空间中训练扩散 UNet
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
import random
import pickle

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio.scheme3 import LatentSpaceConditionalModel
from midi.custom.config import config
from midi.preprocess import preprocess_midi


# 使用自定义的配对数据加载函数
def iter_smd_pairs_for_latent(midi_dir: str, audio_latent_dir: str, split: str = "train"):
    """
    迭代 SMD 数据集的 MIDI 和潜在特征配对
    
    Args:
        midi_dir: MIDI 文件目录
        audio_latent_dir: 音频潜在特征目录
        split: 数据分割 ("train", "val", "test")
        
    Yields:
        (midi_path, audio_latent_path) 元组
    """
    import glob
    midi_files = glob.glob(os.path.join(midi_dir, "**", "*.mid"), recursive=True)
    midi_files.extend(glob.glob(os.path.join(midi_dir, "**", "*.midi"), recursive=True)) # mide_files.extend的作用是：将midi_dir目录下的所有midi文件添加到midi_files列表中
    
    # 简单的数据分割（可以根据需要改进）
    random.shuffle(midi_files)
    if split == "train":
        midi_files = midi_files[:int(len(midi_files) * 0.8)]
    elif split == "val":
        midi_files = midi_files[int(len(midi_files) * 0.8):int(len(midi_files) * 0.9)]
    else:  # test
        midi_files = midi_files[int(len(midi_files) * 0.9):]
    
    for midi_path in midi_files:
        # 找到对应的音频潜在特征文件
        basename = os.path.splitext(os.path.basename(midi_path))[0] # os.path.splitext(path) 用于将路径拆分为文件名和扩展名
        audio_latent_path = os.path.join(audio_latent_dir, f"{basename}.pickle")
        
        if os.path.exists(audio_latent_path):
            yield midi_path, audio_latent_path


def train_scheme3_latent_conditional(
    # 数据路径
    audio_latent_dir: str,
    midi_dir: str,
    # 模型参数
    latent_dim: int = 32,
    compression_factor: int = 4,
    midi_vocab_size: int = None,
    embedding_dim: int = 256,
    num_layers: int = 6,
    max_seq: int = 2048,
    dropout: float = 0.2,
    # 扩散模型参数
    num_timesteps: int = 1000,
    schedule_type: str = 'cosine',
    # UNet 参数
    base_channels: int = 64,
    num_res_blocks: int = 2,
    # 训练参数
    batch_size: int = 4,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    # 无分类器引导参数
    condition_dropout_rate: float = 0.15,  # 条件丢弃率（用于无分类器引导训练）
    # 其他参数
    save_dir: str = "saved/scheme3",
    log_dir: str = "logs/scheme3",
    resume: str = None,
):
    """
    训练方案三潜在空间条件生成模型（扩散模型）
    
    Args:
        audio_latent_dir: 预计算的音频潜在特征目录（.pickle 文件）
        midi_dir: MIDI 文件目录
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ===== 初始化 MIDI 配置（兼容 MusicTransformer-pytorch）=====
    # 如果尚未设置 event_dim / pad_token，则按原项目约定初始化：
    # event_dim = 128(note_on) + 128(note_off) + 100(time_shift) + 32(velocity) = 388
    if 'event_dim' not in config.dict:
        config.event_dim = 388
    # 如果还没有 pad_token / vocab_size 等，就根据 event_dim 派生
    if 'pad_token' not in config.dict:
        config._set_vocab_params()
    # 如果未显式指定 midi_vocab_size，则使用配置中的 vocab_size，避免与 event_dim 不一致
    if midi_vocab_size is None:
        midi_vocab_size = config.vocab_size
    # 现在可以安全使用 config.pad_token / config.vocab_size
    # ==========================================================
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 阶段二：只使用配对的 MIDI + 预计算潜在特征
    print("Loading paired MIDI and latent audio data...")
    train_pairs = list(iter_smd_pairs_for_latent(midi_dir, audio_latent_dir, split="train"))
    val_pairs = list(iter_smd_pairs_for_latent(midi_dir, audio_latent_dir, split="val"))
    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    
    # 模型
    model = LatentSpaceConditionalModel(
        latent_dim=latent_dim,
        compression_factor=compression_factor,
        midi_vocab_size=midi_vocab_size,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        max_seq=max_seq,
        dropout=dropout,
        num_timesteps=num_timesteps,
        schedule_type=schedule_type,
        base_channels=base_channels,
        num_res_blocks=num_res_blocks,
    ).to(device)
    
    # 阶段二：始终冻结 VAE 编码器和解码器（严格两阶段训练）
    print("Freezing VAE encoder and decoder (two-stage training)...")
    for param in model.vae_encoder.parameters():
        param.requires_grad = False
    for param in model.vae_decoder.parameters():
        param.requires_grad = False
    
    # 优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    
    # 损失函数
    mse_loss_fn = nn.MSELoss()  # 扩散模型和 VAE 重建损失
    
    # 加载检查点
    start_epoch = 0
    if resume:
        print(f"Loading checkpoint from {resume}...")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    
    # 训练循环（仅扩散阶段）
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0  # 实际参与优化的 batch 数量
        
        # 扩散模型训练
        random.shuffle(train_pairs)
        num_steps = len(train_pairs) // batch_size
        
        progress_bar = tqdm(range(num_steps), desc=f"Epoch {epoch+1}/{num_epochs} [Diffusion]")
        for batch_idx in progress_bar:
            # 获取批次数据
            batch_pairs = train_pairs[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            batch_midi_tokens = []
            batch_latents = []
            
            for midi_path, audio_latent_path in batch_pairs:
                # 加载 MIDI tokens
                try:
                    midi_tokens = preprocess_midi(midi_path)
                    if len(midi_tokens) > max_seq:
                        midi_tokens = midi_tokens[:max_seq]
                    batch_midi_tokens.append(midi_tokens)
                except Exception as e:
                    print(f"Warning: Failed to process MIDI {midi_path}: {e}")
                    continue
                
                # 加载预计算的潜在特征
                if os.path.exists(audio_latent_path):
                    try:
                        with open(audio_latent_path, "rb") as f:
                            latent = pickle.load(f)  # 期望形状: (latent_dim, T_compressed)
                        latent_tensor = torch.from_numpy(latent).float()
                        batch_latents.append(latent_tensor)
                    except Exception as e:
                        print(f"Warning: Failed to load latent {audio_latent_path}: {e}")
                        continue
                else:
                    continue
            
            if len(batch_latents) == 0 or len(batch_midi_tokens) == 0:
                continue
            
            # 确保 MIDI 和潜在特征数量匹配
            min_len = min(len(batch_midi_tokens), len(batch_latents))
            batch_midi_tokens = batch_midi_tokens[:min_len]
            batch_latents = batch_latents[:min_len]
            
            # 填充到相同长度
            max_midi_len = max(len(tokens) for tokens in batch_midi_tokens)
            max_latent_len = max(latent.shape[1] for latent in batch_latents)
            
            # MIDI tokens
            padded_midi_tokens = []
            for tokens in batch_midi_tokens:
                padded = tokens + [config.pad_token] * (max_midi_len - len(tokens))
                padded_midi_tokens.append(padded)
            
            # 潜在特征
            padded_latents = []
            for latent in batch_latents:
                if latent.shape[1] < max_latent_len:
                    # 对形状为 (latent_dim, T) 的潜在特征，仅在时间维度右侧补零
                    pad_width = (0, max_latent_len - latent.shape[1])  # (pad_left, pad_right)
                    latent = torch.nn.functional.pad(latent, pad_width, mode="constant")
                padded_latents.append(latent)
            
            # 转换为 tensor
            midi_tokens_tensor = torch.LongTensor(padded_midi_tokens).to(device)
            latents_tensor = torch.stack(padded_latents).to(device)
            
            # 前向传播（扩散模型训练，支持无分类器引导，直接使用预计算潜在特征）
            predicted_noise, noise = model.forward_with_latent(
                latents_tensor,
                midi_tokens_tensor,
                condition_dropout_rate=condition_dropout_rate,
            )
            
            # 计算损失（预测噪声和真实噪声的 MSE）
            loss = mse_loss_fn(predicted_noise, noise)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            # 记录
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })
        
        # 验证阶段（扩散）
        val_loss = 0
        val_num_batches = 0
        
        model.eval()
        with torch.no_grad():
            random.shuffle(val_pairs)
            num_val_batches = len(val_pairs) // batch_size
            
            for batch_idx in range(num_val_batches):
                batch_pairs = val_pairs[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                batch_midi_tokens = []
                batch_latents = []
                
                for midi_path, audio_latent_path in batch_pairs:
                    try:
                        midi_tokens = preprocess_midi(midi_path)
                        if len(midi_tokens) > max_seq:
                            midi_tokens = midi_tokens[:max_seq]
                        batch_midi_tokens.append(midi_tokens)
                    except Exception:
                        continue
                    
                    if os.path.exists(audio_latent_path):
                        try:
                            with open(audio_latent_path, "rb") as f:
                                latent = pickle.load(f)
                            latent_tensor = torch.from_numpy(latent).float()
                            batch_latents.append(latent_tensor)
                        except Exception:
                            continue
                
                if len(batch_latents) == 0 or len(batch_midi_tokens) == 0:
                    continue
                
                min_len = min(len(batch_midi_tokens), len(batch_latents))
                batch_midi_tokens = batch_midi_tokens[:min_len]
                batch_latents = batch_latents[:min_len]
                
                # 填充到相同长度
                max_midi_len = max(len(tokens) for tokens in batch_midi_tokens)
                max_latent_len = max(latent.shape[1] for latent in batch_latents)
                
                padded_midi_tokens = []
                for tokens in batch_midi_tokens:
                    padded = tokens + [config.pad_token] * (max_midi_len - len(tokens))
                    padded_midi_tokens.append(padded)
                
                padded_latents = []
                for latent in batch_latents:
                    if latent.shape[1] < max_latent_len:
                        # 对形状为 (latent_dim, T) 的潜在特征，仅在时间维度右侧补零
                        pad_width = (0, max_latent_len - latent.shape[1])  # (pad_left, pad_right)
                        latent = torch.nn.functional.pad(latent, pad_width, mode="constant")
                    padded_latents.append(latent)
                
                midi_tokens_tensor = torch.LongTensor(padded_midi_tokens).to(device)
                latents_tensor = torch.stack(padded_latents).to(device)
                
                # 扩散模型前向传播（验证时不使用条件丢弃），直接使用预计算潜在特征
                predicted_noise, noise = model.forward_with_latent(
                    latents_tensor,
                    midi_tokens_tensor,
                    condition_dropout_rate=0.0,  # 验证时总是使用条件
                )
                loss = mse_loss_fn(predicted_noise, noise)
                
                val_loss += loss.item()
                val_num_batches += 1
        
        # 记录到 TensorBoard
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            writer.add_scalar('Loss/Train', avg_loss, epoch)
        
        if val_num_batches > 0:
            avg_val_loss = val_loss / val_num_batches
            writer.add_scalar('Loss/Val', avg_val_loss, epoch)
            print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        else:
            if num_batches > 0:
                print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % 10 == 0 and num_batches > 0:
            checkpoint_path = os.path.join(save_dir, f"scheme3_epoch_{epoch+1}.pt")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss if num_batches > 0 else 0,
            }
            if val_num_batches > 0:
                checkpoint['val_loss'] = avg_val_loss
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Scheme 3 Latent Diffusion Model")
    parser.add_argument("--audio_latent_dir", type=str, required=True, help="Audio latent features directory")
    parser.add_argument("--midi_dir", type=str, required=True, help="MIDI files directory")
    parser.add_argument("--save_dir", type=str, default="saved/scheme3", help="Model save directory")
    parser.add_argument("--log_dir", type=str, default="logs/scheme3", help="Log directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--schedule_type", type=str, default="cosine", choices=["linear", "cosine"],
                       help="Noise schedule type")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--condition_dropout_rate", type=float, default=0.15,
                       help="Condition dropout rate for classifier-free guidance (0.0-1.0, recommended: 0.1-0.2)")
    
    args = parser.parse_args()
    
    train_scheme3_latent_conditional(
        audio_latent_dir=args.audio_latent_dir,
        midi_dir=args.midi_dir,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_timesteps=args.num_timesteps,
        schedule_type=args.schedule_type,
        resume=args.resume,
        condition_dropout_rate=args.condition_dropout_rate,
    )
