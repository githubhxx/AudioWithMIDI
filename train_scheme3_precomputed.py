"""
方案三训练脚本（优化版：使用预计算潜在特征）
方式 3.2：潜在空间扩散模型

本文件是 train_scheme3.py 的优化版本，直接使用预计算的潜在特征（.pickle文件），
而不是从WAV文件重新计算Mel频谱和VAE编码。

优势：
1. 训练速度显著提升（2-5倍）
2. 节省计算资源（不需要STFT和VAE编码）
3. 更符合两阶段训练理念

训练策略：
1. 两阶段训练（推荐）：
   - 阶段一：预训练 VAE 编码器/解码器（使用 train_scheme3_vae.py）
   - 阶段二：使用预计算潜在特征训练扩散模型（使用本文件）

使用前准备：
1. 完成阶段一：训练VAE（train_scheme3_vae.py）
2. 生成潜在特征：使用 generate_latent_vectors.py 批量生成潜在特征
3. 阶段二训练：使用本文件训练扩散模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
import pickle
from tqdm import tqdm
from tensorboardX import SummaryWriter
import random

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio.scheme3 import LatentSpaceConditionalModel
from midi.custom.config import config
from midi.preprocess import preprocess_midi


def iter_smd_pairs_for_latent(midi_dir: str, audio_latent_dir: str, split: str = "train"):
    """
    迭代 SMD 数据集的 MIDI 和潜在特征配对
    
    Args:
        midi_dir: MIDI 文件目录
        audio_latent_dir: 音频潜在特征目录（.pickle 文件）
        split: 数据分割 ("train", "val", "test")
        
    Yields:
        (midi_path, audio_latent_path) 元组
    """
    import glob
    midi_files = glob.glob(os.path.join(midi_dir, "**", "*.mid"), recursive=True)
    midi_files.extend(glob.glob(os.path.join(midi_dir, "**", "*.midi"), recursive=True))
    
    # 简单的数据分割
    random.shuffle(midi_files)
    if split == "train":
        midi_files = midi_files[:int(len(midi_files) * 0.8)]
    elif split == "val":
        midi_files = midi_files[int(len(midi_files) * 0.8):int(len(midi_files) * 0.9)]
    else:  # test
        midi_files = midi_files[int(len(midi_files) * 0.9):]
    
    for midi_path in midi_files:
        # 找到对应的音频潜在特征文件
        basename = os.path.splitext(os.path.basename(midi_path))[0]
        audio_latent_path = os.path.join(audio_latent_dir, f"{basename}.pickle")
        
        if os.path.exists(audio_latent_path):
            yield midi_path, audio_latent_path


def load_latent_from_pickle(pickle_path: str, max_frames: int = None):
    """
    从 Pickle 文件加载潜在特征
    
    Args:
        pickle_path: Pickle 文件路径
        max_frames: 最大帧数（时间维度），如果为 None 则返回完整特征
        
    Returns:
        latent: 潜在特征 numpy 数组 (latent_dim, T_compressed) 或 None
    """
    if not os.path.exists(pickle_path):
        return None
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        # 转换为 numpy 数组
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # 确保是 2D 数组 (latent_dim, T_compressed)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            # 如果是 3D，取第一个样本或展平
            data = data[0] if data.shape[0] == 1 else data.reshape(data.shape[0], -1)
        
        # 时间维度裁剪
        if max_frames is not None and data.shape[1] > max_frames:
            start = random.randrange(0, data.shape[1] - max_frames)
            data = data[:, start:start + max_frames]
        
        return data
        
    except Exception as e:
        print(f"Warning: Failed to load latent from {pickle_path}: {e}")
        return None


def train_scheme3_with_precomputed_latent(
    # 数据路径
    audio_latent_dir: str,
    midi_dir: str,
    # 模型参数
    n_mel_channels: int = 80,
    latent_dim: int = 32,
    compression_factor: int = 4,
    midi_vocab_size: int = 390,
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
    max_frames: int = 512,
    # 其他参数
    save_dir: str = "saved/scheme3",
    log_dir: str = "logs/scheme3",
    resume: str = None,  # VAE 检查点路径（必需！）
):
    """
    使用预计算潜在特征训练方案三潜在空间条件生成模型（扩散模型）
    
    本函数直接使用预计算的潜在特征（.pickle文件），不需要从WAV文件重新计算。
    
    Args:
        audio_latent_dir: 音频潜在特征目录（.pickle 文件）
        midi_dir: MIDI 文件目录
        resume: VAE 检查点路径（必需！用于加载预训练的VAE）
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
    # 现在可以安全使用 config.pad_token / config.vocab_size
    # ==========================================================


    # 检查 VAE 检查点
    if resume is None:
        raise ValueError("必须提供 VAE 检查点路径（--resume 参数）！")
    if not os.path.exists(resume):
        raise FileNotFoundError(f"VAE 检查点不存在: {resume}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 数据加载：需要配对的数据（MIDI + 预计算潜在特征）
    print("Loading paired MIDI and precomputed latent data...")
    train_pairs = list(iter_smd_pairs_for_latent(midi_dir, audio_latent_dir, split="train"))
    val_pairs = list(iter_smd_pairs_for_latent(midi_dir, audio_latent_dir, split="val"))
    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    
    if len(train_pairs) == 0:
        raise ValueError(f"未找到训练数据对！请检查 {midi_dir} 和 {audio_latent_dir}")
    
    # 模型
    model = LatentSpaceConditionalModel(
        n_mel_channels=n_mel_channels,
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
    
    # 加载预训练 VAE（必需！）
    print(f"Loading VAE checkpoint from {resume}...")
    checkpoint = torch.load(resume, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("VAE loaded successfully!")
    
    # 冻结 VAE 参数（只训练扩散模型）
    print("Freezing VAE encoder and decoder...")
    for param in model.vae_encoder.parameters():
        param.requires_grad = False
    for param in model.vae_decoder.parameters():
        param.requires_grad = False
    
    # 优化器（只优化可训练参数：MIDI编码器 + UNet）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Frozen parameters (VAE): {sum(p.numel() for p in model.vae_encoder.parameters()) + sum(p.numel() for p in model.vae_decoder.parameters()):,}")
    
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    
    # 损失函数
    mse_loss_fn = nn.MSELoss()
    
    # 加载训练检查点（如果提供）
    start_epoch = 0
    if os.path.exists(os.path.join(save_dir, "latest.pt")):
        latest_checkpoint = os.path.join(save_dir, "latest.pt")
        print(f"Loading training checkpoint from {latest_checkpoint}...")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")
    
    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # 扩散模型训练
        random.shuffle(train_pairs)
        num_batches_total = len(train_pairs) // batch_size
        
        progress_bar = tqdm(range(num_batches_total), desc=f"Epoch {epoch+1}/{num_epochs} [Diffusion]")
        for batch_idx in progress_bar:
            # 获取批次数据
            batch_pairs = train_pairs[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            batch_midi_tokens = []
            batch_latents = []  # 直接使用潜在特征
            
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
                
                # 直接加载预计算的潜在特征
                latent_data = load_latent_from_pickle(audio_latent_path, max_frames=max_frames)
                if latent_data is None:
                    continue
                
                # 转换为 tensor
                latent_tensor = torch.FloatTensor(latent_data)  # (latent_dim, T_compressed)
                batch_latents.append(latent_tensor)
            
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
                    pad_width = (0, 0, 0, max_latent_len - latent.shape[1])
                    latent = torch.nn.functional.pad(latent, pad_width, mode='constant')
                padded_latents.append(latent)
            
            # 转换为 tensor
            midi_tokens_tensor = torch.LongTensor(padded_midi_tokens).to(device)
            latents_tensor = torch.stack(padded_latents).to(device)  # (B, latent_dim, T_compressed)
            
            # 前向传播（使用预计算潜在特征）
            # 注意：这里直接使用潜在特征，不通过VAE编码器
            predicted_noise, noise = model.forward_with_latent(
                latents_tensor,
                midi_tokens_tensor,
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
        
        # 验证阶段
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
                    
                    latent_data = load_latent_from_pickle(audio_latent_path, max_frames=max_frames)
                    if latent_data is None:
                        continue
                    
                    latent_tensor = torch.FloatTensor(latent_data)
                    batch_latents.append(latent_tensor)
                
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
                        pad_width = (0, 0, 0, max_latent_len - latent.shape[1])
                        latent = torch.nn.functional.pad(latent, pad_width, mode='constant')
                    padded_latents.append(latent)
                
                midi_tokens_tensor = torch.LongTensor(padded_midi_tokens).to(device)
                latents_tensor = torch.stack(padded_latents).to(device)
                
                # 扩散模型前向传播
                predicted_noise, noise = model.forward_with_latent(
                    latents_tensor,
                    midi_tokens_tensor,
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
                'train_stage': 'diffusion',
            }
            if val_num_batches > 0:
                checkpoint['val_loss'] = avg_val_loss
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # 保存最新检查点
        latest_checkpoint_path = os.path.join(save_dir, "latest.pt")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss if num_batches > 0 else 0,
            'train_stage': 'diffusion',
        }
        if val_num_batches > 0:
            checkpoint['val_loss'] = avg_val_loss
        torch.save(checkpoint, latest_checkpoint_path)
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Scheme 3 Diffusion Model with Precomputed Latent Features")
    parser.add_argument("--audio_latent_dir", type=str, required=True, 
                       help="Audio latent features directory (.pickle files)")
    parser.add_argument("--midi_dir", type=str, required=True, 
                       help="MIDI files directory")
    parser.add_argument("--resume", type=str, required=True, 
                       help="VAE checkpoint path (required! for loading pretrained VAE)")
    parser.add_argument("--save_dir", type=str, default="saved/scheme3", 
                       help="Model save directory")
    parser.add_argument("--log_dir", type=str, default="logs/scheme3", 
                       help="Log directory")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, 
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--num_timesteps", type=int, default=1000, 
                       help="Number of diffusion timesteps")
    parser.add_argument("--schedule_type", type=str, default="cosine", 
                       choices=["linear", "cosine"],
                       help="Noise schedule type")
    parser.add_argument("--max_frames", type=int, default=512, 
                       help="Maximum frames for latent features")
    
    args = parser.parse_args()
    
    train_scheme3_with_precomputed_latent(
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
        max_frames=args.max_frames,
    )

