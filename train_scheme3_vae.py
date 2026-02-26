"""
方案三 VAE 训练脚本（独立模块）
预训练 VAE 编码器/解码器，用于潜在空间扩散模型

训练目标：
- 学习音频 Mel 频谱到潜在空间的映射
- 使用重建损失（MSE）训练编码器-解码器
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
import glob
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio.scheme3 import LatentSpaceConditionalModel # LatentSpaceConditionalModel的作用是：包含VAE编码器/解码器、MIDI编码器、Unet扩散模型
from audio.stft import TacotronSTFT # TacotronSTFT的作用是：TacotronSTFT是用于将音频转换为Mel频谱的模型，它是一个基于Tacotron的模型，用于将音频转换为Mel频谱。
from audio.tools import wav_to_fbank # wav文件预处理函数


def load_wav_files(audio_wav_dir: str, train_ratio: float = 0.9): 
    """
    加载 WAV 文件并分割训练/验证集
    
    Args:
        audio_wav_dir: WAV 文件目录
        train_ratio: 训练集比例
        
    Returns:
        train_wav_files: 训练集 WAV 文件列表
        val_wav_files: 验证集 WAV 文件列表
    """
    wav_files = glob.glob(os.path.join(audio_wav_dir, "**", "*.wav"), recursive=True) # glob.glob 用于查找指定路径下的所有文件
    random.shuffle(wav_files)
    split_idx = int(len(wav_files) * train_ratio)
    train_wav_files = wav_files[:split_idx]
    val_wav_files = wav_files[split_idx:]
    return train_wav_files, val_wav_files


def process_wav_batch(wav_files, stft_processor, max_frames, device):
    """
    处理一批 WAV 文件，转换为 Mel 频谱
    
    Args:
        wav_files: WAV 文件路径列表
        stft_processor: STFT 处理器
        max_frames: 最大帧数
        device: 设备
        
    Returns:
        mel_specs_tensor: Mel 频谱 tensor (B, n_mel_channels, T) 或 None
    """
    batch_mel_specs = []
    
    for wav_path in wav_files:
        if os.path.exists(wav_path):
            try:
                fbank, _, _ = wav_to_fbank(wav_path, target_length=max_frames, fn_STFT=stft_processor)
                mel_spec = torch.FloatTensor(fbank.T).unsqueeze(0)  # (1, n_mel_channels, T)
                batch_mel_specs.append(mel_spec.squeeze(0))
            except Exception as e:
                print(f"Warning: Failed to process {wav_path}: {e}")
                continue
    
    if len(batch_mel_specs) == 0:
        return None
    
    # 填充到相同长度
    max_mel_len = max(mel.shape[1] for mel in batch_mel_specs)
    padded_mel_specs = []
    for mel in batch_mel_specs:
        if mel.shape[1] < max_mel_len:
            pad_width = (0, 0, 0, max_mel_len - mel.shape[1])
            mel = torch.nn.functional.pad(mel, pad_width, mode='constant')
        padded_mel_specs.append(mel)
    
    return torch.stack(padded_mel_specs).to(device)


def train_vae(
    # 数据路径
    audio_wav_dir: str,
    # 模型参数
    n_mel_channels: int = 80,
    latent_dim: int = 32,
    compression_factor: int = 4,
    # VAE 参数
    n_filters: int = 128,
    n_residual_layers: int = 3,
    # 训练参数
    batch_size: int = 4,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    max_frames: int = 512,
    # 其他参数
    save_dir: str = "saved/scheme3/vae",
    log_dir: str = "logs/scheme3/vae",
    resume: str = None,
):
    """
    训练 VAE 编码器/解码器
    
    Args:
        audio_wav_dir: 音频 WAV 文件目录
        n_mel_channels: Mel 频谱通道数
        latent_dim: 潜在空间维度
        compression_factor: 时间压缩因子
        n_filters: VAE 卷积过滤器数量
        n_residual_layers: VAE 残差层数量
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        max_frames: 最大帧数
        save_dir: 模型保存目录
        log_dir: 日志目录
        resume: 恢复训练的检查点路径
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 加载数据
    print("Loading audio data for VAE training...")
    train_wav_files, val_wav_files = load_wav_files(audio_wav_dir)
    print(f"Train WAV files: {len(train_wav_files)}, Val WAV files: {len(val_wav_files)}")
    
    if len(train_wav_files) == 0:
        raise ValueError(f"No WAV files found in {audio_wav_dir}")
    
    # 创建模型（只使用 VAE 部分）
    model = LatentSpaceConditionalModel(
        n_mel_channels=n_mel_channels,
        latent_dim=latent_dim,
        compression_factor=compression_factor,
        midi_vocab_size=390,  # 不需要，但模型需要
        embedding_dim=256,     # 不需要，但模型需要
        num_layers=6,          # 不需要，但模型需要
        max_seq=2048,          # 不需要，但模型需要
        dropout=0.2,           # 不需要，但模型需要
        num_timesteps=1000,    # 不需要，但模型需要
        schedule_type='cosine', # 不需要，但模型需要
        base_channels=64,      # 不需要，但模型需要
        num_res_blocks=2,      # 不需要，但模型需要
        n_filters=n_filters,
        n_residual_layers=n_residual_layers,
    ).to(device)
    
    # 只训练 VAE 部分，冻结其他部分 # 冻结策略：其它模块需要参数但是不会求导计算
    for param in model.midi_encoder.parameters():
        param.requires_grad = False
    for param in model.unet.parameters():
        param.requires_grad = False
    
    # 优化器（只优化 VAE） # 如果p.requires_grad为True，则该参数会参与优化器的优化
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    
    # 损失函数
    mse_loss_fn = nn.MSELoss()
    
    # 加载检查点
    start_epoch = 0
    if resume:
        print(f"Loading checkpoint from {resume}...")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
    
    # STFT 处理器
    stft_processor = TacotronSTFT(
        filter_length=1024,
        hop_length=160,
        win_length=1024,
        n_mel_channels=n_mel_channels,
        sampling_rate=16000,
        mel_fmin=0.0,
        mel_fmax=8000.0,
    ) 
    
    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # 训练阶段
        random.shuffle(train_wav_files)
        num_batches = len(train_wav_files) // batch_size
        
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs} [VAE]")
        for batch_idx in progress_bar:
            # 获取批次 WAV 文件
            batch_wav_files = train_wav_files[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            # 处理批次数据
            mel_specs_tensor = process_wav_batch(batch_wav_files, stft_processor, max_frames, device)
            if mel_specs_tensor is None:
                continue
            
            # VAE 前向传播：编码 -> 解码
            latent = model.vae_encoder(mel_specs_tensor)
            reconstructed_mel = model.vae_decoder(latent)
            
            # 计算重建损失
            loss = mse_loss_fn(reconstructed_mel, mel_specs_tensor)
            
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
            random.shuffle(val_wav_files)
            num_val_batches = len(val_wav_files) // batch_size
            
            for batch_idx in range(num_val_batches):
                batch_wav_files = val_wav_files[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                
                mel_specs_tensor = process_wav_batch(batch_wav_files, stft_processor, max_frames, device)
                if mel_specs_tensor is None:
                    continue
                
                # VAE 前向传播
                latent = model.vae_encoder(mel_specs_tensor)
                reconstructed_mel = model.vae_decoder(latent)
                loss = mse_loss_fn(reconstructed_mel, mel_specs_tensor)
                
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
            checkpoint_path = os.path.join(save_dir, f"vae_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss if num_batches > 0 else 0,
                'val_loss': avg_val_loss if val_num_batches > 0 else None,
                'train_stage': 'vae',
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    writer.close()
    print("VAE training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Scheme 3 VAE Encoder/Decoder")
    parser.add_argument("--audio_wav_dir", type=str, required=True, help="Audio WAV files directory")
    parser.add_argument("--save_dir", type=str, default="saved/scheme3/vae", help="Model save directory")
    parser.add_argument("--log_dir", type=str, default="logs/scheme3/vae", help="Log directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_mel_channels", type=int, default=80, help="Mel spectrogram channels")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent space dimension")
    parser.add_argument("--compression_factor", type=int, default=4, help="Time compression factor")
    parser.add_argument("--n_filters", type=int, default=128, help="VAE convolution filters")
    parser.add_argument("--n_residual_layers", type=int, default=3, help="VAE residual layers")
    parser.add_argument("--max_frames", type=int, default=512, help="Maximum frames")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    train_vae(
        audio_wav_dir=args.audio_wav_dir,
        n_mel_channels=args.n_mel_channels,
        latent_dim=args.latent_dim,
        compression_factor=args.compression_factor,
        n_filters=args.n_filters,
        n_residual_layers=args.n_residual_layers,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_frames=args.max_frames,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume=args.resume,
    )

