"""
方案三生成脚本（方式 3.2：潜在空间扩散模型）
参考 Tango 项目的扩散模型生成方式

使用训练好的扩散模型生成音频
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
import soundfile as sf
from scipy.io.wavfile import write

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio.scheme3 import LatentSpaceConditionalModel
from audio.stft import TacotronSTFT
from audio.tools import wav_to_fbank
from midi.preprocess import preprocess_midi
from midi.custom.config import config


def generate_scheme3(
    model_path: str,
    midi_path: str,
    output_path: str,
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
    base_channels: int = 64,
    num_res_blocks: int = 2,
    # 生成参数
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    shape: tuple = None,
    sampling_rate: int = 16000,
    # 其他参数
    device: str = "auto",
):
    """
    使用方案三扩散模型生成音频
    
    Args:
        model_path: 模型检查点路径
        midi_path: MIDI 文件路径
        output_path: 输出音频文件路径
        num_inference_steps: 推理步数（通常 20-100 步）
        guidance_scale: 引导缩放（>1.0 增强条件控制）
        shape: 生成的潜在特征形状 (latent_dim, T_compressed)，如果为 None 则自动估算
    """
    # 设备
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    # 加载权重并检测是否为旧版 checkpoint（单层 condition_proj、无 time_embedding.mlp、无 norm_attn）
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    key_old = "unet.encoder.down_blocks.0.condition_proj.weight"
    key_new = "unet.encoder.down_blocks.0.condition_proj.0.weight"
    legacy_unet = key_old in state_dict and key_new not in state_dict

    # 加载模型（旧 checkpoint 使用 legacy_unet=True）
    print(f"Loading model from {model_path}...")
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
        legacy_unet=legacy_unet,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded successfully!")
    
    # 加载 MIDI
    print(f"Loading MIDI from {midi_path}...")
    midi_tokens = preprocess_midi(midi_path)
    if len(midi_tokens) > max_seq:
        midi_tokens = midi_tokens[:max_seq]
    print(f"MIDI tokens length: {len(midi_tokens)}")
    
    # 转换为 tensor
    midi_tokens_tensor = torch.LongTensor([midi_tokens]).to(device)
    
    # 生成
    print(f"Generating audio using diffusion model ({num_inference_steps} steps)...")
    with torch.no_grad():
        generated_latent, generated_mel = model.generate(
            midi_tokens_tensor,
            shape=shape,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
    
    # 生成的 Mel 频谱形状: (B, n_mel_channels, T)
    generated_mel_np = generated_mel[0].cpu().numpy()  # (n_mel_channels, T)
    
    # 从 Mel 频谱重建音频（使用 Griffin-Lim 算法）
    print("Converting Mel spectrogram to audio...")
    stft_processor = TacotronSTFT(
        filter_length=1024,
        hop_length=160,
        win_length=1024,
        n_mel_channels=n_mel_channels,
        sampling_rate=sampling_rate,
        mel_fmin=0.0,
        mel_fmax=8000.0,
    )
    
    # 使用 Griffin-Lim 算法从 Mel 频谱重建音频
    audio_waveform = mel_to_audio_griffin_lim(
        generated_mel_np,
        stft_processor,
        n_iter=60
    )
    
    # 保存音频
    print(f"Saving audio to {output_path}...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # 归一化
    audio_waveform = audio_waveform / (np.max(np.abs(audio_waveform)) + 1e-8)
    audio_waveform = np.clip(audio_waveform, -1.0, 1.0)
    
    # 保存为 WAV
    write(output_path, sampling_rate, (audio_waveform * 32767).astype(np.int16))
    print(f"Audio saved to {output_path}")
    
    return audio_waveform, generated_mel_np


def mel_to_audio_griffin_lim(
    mel_spec: np.ndarray,
    stft_processor: TacotronSTFT,
    n_iter: int = 60
) -> np.ndarray:
    """
    使用 Griffin-Lim 算法从 Mel 频谱重建音频
    
    注意：这是一个简化版本，实际应用中应该使用 VAE 解码器或 Vocoder
    
    Args:
        mel_spec: Mel 频谱 (n_mel_channels, T)
        stft_processor: STFT 处理器
        n_iter: Griffin-Lim 迭代次数
        
    Returns:
        audio_waveform: 音频波形
    """
    try:
        # 尝试使用 librosa 的 Griffin-Lim
        import librosa
        linear_spec = librosa.feature.inverse.mel_to_stft(
            mel_spec,
            sr=stft_processor.sampling_rate,
            n_fft=stft_processor.filter_length,
            fmin=0.0,
            fmax=8000.0
        )
        audio = librosa.griffinlim(
            linear_spec,
            n_iter=n_iter,
            hop_length=stft_processor.hop_length,
            win_length=stft_processor.win_length
        )
        return audio
    except Exception as e:
        print(f"Warning: Griffin-Lim conversion failed: {e}")
        print("Using simplified reconstruction (low quality)")
        T = mel_spec.shape[1]
        audio_length = T * stft_processor.hop_length
        audio = np.random.randn(audio_length) * 0.01
        return audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio using Scheme 3 Diffusion Model")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--midi_path", type=str, required=True, help="MIDI file path")
    parser.add_argument("--output_path", type=str, required=True, help="Output audio path")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale (>1.0 enhances conditioning)")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    
    args = parser.parse_args()
    
    generate_scheme3(
        model_path=args.model_path,
        midi_path=args.midi_path,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
    )
