"""
潜在空间编码器模块
参考 Tango/AudioLDM 的潜在空间表示方式

提供：
1. 将音频编码到潜在空间（基于 Mel 频谱的特征压缩）
2. 潜在空间到音频的解码（简化版本）
3. 潜在空间特征提取，用于训练扩散模型或 Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from .stft import TacotronSTFT
from .tools import wav_to_fbank, read_wav_file


class LatentAudioEncoder(nn.Module):
    """
    潜在空间音频编码器
    参考 AudioLDM/Tango 的设计，将音频特征编码到潜在空间
    
    架构：
    - 输入：Mel 频谱图 (n_mel_channels, T)
    - 编码器：卷积层 + 残差块
    - 输出：潜在空间特征 (latent_dim, T_compressed)
    """
    
    def __init__(
        self,
        n_mel_channels: int = 80,
        latent_dim: int = 32,
        n_filters: int = 128,
        n_residual_layers: int = 3,
        compression_factor: int = 4,
    ):
        """
        Args:
            n_mel_channels: Mel 频谱通道数
            latent_dim: 潜在空间维度
            n_filters: 卷积过滤器数量
            n_residual_layers: 残差层数量
            compression_factor: 时间维度压缩因子
        """
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.latent_dim = latent_dim
        self.compression_factor = compression_factor
        
        # 初始卷积层
        self.conv_in = nn.Conv1d(
            n_mel_channels, n_filters, kernel_size=7, stride=1, padding=3
        )
        
        # 残差块
        self.residual_layers = nn.ModuleList([
            ResidualBlock(n_filters, dilation=2 ** i)
            for i in range(n_residual_layers)
        ])
        
        # 压缩层（时间维度下采样）
        self.conv_compress = nn.Conv1d(
            n_filters, n_filters, 
            kernel_size=compression_factor, 
            stride=compression_factor, 
            padding=0
        )
        
        # 输出层
        self.conv_out = nn.Conv1d(n_filters, latent_dim, kernel_size=1)
        
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        编码 Mel 频谱到潜在空间
        
        Args:
            mel_spec: (B, n_mel_channels, T) 或 (n_mel_channels, T)
            
        Returns:
            latent: (B, latent_dim, T//compression_factor) 或 (latent_dim, T//compression_factor)
        """
        # 确保是 3D tensor
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 初始卷积
        x = F.relu(self.conv_in(mel_spec))
        
        # 残差块
        for residual in self.residual_layers:
            x = residual(x)
        
        # 压缩时间维度
        x = F.relu(self.conv_compress(x))
        
        # 输出层
        latent = self.conv_out(x)
        
        if squeeze_output:
            latent = latent.squeeze(0)
        
        return latent


class LatentAudioDecoder(nn.Module):
    """
    潜在空间音频解码器
    将潜在空间特征解码回 Mel 频谱图
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        n_mel_channels: int = 80,
        n_filters: int = 128,
        n_residual_layers: int = 3,
        compression_factor: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mel_channels = n_mel_channels
        self.compression_factor = compression_factor
        
        # 输入层
        self.conv_in = nn.Conv1d(latent_dim, n_filters, kernel_size=1)
        
        # 上采样层（恢复时间维度）
        self.conv_upsample = nn.ConvTranspose1d(
            n_filters, n_filters,
            kernel_size=compression_factor,
            stride=compression_factor,
            padding=0
        )
        
        # 残差块
        self.residual_layers = nn.ModuleList([
            ResidualBlock(n_filters, dilation=2 ** i)
            for i in range(n_residual_layers)
        ])
        
        # 输出层
        self.conv_out = nn.Conv1d(
            n_filters, n_mel_channels, kernel_size=7, stride=1, padding=3
        )
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        解码潜在空间特征到 Mel 频谱
        
        Args:
            latent: (B, latent_dim, T) 或 (latent_dim, T)
            
        Returns:
            mel_spec: (B, n_mel_channels, T*compression_factor) 或 (n_mel_channels, T*compression_factor)
        """
        if latent.dim() == 2:
            latent = latent.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 输入层
        x = F.relu(self.conv_in(latent))
        
        # 上采样
        x = F.relu(self.conv_upsample(x))
        
        # 残差块
        for residual in self.residual_layers:
            x = residual(x)
        
        # 输出层
        mel_spec = self.conv_out(x)
        
        if squeeze_output:
            mel_spec = mel_spec.squeeze(0)
        
        return mel_spec


class ResidualBlock(nn.Module):
    """残差块，用于编码器和解码器"""
    
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)


class LatentAudioProcessor:
    """
    潜在空间音频处理器
    提供音频到潜在空间的完整处理流程（参考 Tango 风格）
    """
    
    def __init__(
        self,
        sampling_rate: int = 16000,
        n_mel_channels: int = 80,
        filter_length: int = 1024,
        hop_length: int = 160,
        win_length: int = 1024,
        mel_fmin: float = 0.0,
        mel_fmax: float = 8000.0,
        latent_dim: int = 32,
        compression_factor: int = 4,
        use_pretrained_encoder: bool = False,
        encoder_path: Optional[str] = None,
    ):
        """
        Args:
            sampling_rate: 采样率
            n_mel_channels: Mel 频谱通道数
            filter_length: FFT 窗口长度
            hop_length: 帧移
            win_length: 窗口长度
            mel_fmin: Mel 滤波器最小频率
            mel_fmax: Mel 滤波器最大频率
            latent_dim: 潜在空间维度
            compression_factor: 时间压缩因子
            use_pretrained_encoder: 是否使用预训练编码器
            encoder_path: 预训练编码器路径
        """
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.hop_length = hop_length
        self.latent_dim = latent_dim
        self.compression_factor = compression_factor
        
        # STFT 处理器
        self.stft = TacotronSTFT(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            sampling_rate=sampling_rate,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
        )
        
        # 潜在空间编码器
        self.encoder = LatentAudioEncoder(
            n_mel_channels=n_mel_channels,
            latent_dim=latent_dim,
            compression_factor=compression_factor,
        )
        
        # 加载预训练编码器（如果提供）
        if use_pretrained_encoder and encoder_path:
            ckpt = torch.load(encoder_path, map_location="cpu")
            # 支持两种情况：
            # 1) 直接是 LatentAudioEncoder 的 state_dict
            # 2) 是 LatentSpaceConditionalModel 的 checkpoint，包含 'vae_encoder.' 前缀
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            else:
                state = ckpt

            if any(k.startswith("vae_encoder.") for k in state.keys()):
                encoder_state = {
                    k.replace("vae_encoder.", ""): v
                    for k, v in state.items()
                    if k.startswith("vae_encoder.")
                }
            else:
                encoder_state = state

            missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
            if missing or unexpected:
                print(
                    f"[LatentAudioProcessor] Loaded encoder from '{encoder_path}' "
                    f"with missing keys={missing}, unexpected keys={unexpected}"
                )
            else:
                print(f"[LatentAudioProcessor] Loaded pretrained encoder from '{encoder_path}'")
            self.encoder.eval()
    
    def encode_wav_to_latent(
        self, 
        wav_file: str, 
        target_length: Optional[int] = None
    ) -> np.ndarray:
        """
        将 WAV 文件编码到潜在空间
        
        Args:
            wav_file: WAV 文件路径
            target_length: 目标长度（帧数）
            
        Returns:
            latent: 潜在空间特征 (latent_dim, T_compressed)
        """
        # 计算 Mel 频谱
        fbank, log_magnitudes_stft, energy = wav_to_fbank(
            wav_file, target_length=target_length, fn_STFT=self.stft
        )
        
        # 转换为 torch tensor (n_mel_channels, T)
        mel_spec = torch.FloatTensor(fbank.T)
        
        # 编码到潜在空间
        with torch.no_grad():
            latent = self.encoder(mel_spec)
        
        # 转换为 numpy
        latent_np = latent.cpu().numpy()
        
        return latent_np
    
    def encode_waveform_to_latent(
        self,
        waveform: np.ndarray,
        target_length: Optional[int] = None
    ) -> np.ndarray:
        """
        从音频波形数组编码到潜在空间
        
        Args:
            waveform: 音频波形数组 (n_samples,)
            target_length: 目标长度（帧数）
            
        Returns:
            latent: 潜在空间特征
        """
        # 确保是单声道
        if len(waveform.shape) > 1:
            waveform = waveform[0] if waveform.shape[0] == 1 else np.mean(waveform, axis=0)
        
        # 转换为 torch tensor
        waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0)
        
        # 计算 Mel 频谱
        mel_spec, log_magnitudes_stft, energy = self.stft.mel_spectrogram(waveform_tensor)
        mel_spec = mel_spec.squeeze(0)  # (n_mel_channels, T)
        
        # 编码到潜在空间
        with torch.no_grad():
            latent = self.encoder(mel_spec)
        
        return latent.cpu().numpy()
    
    def get_mel_from_wav(
        self,
        wav_file: str,
        target_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 WAV 文件提取 Mel 频谱和能量
        
        Args:
            wav_file: WAV 文件路径
            target_length: 目标长度
            
        Returns:
            (mel_spec, energy): Mel 频谱和能量
        """
        fbank, log_magnitudes_stft, energy = wav_to_fbank(
            wav_file, target_length=target_length, fn_STFT=self.stft
        )
        return fbank, energy


def create_latent_encoder(
    latent_dim: int = 32,
    compression_factor: int = 4,
    n_mel_channels: int = 80,
) -> LatentAudioEncoder:
    """
    创建潜在空间编码器（便捷函数）
    """
    return LatentAudioEncoder(
        n_mel_channels=n_mel_channels,
        latent_dim=latent_dim,
        compression_factor=compression_factor,
    )


def create_latent_decoder(
    latent_dim: int = 32,
    compression_factor: int = 4,
    n_mel_channels: int = 80,
) -> LatentAudioDecoder:
    """
    创建潜在空间解码器（便捷函数）
    """
    return LatentAudioDecoder(
        latent_dim=latent_dim,
        n_mel_channels=n_mel_channels,
        compression_factor=compression_factor,
    )


if __name__ == '__main__':
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        wav_file = sys.argv[1]
        print(f"测试潜在空间编码: {wav_file}")
        
        processor = LatentAudioProcessor()
        latent = processor.encode_wav_to_latent(wav_file)
        
        print(f"潜在空间特征形状: {latent.shape}")
        print(f"潜在空间维度: {processor.latent_dim}")
        print(f"压缩因子: {processor.compression_factor}")
    else:
        print("用法: python latent_encoder.py <wav_file>")

