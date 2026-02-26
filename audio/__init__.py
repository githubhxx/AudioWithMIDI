"""
音频处理模块 - 方案三专用
包含VAE编码器/解码器和潜在空间处理
"""

from .latent_encoder import (
    LatentAudioEncoder,
    LatentAudioDecoder,
    LatentAudioProcessor,
    create_latent_encoder,
    create_latent_decoder,
)
from .latent_preprocess import (
    preprocess_audio_to_latent,
    preprocess_audio_files_to_latent,
)
from .stft import TacotronSTFT
from .tools import wav_to_fbank, read_wav_file

__all__ = [
    'LatentAudioEncoder',
    'LatentAudioDecoder',
    'LatentAudioProcessor',
    'create_latent_encoder',
    'create_latent_decoder',
    'preprocess_audio_to_latent',
    'preprocess_audio_files_to_latent',
    'TacotronSTFT',
    'wav_to_fbank',
    'read_wav_file',
]

