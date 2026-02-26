"""
潜在空间音频预处理模块
批量处理WAV文件，将其编码到潜在空间并保存为Pickle文件
参考 Tango 的潜在空间处理方式
"""

import pickle
import os
import sys
from progress.bar import Bar
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工具函数
try:
    from midi.utils import find_files_by_extensions
except ImportError:
    # 如果无法导入，定义简化版本
    def find_files_by_extensions(root, exts=[]):
        def _has_ext(name):
            if not exts:
                return True
            name = name.lower()
            for ext in exts:
                if name.endswith(ext):
                    return True
            return False
        for path, _, files in os.walk(root):
            for name in files:
                if _has_ext(name):
                    yield os.path.join(path, name)

from audio.latent_encoder import LatentAudioProcessor


def preprocess_audio_to_latent(
    wav_file: str,
    processor: LatentAudioProcessor = None,
    target_length: int = None,
) -> np.ndarray:
    """
    预处理单个音频文件到潜在空间
    
    Args:
        wav_file: WAV文件路径
        processor: 潜在空间处理器实例，如果为None则创建新的
        target_length: 目标序列长度（帧数）
        
    Returns:
        潜在空间特征数组 (latent_dim, T_compressed)
    """
    if processor is None:
        processor = LatentAudioProcessor()
    return processor.encode_wav_to_latent(wav_file, target_length)


def preprocess_audio_files_to_latent(
    audio_root: str,
    save_dir: str,
    target_length: int = None,
    latent_dim: int = 32,
    compression_factor: int = 4,
    encoder_checkpoint: str = None,
):
    """
    批量处理指定目录下的所有WAV文件到潜在空间
    
    Args:
        audio_root: 包含WAV文件的根目录
        save_dir: 保存Pickle文件的目录
        target_length: 目标序列长度（帧数），如果为None则使用音频实际长度
        latent_dim: 潜在空间维度
        compression_factor: 时间压缩因子
    """
    # 查找所有WAV文件
    audio_paths = list(find_files_by_extensions(audio_root, ['.wav', '.WAV']))
    
    if len(audio_paths) == 0:
        print(f"警告: 在 {audio_root} 中未找到WAV文件")
        return
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建处理器（所有文件共享同一个处理器）
    processor = LatentAudioProcessor(
        latent_dim=latent_dim,
        compression_factor=compression_factor,
        use_pretrained_encoder=encoder_checkpoint is not None,
        encoder_path=encoder_checkpoint,
    )
    
    print(f"找到 {len(audio_paths)} 个WAV文件")
    print(f"保存目录: {save_dir}")
    print(f"潜在空间维度: {latent_dim}")
    print(f"压缩因子: {compression_factor}")
    
    success_count = 0
    error_count = 0
    
    for path in Bar('Processing').iter(audio_paths):
        print(' ', end=f'[{path}]', flush=True)
        
        try:
            # 编码音频文件到潜在空间
            latent = preprocess_audio_to_latent(
                path,
                processor=processor,
                target_length=target_length,
            )
            
            # 保存为Pickle文件
            file_name = os.path.basename(path)
            file_name_without_ext = os.path.splitext(file_name)[0]
            save_path = os.path.join(save_dir, f'{file_name_without_ext}.pickle')
            
            with open(save_path, 'wb') as f:
                pickle.dump(latent, f)
            
            success_count += 1
            print(f' ✓ (潜在空间形状: {latent.shape})')
            
        except KeyboardInterrupt:
            print(' 中断')
            return
        except Exception as e:
            error_count += 1
            print(f' ✗ 错误: {str(e)}')
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n处理完成: 成功 {success_count} 个, 失败 {error_count} 个")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python latent_preprocess.py <audio_root> <save_dir> [target_length] [latent_dim] [compression_factor] [encoder_checkpoint]")
        print("示例: python latent_preprocess.py ./audio_data ./audio_latent_pickle 1024 32 4 saved/scheme3/vae/vae_epoch_100.pt")
        sys.exit(1)
    
    audio_root = sys.argv[1]
    save_dir = sys.argv[2]
    target_length = int(sys.argv[3]) if len(sys.argv) > 3 else None
    latent_dim = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    compression_factor = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    encoder_checkpoint = sys.argv[6] if len(sys.argv) > 6 else None
    
    preprocess_audio_files_to_latent(
        audio_root,
        save_dir,
        target_length,
        latent_dim,
        compression_factor,
        encoder_checkpoint,
    )

