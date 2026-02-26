import torch
import numpy as np
import torchaudio


def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    log_magnitudes_stft = (
        torch.squeeze(log_magnitudes_stft, 0).numpy().astype(np.float32)
    )
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return melspec, log_magnitudes_stft, energy


def _pad_spec(fbank, target_length=1024):
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    if fbank.size(-1) % 2 != 0:
        fbank = fbank[..., :-1]

    return fbank


def pad_wav(waveform, segment_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length
    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    elif waveform_length < segment_length:
        temp_wav = np.zeros((1, segment_length))
        temp_wav[:, :waveform_length] = waveform
    return temp_wav

def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5


def read_wav_file(filename, segment_length):
    # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
    waveform, sr = torchaudio.load(filename)  # Faster!!!
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    waveform = waveform.numpy()[0, ...]

    # Normalize to approximately [-0.5, 0.5] once, similar in spirit to
    # AudioLDM’s waveform normalization. Downstream STFT assumes inputs
    # are within [-1, 1], so this keeps us well inside that range without
    # double-scaling.
    waveform = normalize_wav(waveform)
    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, segment_length)

    return waveform


def wav_to_fbank(filename, target_length=1024, fn_STFT=None):
    """
    将 WAV 文件转换为 Mel 滤波器组特征（fbank）
    
    参考 Tango 原始实现：主要返回 mel 频谱特征，用于 VAE 编码。
    为了向后兼容 AudioEncoder，仍然返回 (fbank, log_magnitudes_stft, energy)。
    
    Args:
        filename: WAV 文件路径
        target_length: 目标长度（帧数），如果为 None 则不进行长度限制
        fn_STFT: STFT 处理器实例
        
    Returns:
        fbank: Mel 滤波器组特征 (n_frames, n_mel_channels)
        log_magnitudes_stft: 对数幅度谱 (n_frames, n_freq_bins)
        energy: 能量特征 (n_frames,)
    """
    assert fn_STFT is not None

    # hop size is 160 at 16kHz → 10 ms per frame
    # 当 target_length 为 None 时，传递 None 给 read_wav_file（不进行长度限制）
    segment_length = target_length * 160 if target_length is not None else None
    waveform = read_wav_file(filename, segment_length)

    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    # 计算 Mel 频谱、对数幅度谱和能量
    # 注意：对于 VAE 编码，主要使用 fbank；energy 主要用于 AudioEncoder 的事件提取
    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)

    fbank = torch.FloatTensor(fbank.T)  # (n_frames, n_mel_channels)
    log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

    # 只有当 target_length 不为 None 时才进行填充
    if target_length is not None:
        fbank = _pad_spec(fbank, target_length)
        log_magnitudes_stft = _pad_spec(log_magnitudes_stft, target_length)

    # Energy 处理：主要用于 AudioEncoder 的事件提取
    # 如果项目主要使用 VAE 编码，可以忽略此返回值
    energy = torch.FloatTensor(energy)
    if energy.dim() == 1:
        energy = energy.unsqueeze(-1)
    if target_length is not None:
        energy = _pad_spec(energy, target_length).squeeze(-1)
    else:
        energy = energy.squeeze(-1)

    return fbank, log_magnitudes_stft, energy
