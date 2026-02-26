"""
方案三生成质量评估脚本

参考 Tango 项目在 AudioCaps 上的客观评估思路：
- Tango 使用 CLAP / AudioLDM-Eval 等特征，在潜在/嵌入空间中计算 FAD、KL、IS、CLAP 等指标。
- 本项目当前以 VAE 的 Mel 频谱潜在空间为核心表示，因此这里实现一组
  “基于 Mel 频谱和能量包络”的客观指标，作为音乐生成质量的度量。

指标包括（按 Tango 的“特征空间度量 + 感知相关度量”的思路设计）：
- mel_l1:      Mel 频谱 L1 距离（内容接近程度）
- mel_l2:      Mel 频谱 L2 距离（均方误差）
- mel_sc:      Mel 频谱谱收敛度（Spectral Convergence）
- mel_log:     Log-Mel 频谱距离（类似 Log-STFT / Mel 损失）
- energy_corr: 能量包络相关系数（整体动态/节奏起伏的一致性）
此外（数据集级别）：
- fd_mel:      在 Mel 帧特征空间上的 Fréchet Distance（数学形式等价于 FAD/FD）
- kl_mel:      在 Mel 帧特征空间上，参考分布与生成分布拟合高斯后的 KL(P||Q)

用法示例：

python evaluate_scheme3.py \
  --model_path saved/scheme3/scheme3_epoch_100.pt \
  --midi_dir SMD-piano_v2/midi \
  --ref_audio_dir SMD-piano_v2/midi_wav_22050_mono \
  --max_eval 50 \
  --output_json logs/scheme3_eval.json
"""

import os
import sys
import json
import glob
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio.scheme3 import LatentSpaceConditionalModel  # type: ignore
from audio.stft import TacotronSTFT  # type: ignore
from audio.tools import get_mel_from_wav, read_wav_file  # type: ignore
from audio.metrics import (
    mel_l1_distance,
    mel_l2_distance,
    mel_spectral_convergence,
    mel_log_distance,
    energy_envelope_correlation,
    summarize_metrics,
    frechet_distance_from_features,
    gaussian_kl_from_features,
)
from midi.preprocess import preprocess_midi  # type: ignore
from midi.custom.config import config  # type: ignore


def _load_model(
    model_path: str,
    device: torch.device,
    n_mel_channels: int = 80,
    latent_dim: int = 32,
    compression_factor: int = 4,
    embedding_dim: int = 256,
    num_layers: int = 6,
    max_seq: int = 2048,
    dropout: float = 0.2,
    num_timesteps: int = 1000,
    schedule_type: str = "cosine",
    base_channels: int = 64,
    num_res_blocks: int = 2,
) -> LatentSpaceConditionalModel:
    """
    加载方案三扩散模型（与 train_scheme3.py / generate_scheme3.py 保持一致）。
    """
    # MIDI 词表大小从 config 中获取，保持一致性
    if "event_dim" not in config.dict:
        config.event_dim = 388
    if "pad_token" not in config.dict:
        config._set_vocab_params()
    midi_vocab_size = config.vocab_size

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

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

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _collect_midi_ref_pairs(midi_dir: str, ref_audio_dir: str, max_eval: int = None) -> List[Tuple[str, str]]:
    """
    根据文件名匹配 MIDI 和参考音频：
    - 假设 MIDI 和参考 WAV 具有相同的 basename（不含扩展名）。
    """
    midi_files = glob.glob(os.path.join(midi_dir, "**", "*.mid"), recursive=True)
    midi_files.extend(glob.glob(os.path.join(midi_dir, "**", "*.midi"), recursive=True))
    midi_files = sorted(midi_files)

    pairs: List[Tuple[str, str]] = []
    for midi_path in midi_files:
        base = os.path.splitext(os.path.basename(midi_path))[0]
        # 在参考目录下查找同名 wav（允许不同采样率版本并优先简单匹配）
        candidate = os.path.join(ref_audio_dir, base + ".wav")
        if os.path.exists(candidate):
            pairs.append((midi_path, candidate))
        if max_eval is not None and len(pairs) >= max_eval:
            break
    return pairs


def evaluate_pair(
    model: LatentSpaceConditionalModel,
    device: torch.device,
    midi_path: str,
    ref_wav_path: str,
    stft_processor: TacotronSTFT,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    对单条 (MIDI, 参考WAV) 配对进行生成与客观指标评估。
    """
    # 1) 预处理 MIDI
    midi_tokens = preprocess_midi(midi_path)
    max_seq = model.max_seq if hasattr(model, "max_seq") else 2048
    if len(midi_tokens) > max_seq:
        midi_tokens = midi_tokens[:max_seq]
    midi_tokens_tensor = torch.LongTensor([midi_tokens]).to(device)

    # 2) 生成 Mel（直接在 VAE 解码后的 Mel 空间评估）
    with torch.no_grad():
        _, mel_gen = model.generate(
            midi_tokens_tensor,
            shape=None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
    # (B, n_mel_channels, T) -> (n_mel_channels, T)
    mel_gen_np = mel_gen[0].detach().cpu().numpy()

    # 3) 计算参考音频的 Mel 与能量（与训练/预处理保持一致）
    #    这里复用 audio.tools.read_wav_file + get_mel_from_wav
    #    waveform 以 16kHz 单声道读取，STFT 配置与模型默认一致。
    segment_length = None  # 评估时不过度裁剪，后续按时间维度对齐
    waveform = read_wav_file(ref_wav_path, segment_length=segment_length)
    waveform = waveform[0, ...]  # (T,)
    waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0)  # (1, T)
    with torch.no_grad():
        melspec_ref, _, energy_ref = stft_processor.mel_spectrogram(waveform_tensor)
    # melspec_ref: (1, n_mel_channels, T) -> (n_mel_channels, T)
    mel_ref_np = melspec_ref[0].detach().cpu().numpy().astype(np.float32)
    energy_ref_np = energy_ref[0].detach().cpu().numpy().astype(np.float32)  # (T,)

    # 4) 为能量包络构造一份“生成端”的能量近似：这里简单使用 Mel 能量求和
    energy_gen_np = np.sum(np.abs(mel_gen_np), axis=0)

    # 5) 计算各项指标
    metrics: Dict[str, float] = {}
    metrics["mel_l1"] = mel_l1_distance(mel_gen_np, mel_ref_np)
    metrics["mel_l2"] = mel_l2_distance(mel_gen_np, mel_ref_np)
    metrics["mel_sc"] = mel_spectral_convergence(mel_gen_np, mel_ref_np)
    metrics["mel_log"] = mel_log_distance(mel_gen_np, mel_ref_np)
    metrics["energy_corr"] = energy_envelope_correlation(energy_gen_np, energy_ref_np)

    # 6) 为 FAD/FD/KL 提供特征：
    #    这里使用每一帧的 Mel 向量作为特征样本，即 (T, n_mel_channels)。
    #    注意：T_gen 与 T_ref 可能不同，这里分别保留，聚合到整个数据集后
    #    通过协方差估计得到全局分布。
    feats_gen = mel_gen_np.T.astype(np.float32)  # (T_gen, C)
    feats_ref = mel_ref_np.T.astype(np.float32)  # (T_ref, C)

    return metrics, feats_gen, feats_ref


def main():
    parser = argparse.ArgumentParser(description="Evaluate Scheme 3 generation quality (Tango-style metrics).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to scheme3 diffusion checkpoint (.pt)")
    parser.add_argument("--midi_dir", type=str, required=True, help="Directory containing MIDI files")
    parser.add_argument("--ref_audio_dir", type=str, required=True, help="Directory containing reference WAV (same basename as MIDI)")
    parser.add_argument("--max_eval", type=int, default=None, help="Max number of (MIDI, ref) pairs to evaluate")
    parser.add_argument("--output_json", type=str, default="logs/scheme3_eval.json", help="Output JSON path for metrics")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Diffusion inference steps")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Classifier-free guidance scale")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.model_path} ...")
    model = _load_model(args.model_path, device)
    n_mel_channels = 80
    stft_processor = TacotronSTFT(
        filter_length=1024,
        hop_length=160,
        win_length=1024,
        n_mel_channels=n_mel_channels,
        sampling_rate=16000,
        mel_fmin=0.0,
        mel_fmax=8000.0,
    )

    pairs = _collect_midi_ref_pairs(args.midi_dir, args.ref_audio_dir, args.max_eval)
    if not pairs:
        print("No (MIDI, ref WAV) pairs found. Check --midi_dir and --ref_audio_dir (basename must match).")
        return
    print(f"Found {len(pairs)} (MIDI, ref) pairs. Evaluating ...")

    all_metrics: List[Dict[str, float]] = []
    feats_gen_list: List[np.ndarray] = []
    feats_ref_list: List[np.ndarray] = []

    for i, (midi_path, ref_wav_path) in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {os.path.basename(midi_path)} ...")
        try:
            metrics, feats_gen, feats_ref = evaluate_pair(
                model, device, midi_path, ref_wav_path, stft_processor,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
            all_metrics.append(metrics)
            feats_gen_list.append(feats_gen)
            feats_ref_list.append(feats_ref)
        except Exception as e:
            print(f"    Error: {e}")
            continue

    if not all_metrics:
        print("No samples evaluated successfully.")
        return

    metric_list = {k: [m[k] for m in all_metrics] for k in all_metrics[0]}
    summary = summarize_metrics(metric_list)

    feats_gen_all = np.vstack(feats_gen_list)
    feats_ref_all = np.vstack(feats_ref_list)
    summary["fd_mel"] = frechet_distance_from_features(feats_gen_all, feats_ref_all)
    summary["kl_mel"] = gaussian_kl_from_features(feats_ref_all, feats_gen_all)

    out_path = args.output_json
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {out_path}")
    print("Summary:", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
