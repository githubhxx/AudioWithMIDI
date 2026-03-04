"""
多尺度音频切片 latent 预计算与索引构建。

核心思路：
1) 离线阶段：基于 MIDI 的 bar/downbeat 网格枚举窗口，切音频并编码为 VAE latent。
2) 在线训练：只做 MIDI 切分与编码，并读取对应窗口 latent，避免重复音频前向。
"""

from __future__ import annotations

import glob
import json
import os
import pickle
from typing import Dict, List

import pretty_midi
import torch
import torchaudio
from tqdm import tqdm

from audio.latent_encoder import LatentAudioEncoder
from audio.stft import TacotronSTFT
from .segmenter import TimeBasedMultiScaleSegmenter


def _collect_pairs(audio_dir: str, midi_dir: str) -> List[Dict[str, str]]:
    def _group_by_basename(paths: List[str]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for p in paths:
            out.setdefault(os.path.splitext(os.path.basename(p))[0], []).append(p)
        return out

    audio_files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)
    midi_files = glob.glob(os.path.join(midi_dir, "**", "*.mid"), recursive=True)
    midi_files += glob.glob(os.path.join(midi_dir, "**", "*.midi"), recursive=True)

    audio_map = _group_by_basename(audio_files)
    midi_map = _group_by_basename(midi_files)

    pairs = []
    for base in sorted(set(audio_map.keys()) & set(midi_map.keys())):
        # basename 重名会导致错误配对，直接跳过以避免污染缓存索引
        if len(audio_map[base]) != 1 or len(midi_map[base]) != 1:
            continue
        pairs.append({"basename": base, "audio_path": audio_map[base][0], "midi_path": midi_map[base][0]})
    return pairs


def _load_vae_encoder_checkpoint(encoder: LatentAudioEncoder, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    # 兼容完整模型 checkpoint（包含 vae_encoder. 前缀）
    if any(k.startswith("vae_encoder.") for k in state.keys()):
        state = {k.replace("vae_encoder.", ""): v for k, v in state.items() if k.startswith("vae_encoder.")}

    encoder.load_state_dict(state, strict=False)


def _estimate_global_position(downbeats: List[float], beats: List[float], start_sec: float, end_sec: float):
    def _search_leq_index(ts: List[float], t: float) -> int:
        idx = 0
        for i, v in enumerate(ts):
            if v <= t:
                idx = i
            else:
                break
        return idx

    bar_start = _search_leq_index(downbeats, start_sec)
    bar_end = max(bar_start, _search_leq_index(downbeats, end_sec) - 1)
    beat_start = _search_leq_index(beats, start_sec)
    beat_end = max(beat_start, _search_leq_index(beats, end_sec) - 1)

    return {
        "bar_start_index": int(max(0, bar_start)),
        "bar_end_index": int(max(0, bar_end)),
        "beat_start_index": int(max(0, beat_start)),
        "beat_end_index": int(max(0, beat_end)),
    }


def _iter_windows(downbeats: List[float], scale_bars: Dict[str, int]):
    total_bars = max(1, len(downbeats) - 1)
    for scale, bars in scale_bars.items():
        bars = max(1, int(bars))
        if bars > total_bars:
            # 退化为整段
            yield scale, 0, total_bars
            continue
        for bar_start in range(0, total_bars - bars + 1):
            yield scale, bar_start, bar_start + bars


def build_multiscale_latent_cache(
    audio_dir: str,
    midi_dir: str,
    output_dir: str,
    vae_checkpoint: str,
    latent_dim: int = 32,
    compression_factor: int = 4,
    target_sample_rate: int = 16000,
    scale_bars: Dict[str, int] | None = None,
):
    os.makedirs(output_dir, exist_ok=True)
    latent_dir = os.path.join(output_dir, "latents")
    os.makedirs(latent_dir, exist_ok=True)

    scale_bars = scale_bars or {"high": 1, "mid": 4, "low": 8}
    pairs = _collect_pairs(audio_dir, midi_dir)

    segmenter = TimeBasedMultiScaleSegmenter(target_sample_rate=target_sample_rate, scale_bars=scale_bars, seed=1234)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = LatentAudioEncoder(latent_dim=latent_dim, compression_factor=compression_factor).to(device)
    _load_vae_encoder_checkpoint(encoder, vae_checkpoint)
    encoder.eval()

    stft = TacotronSTFT(
        filter_length=1024,
        hop_length=160,
        win_length=1024,
        n_mel_channels=80,
        sampling_rate=target_sample_rate,
        mel_fmin=0.0,
        mel_fmax=8000.0,
    ).to(device)

    records: List[Dict] = []

    for pair in tqdm(pairs, desc="Precompute multiscale latent"):
        base = pair["basename"]
        audio_path = pair["audio_path"]
        midi_path = pair["midi_path"]

        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.dim() == 2 and waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != target_sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sample_rate)
                sr = target_sample_rate

            midi_obj = pretty_midi.PrettyMIDI(midi_path)
            downbeats, beats = segmenter._build_music_grids(midi_obj, total_audio_duration_sec=waveform.size(-1) / sr)

            for scale, b0, b1 in _iter_windows(downbeats, scale_bars):
                start_sec = float(downbeats[b0])
                end_sec = float(downbeats[b1])
                if end_sec <= start_sec:
                    continue

                audio_seg = segmenter._slice_audio_by_seconds(waveform, sr, start_sec, end_sec)

                y = torch.clamp(audio_seg.to(device), -1.0, 1.0)
                mel, _, _ = stft.mel_spectrogram(y)
                with torch.no_grad():
                    latent = encoder(mel).squeeze(0).cpu().numpy()  # (latent_dim, T)

                latent_name = f"{base}__{scale}__b{b0}-{b1}.pickle"
                latent_path = os.path.join(latent_dir, latent_name)
                with open(latent_path, "wb") as f:
                    pickle.dump(latent, f)

                gp = _estimate_global_position(downbeats, beats, start_sec, end_sec)
                records.append(
                    {
                        "basename": base,
                        "audio_path": audio_path,
                        "midi_path": midi_path,
                        "scale": scale,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "duration_sec": end_sec - start_sec,
                        "bar_start_index": gp["bar_start_index"],
                        "bar_end_index": gp["bar_end_index"],
                        "beat_start_index": gp["beat_start_index"],
                        "beat_end_index": gp["beat_end_index"],
                        "latent_path": latent_path,
                    }
                )
        except Exception as e:
            print(f"[WARN] skip {base}: {e}")
            continue

    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "audio_dir": audio_dir,
                    "midi_dir": midi_dir,
                    "vae_checkpoint": vae_checkpoint,
                    "latent_dim": latent_dim,
                    "compression_factor": compression_factor,
                    "target_sample_rate": target_sample_rate,
                    "scale_bars": scale_bars,
                    "num_records": len(records),
                },
                "records": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return index_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build multiscale latent cache")
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--midi_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vae_checkpoint", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--compression_factor", type=int, default=4)
    parser.add_argument("--target_sample_rate", type=int, default=16000)
    parser.add_argument("--scale_bars_high", type=int, default=1)
    parser.add_argument("--scale_bars_mid", type=int, default=4)
    parser.add_argument("--scale_bars_low", type=int, default=8)
    args = parser.parse_args()

    scale_bars = {"high": args.scale_bars_high, "mid": args.scale_bars_mid, "low": args.scale_bars_low}
    p = build_multiscale_latent_cache(
        audio_dir=args.audio_dir,
        midi_dir=args.midi_dir,
        output_dir=args.output_dir,
        vae_checkpoint=args.vae_checkpoint,
        latent_dim=args.latent_dim,
        compression_factor=args.compression_factor,
        target_sample_rate=args.target_sample_rate,
        scale_bars=scale_bars,
    )
    print(f"cache index saved: {p}")
