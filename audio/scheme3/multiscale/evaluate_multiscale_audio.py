"""
多尺度模型音频客观评估脚本（可直接运行版）。

功能：
1) 读取 multiscale checkpoint（默认 multiscale_best.pt）
2) 对 val/test MIDI 批量生成 wav
3) 与参考 wav 计算客观指标：mel_l1/mel_l2/mel_sc/mel_log + fd_mel/kl_mel
4) 导出 summary JSON/CSV 与 per-sample 明细 CSV

示例：
python -m audio.scheme3.multiscale.evaluate_multiscale_audio \
  --cache_index_path /path/to/index.json \
  --save_dir /path/to/multiscale_train \
  --split both \
  --max_eval 100 \
  --output_dir /path/to/multiscale_train/audio_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.io.wavfile import write as wav_write

from audio.metrics import (
    frechet_distance_from_features,
    gaussian_kl_from_features,
)
from audio.stft import TacotronSTFT
from audio.tools import read_wav_file
from midi.preprocess import preprocess_midi

from .cached_dataset import CachedMultiscaleLatentDataset
from .mel_config import DEFAULT_MEL_CONFIG
from .model import MultiScaleLatentSpaceConditionalModel


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate multiscale model with audio objective metrics")
    p.add_argument("--cache_index_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="", help="训练目录，默认读取 multiscale_best.pt")
    p.add_argument("--checkpoint_path", type=str, default="", help="显式 checkpoint 路径，优先级高于 save_dir")

    p.add_argument("--split", type=str, default="val", choices=["val", "test", "both"])
    p.add_argument("--max_eval", type=int, default=100, help="每个 split 最多评估多少首曲子（按 basename 去重）")

    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=1.0)

    p.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu/mps")
    p.add_argument("--output_dir", type=str, default="", help="默认 save_dir/audio_eval")
    p.add_argument("--save_generated_wav", action="store_true", help="保存生成 wav")

    p.add_argument("--n_fft", type=int, default=DEFAULT_MEL_CONFIG.n_fft)
    p.add_argument("--hop_length", type=int, default=DEFAULT_MEL_CONFIG.hop_length)
    p.add_argument("--win_length", type=int, default=DEFAULT_MEL_CONFIG.win_length)
    p.add_argument("--n_mels", type=int, default=DEFAULT_MEL_CONFIG.n_mels)
    p.add_argument("--sampling_rate", type=int, default=DEFAULT_MEL_CONFIG.sampling_rate)
    p.add_argument("--mel_fmin", type=float, default=DEFAULT_MEL_CONFIG.fmin)
    p.add_argument("--mel_fmax", type=float, default=DEFAULT_MEL_CONFIG.fmax)
    p.add_argument("--center", type=int, default=int(DEFAULT_MEL_CONFIG.center))

    # fallback 模型参数（当 checkpoint 不含 args 时）
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--midi_vocab_size", type=int, default=390)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max_seq", type=int, default=2048)

    return p.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to CPU")
        return torch.device("cpu")

    if device_str == "mps":
        mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
        if not mps_ok:
            print("[WARN] MPS not available, fallback to CPU")
            return torch.device("cpu")

    return torch.device(device_str)


def resolve_checkpoint(args) -> str:
    if args.checkpoint_path:
        ckpt = args.checkpoint_path
    else:
        if not args.save_dir:
            raise ValueError("请提供 --checkpoint_path 或 --save_dir")
        ckpt = os.path.join(args.save_dir, "multiscale_best.pt")

    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")
    return ckpt


def apply_mel_args_from_ckpt(cli_args, ckpt: Dict):
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    for k in ["n_fft", "hop_length", "win_length", "n_mels", "sampling_rate", "mel_fmin", "mel_fmax", "center"]:
        if k in ckpt_args:
            setattr(cli_args, k, ckpt_args[k])


def build_model_from_ckpt(ckpt: Dict, cli_args, device: torch.device) -> MultiScaleLatentSpaceConditionalModel:
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    model = MultiScaleLatentSpaceConditionalModel(
        latent_dim=int(ckpt_args.get("latent_dim", cli_args.latent_dim)),
        embedding_dim=int(ckpt_args.get("embedding_dim", cli_args.embedding_dim)),
        midi_vocab_size=int(ckpt_args.get("midi_vocab_size", cli_args.midi_vocab_size)),
        num_layers=int(ckpt_args.get("num_layers", cli_args.num_layers)),
        dropout=float(ckpt_args.get("dropout", cli_args.dropout)),
        max_seq=int(ckpt_args.get("max_seq", cli_args.max_seq)),
        use_cross_attention=True,
        cross_attention_heads=8,
    ).to(device)

    for p in model.vae_encoder.parameters():
        p.requires_grad = False
    for p in model.vae_decoder.parameters():
        p.requires_grad = False
    model.vae_encoder.eval()
    model.vae_decoder.eval()

    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def mel_to_audio_griffin_lim(
    mel_spec: np.ndarray,
    *,
    sampling_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    mel_fmin: float,
    mel_fmax: float,
    n_iter: int = 60,
) -> np.ndarray:
    try:
        import librosa

        linear_spec = librosa.feature.inverse.mel_to_stft(
            mel_spec,
            sr=sampling_rate,
            n_fft=n_fft,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        audio = librosa.griffinlim(
            linear_spec,
            n_iter=n_iter,
            hop_length=hop_length,
            win_length=win_length,
        )
        return audio.astype(np.float32)
    except Exception as e:
        print(f"[WARN] Griffin-Lim failed: {e}; fallback to noise")
        t = mel_spec.shape[1]
        return (np.random.randn(t * hop_length) * 0.005).astype(np.float32)


def pick_unique_pairs(cache_index_path: str, split: str, max_eval: int) -> List[Tuple[str, str, str]]:
    ds = CachedMultiscaleLatentDataset(cache_index_path=cache_index_path, split=split)
    seen = set()
    pairs: List[Tuple[str, str, str]] = []

    for r in ds.records:
        b = r["basename"]
        if b in seen:
            continue
        midi_path = r["midi_path"]
        ref_wav = r["audio_path"]
        if os.path.exists(midi_path) and os.path.exists(ref_wav):
            pairs.append((b, midi_path, ref_wav))
            seen.add(b)
        if max_eval > 0 and len(pairs) >= max_eval:
            break

    return pairs


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def align_mel_with_mask(
    mel_gen: np.ndarray,
    mel_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    统一对齐入口：按同长 pad，并返回有效帧 mask（True=双方都有效，False=padding）。
    输入/输出 mel 形状均为 (n_mels, T)。
    """
    if mel_gen.ndim != 2 or mel_ref.ndim != 2:
        raise ValueError(f"mel must be 2D (n_mels, T), got {mel_gen.shape} and {mel_ref.shape}")
    if mel_gen.shape[0] != mel_ref.shape[0]:
        raise ValueError(f"n_mels mismatch: {mel_gen.shape[0]} vs {mel_ref.shape[0]}")

    tg = int(mel_gen.shape[-1])
    tr = int(mel_ref.shape[-1])
    tmax = max(tg, tr)

    gen_pad = np.zeros((mel_gen.shape[0], tmax), dtype=np.float32)
    ref_pad = np.zeros((mel_ref.shape[0], tmax), dtype=np.float32)

    gen_pad[:, :tg] = mel_gen.astype(np.float32, copy=False)
    ref_pad[:, :tr] = mel_ref.astype(np.float32, copy=False)

    valid_mask = np.zeros((tmax,), dtype=bool)
    valid_mask[: min(tg, tr)] = True
    return gen_pad, ref_pad, valid_mask


def masked_feature_frames(mel: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """将 (n_mels, T) 按 mask 筛到 (T_valid, n_mels) 用于分布特征统计。"""
    if mel.ndim != 2:
        raise ValueError(f"mel must be 2D (n_mels,T), got {mel.shape}")
    if valid_mask.ndim != 1 or valid_mask.shape[0] != mel.shape[1]:
        raise ValueError(f"mask shape mismatch, got {valid_mask.shape} for mel {mel.shape}")
    if not np.any(valid_mask):
        return np.zeros((1, mel.shape[0]), dtype=np.float32)
    return mel[:, valid_mask].T.astype(np.float32)


def masked_mel_metrics(mel_gen: np.ndarray, mel_ref: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
    """在有效帧上计算 mel 指标。"""
    if not np.any(valid_mask):
        return {"mel_l1": float("nan"), "mel_l2": float("nan"), "mel_sc": float("nan"), "mel_log": float("nan")}

    mg = mel_gen[:, valid_mask]
    mr = mel_ref[:, valid_mask]

    eps = 1e-8
    diff = mr - mg
    mel_l1 = float(np.mean(np.abs(diff)))
    mel_l2 = float(np.mean(diff ** 2))
    mel_sc = float(np.linalg.norm(diff) / (np.linalg.norm(mr) + eps))

    log_eps = 1e-6
    log_gen = np.log(np.clip(np.abs(mg), a_min=log_eps, a_max=None))
    log_ref = np.log(np.clip(np.abs(mr), a_min=log_eps, a_max=None))
    mel_log = float(np.mean(np.abs(log_gen - log_ref)))

    return {
        "mel_l1": mel_l1,
        "mel_l2": mel_l2,
        "mel_sc": mel_sc,
        "mel_log": mel_log,
    }


def evaluate_split(
    split: str,
    model: MultiScaleLatentSpaceConditionalModel,
    pairs: List[Tuple[str, str, str]],
    args,
    device: torch.device,
    stft: TacotronSTFT,
    output_dir: str,
):
    per_sample: List[Dict] = []
    feats_gen: List[np.ndarray] = []
    feats_ref: List[np.ndarray] = []

    wav_dir = os.path.join(output_dir, "generated", split)
    if args.save_generated_wav:
        os.makedirs(wav_dir, exist_ok=True)

    for i, (basename, midi_path, ref_wav_path) in enumerate(pairs):
        try:
            midi_tokens = preprocess_midi(midi_path)
            max_seq = model.max_seq if hasattr(model, "max_seq") else args.max_seq
            if len(midi_tokens) > max_seq:
                midi_tokens = midi_tokens[:max_seq]

            midi_tensor = torch.LongTensor([midi_tokens]).to(device)
            with torch.no_grad():
                _, mel_gen = model.generate(
                    midi_tensor,
                    shape=None,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                )

            mel_gen_np = mel_gen[0].detach().cpu().numpy().astype(np.float32)

            ref_wave = read_wav_file(ref_wav_path, segment_length=None)
            ref_wave = torch.FloatTensor(ref_wave[0]).unsqueeze(0)
            with torch.no_grad():
                mel_ref_t, _, _ = stft.mel_spectrogram(ref_wave)
            mel_ref_np = mel_ref_t[0].detach().cpu().numpy().astype(np.float32)

            mel_gen_pad, mel_ref_pad, valid_mask = align_mel_with_mask(mel_gen_np, mel_ref_np)

            metrics = masked_mel_metrics(mel_gen_pad, mel_ref_pad, valid_mask)

            feats_gen.append(masked_feature_frames(mel_gen_pad, valid_mask))
            feats_ref.append(masked_feature_frames(mel_ref_pad, valid_mask))

            wav_path = ""
            if args.save_generated_wav:
                audio = mel_to_audio_griffin_lim(
                    mel_gen_np,
                    sampling_rate=args.sampling_rate,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    win_length=args.win_length,
                    mel_fmin=args.mel_fmin,
                    mel_fmax=args.mel_fmax,
                    n_iter=60,
                )
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
                audio = np.clip(audio, -1.0, 1.0)
                wav_path = os.path.join(wav_dir, f"{basename}.wav")
                wav_write(wav_path, args.sampling_rate, (audio * 32767).astype(np.int16))

            row = {
                "split": split,
                "index": i,
                "basename": basename,
                "midi_path": midi_path,
                "ref_wav_path": ref_wav_path,
                "gen_wav_path": wav_path,
                **{k: float(v) for k, v in metrics.items()},
            }
            per_sample.append(row)

        except Exception as e:
            debug_ctx = {
                "midi_tokens_len": int(len(midi_tokens)) if "midi_tokens" in locals() else None,
                "mel_gen_shape": tuple(mel_gen_np.shape) if "mel_gen_np" in locals() else None,
                "mel_ref_shape": tuple(mel_ref_np.shape) if "mel_ref_np" in locals() else None,
                "aligned_gen_shape": tuple(mel_gen_pad.shape) if "mel_gen_pad" in locals() else None,
                "aligned_ref_shape": tuple(mel_ref_pad.shape) if "mel_ref_pad" in locals() else None,
                "valid_frames": int(np.sum(valid_mask)) if "valid_mask" in locals() else None,
                "n_fft": int(args.n_fft),
                "hop_length": int(args.hop_length),
                "win_length": int(args.win_length),
                "n_mels": int(args.n_mels),
                "sampling_rate": int(args.sampling_rate),
                "mel_fmin": float(args.mel_fmin),
                "mel_fmax": float(args.mel_fmax),
                "center": int(args.center),
            }
            err_msg = f"{type(e).__name__}: {e}; debug={json.dumps(debug_ctx, ensure_ascii=False)}"
            print(f"[ERROR] split={split} index={i} basename={basename} -> {err_msg}")
            traceback.print_exc()
            per_sample.append(
                {
                    "split": split,
                    "index": i,
                    "basename": basename,
                    "midi_path": midi_path,
                    "ref_wav_path": ref_wav_path,
                    "gen_wav_path": "",
                    "mel_l1": "",
                    "mel_l2": "",
                    "mel_sc": "",
                    "mel_log": "",
                    "error": err_msg,
                }
            )

    valid_rows = [r for r in per_sample if isinstance(r.get("mel_l1", None), float)]
    summary = {
        "split": split,
        "num_total": len(per_sample),
        "num_success": len(valid_rows),
        "num_failed": len(per_sample) - len(valid_rows),
    }

    if valid_rows:
        for k in ["mel_l1", "mel_l2", "mel_sc", "mel_log"]:
            summary[k] = float(np.mean([float(r[k]) for r in valid_rows]))

    if feats_gen and feats_ref:
        g = np.vstack(feats_gen)
        r = np.vstack(feats_ref)
        summary["fd_mel"] = float(frechet_distance_from_features(g, r))
        summary["kl_mel"] = float(gaussian_kl_from_features(r, g))
    else:
        summary["fd_mel"] = float("nan")
        summary["kl_mel"] = float("nan")

    return summary, per_sample


def main():
    args = parse_args()
    device = resolve_device(args.device)
    ckpt_path = resolve_checkpoint(args)

    output_dir = args.output_dir or (os.path.join(args.save_dir, "audio_eval") if args.save_dir else "logs/multiscale_audio_eval")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    apply_mel_args_from_ckpt(args, ckpt)
    print(
        "Eval mel config:",
        json.dumps(
            {
                "n_fft": int(args.n_fft),
                "hop_length": int(args.hop_length),
                "win_length": int(args.win_length),
                "n_mels": int(args.n_mels),
                "sampling_rate": int(args.sampling_rate),
                "mel_fmin": float(args.mel_fmin),
                "mel_fmax": float(args.mel_fmax),
                "center": int(args.center),
            },
            ensure_ascii=False,
        ),
    )
    model = build_model_from_ckpt(ckpt, args, device)

    stft = TacotronSTFT(
        filter_length=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mel_channels=args.n_mels,
        sampling_rate=args.sampling_rate,
        mel_fmin=args.mel_fmin,
        mel_fmax=args.mel_fmax,
    )

    splits = ["val", "test"] if args.split == "both" else [args.split]

    split_summaries = []
    all_rows = []
    for sp in splits:
        pairs = pick_unique_pairs(args.cache_index_path, sp, args.max_eval)
        print(f"Evaluating split={sp}, pairs={len(pairs)}")

        summary, rows = evaluate_split(
            split=sp,
            model=model,
            pairs=pairs,
            args=args,
            device=device,
            stft=stft,
            output_dir=output_dir,
        )
        split_summaries.append(summary)
        all_rows.extend(rows)

    summary_json = {
        "checkpoint_path": ckpt_path,
        "device": str(device),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "splits": split_summaries,
    }

    with open(os.path.join(output_dir, "audio_metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    _write_csv(
        os.path.join(output_dir, "audio_metrics_summary.csv"),
        split_summaries,
        [
            "split",
            "num_total",
            "num_success",
            "num_failed",
            "mel_l1",
            "mel_l2",
            "mel_sc",
            "mel_log",
            "fd_mel",
            "kl_mel",
        ],
    )

    _write_csv(
        os.path.join(output_dir, "audio_metrics_per_sample.csv"),
        all_rows,
        [
            "split",
            "index",
            "basename",
            "midi_path",
            "ref_wav_path",
            "gen_wav_path",
            "mel_l1",
            "mel_l2",
            "mel_sc",
            "mel_log",
            "error",
        ],
    )

    print(f"Saved: {os.path.join(output_dir, 'audio_metrics_summary.json')}")
    print(f"Saved: {os.path.join(output_dir, 'audio_metrics_summary.csv')}")
    print(f"Saved: {os.path.join(output_dir, 'audio_metrics_per_sample.csv')}")


if __name__ == "__main__":
    main()
