"""
多尺度模型客观评估脚本（可直接运行版）。

功能：
1) 在 val split 上评估 checkpoint 的 masked MSE（与训练损失一致）。
2) 输出整体指标 + 分尺度指标（high/mid/low）。
3) 导出 CSV（summary / per-scale / best-vs-final 对比）。
4) 打印 best/final 对比表（Markdown）。

示例：
python -m audio.scheme3.multiscale.evaluate_multiscale \
  --cache_index_path /path/to/index.json \
  --save_dir /path/to/multiscale_train \
  --output_dir /path/to/multiscale_train/eval \
  --batch_size 8 --num_workers 4 --max_batches 100
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from functools import partial
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .cached_dataset import CachedMultiscaleLatentDataset, collate_cached_multiscale_batch
from .model import MultiScaleLatentSpaceConditionalModel
from .structural_features import StructuralFeatureBuilder
from .train_utils import resize_latent_per_sample, masked_mse_loss

ID_TO_SCALE = {0: "high", 1: "mid", 2: "low"}


@dataclass
class EvalResult:
    name: str
    checkpoint_path: str
    step: int
    overall_loss: float
    num_batches: int
    num_samples: int
    by_scale: Dict[str, Dict[str, float]]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate multiscale checkpoints on val split")
    p.add_argument("--cache_index_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="", help="训练目录（默认从此处找 best/final）")
    p.add_argument("--best_ckpt", type=str, default="", help="best checkpoint 路径（可选）")
    p.add_argument("--final_ckpt", type=str, default="", help="final checkpoint 路径（可选）")

    p.add_argument("--scan_step_ckpts", action="store_true", help="扫描 save_dir 下 multiscale_step_*.pt 并批量评估")
    p.add_argument("--step_glob", type=str, default="multiscale_step_*.pt", help="step ckpt 扫描模式（相对 save_dir）")
    p.add_argument("--step_limit", type=int, default=0, help="最多评估多少个 step ckpt（0=全部）")

    p.add_argument("--output_dir", type=str, default="", help="输出目录，默认 save_dir/eval")
    p.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu/mps")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=100, help="最多评估多少个 batch")
    p.add_argument("--train_log_path", type=str, default="", help="训练日志路径（json line），用于绘制 train_curve")

    p.add_argument("--pad_token", type=int, default=0)
    p.add_argument("--latent_len_high", type=int, default=64)
    p.add_argument("--latent_len_mid", type=int, default=128)
    p.add_argument("--latent_len_low", type=int, default=256)

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


def _extract_step(path: str) -> int:
    m = re.search(r"step_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def resolve_ckpts(args) -> Dict[str, str]:
    best_ckpt = args.best_ckpt
    final_ckpt = args.final_ckpt

    if args.save_dir:
        if not best_ckpt:
            best_ckpt = os.path.join(args.save_dir, "multiscale_best.pt")
        if not final_ckpt:
            final_ckpt = os.path.join(args.save_dir, "multiscale_final.pt")

    ckpts: Dict[str, str] = {}
    if best_ckpt and os.path.exists(best_ckpt):
        ckpts["best"] = best_ckpt
    if final_ckpt and os.path.exists(final_ckpt):
        ckpts["final"] = final_ckpt

    if args.scan_step_ckpts:
        if not args.save_dir:
            raise ValueError("启用 --scan_step_ckpts 时必须提供 --save_dir")
        pattern = os.path.join(args.save_dir, args.step_glob)
        step_paths = sorted(glob.glob(pattern), key=_extract_step)
        if args.step_limit and args.step_limit > 0:
            step_paths = step_paths[: args.step_limit]
        for p in step_paths:
            step = _extract_step(p)
            name = f"step_{step}" if step >= 0 else os.path.splitext(os.path.basename(p))[0]
            ckpts[name] = p

    if not ckpts:
        raise ValueError("未找到可评估 checkpoint。请提供 --save_dir 或显式传入 ckpt 路径")

    return ckpts


def build_model_from_ckpt_args(ckpt_args: Dict, cli_args) -> MultiScaleLatentSpaceConditionalModel:
    # 优先 checkpoint 参数，缺省回退到 CLI 默认
    model = MultiScaleLatentSpaceConditionalModel(
        latent_dim=int(ckpt_args.get("latent_dim", cli_args.latent_dim)),
        embedding_dim=int(ckpt_args.get("embedding_dim", cli_args.embedding_dim)),
        midi_vocab_size=int(ckpt_args.get("midi_vocab_size", cli_args.midi_vocab_size)),
        num_layers=int(ckpt_args.get("num_layers", cli_args.num_layers)),
        dropout=float(ckpt_args.get("dropout", cli_args.dropout)),
        max_seq=int(ckpt_args.get("max_seq", cli_args.max_seq)),
        use_cross_attention=True,
        cross_attention_heads=8,
    )

    # 与训练一致：VAE 冻结
    for p in model.vae_encoder.parameters():
        p.requires_grad = False
    for p in model.vae_decoder.parameters():
        p.requires_grad = False
    model.vae_encoder.eval()
    model.vae_decoder.eval()
    return model


@torch.no_grad()
def evaluate_checkpoint(
    name: str,
    ckpt_path: str,
    dl_val: DataLoader,
    structural_builder: StructuralFeatureBuilder,
    args,
    device: torch.device,
) -> EvalResult:
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    model = build_model_from_ckpt_args(ckpt_args, args).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    total_loss_sum = 0.0
    total_count = 0
    num_batches = 0

    scale_loss_sum = {"high": 0.0, "mid": 0.0, "low": 0.0}
    scale_count = {"high": 0, "mid": 0, "low": 0}

    for bidx, batch in enumerate(dl_val):
        if bidx >= args.max_batches:
            break

        audio_latents = batch["audio_latents"].to(device)
        midi_tokens = batch["midi_tokens"].to(device)
        midi_mask = batch["midi_mask"].to(device)

        if midi_tokens.size(1) > args.max_seq:
            midi_tokens = midi_tokens[:, :args.max_seq]
            midi_mask = midi_mask[:, :args.max_seq]

        scale_id = batch["scale_id"].to(device)
        target_lens = batch["target_latent_len"].to(device)

        latents_resized, latent_mask = resize_latent_per_sample(audio_latents, target_lens)
        structural_features = structural_builder.build(batch).to(device)

        pred_noise, noise = model.forward_with_latent_and_structure(
            audio_latent=latents_resized,
            midi_tokens=midi_tokens,
            structural_features=structural_features,
            scale_id=scale_id,
            midi_mask=midi_mask,
            condition_dropout_rate=0.0,
            scale_dropout_rate=0.0,
            weak_low_alignment=True,
            token_shift_for_weak_alignment=0,
        )

        # overall
        loss = masked_mse_loss(pred_noise, noise, latent_mask)
        bsz = int(audio_latents.size(0))
        total_loss_sum += float(loss.item()) * bsz
        total_count += bsz
        num_batches += 1

        # per-scale (batch 内按样本筛选重算)
        for sid in [0, 1, 2]:
            sname = ID_TO_SCALE[sid]
            sel = (scale_id == sid)
            n = int(sel.sum().item())
            if n == 0:
                continue

            loss_s = masked_mse_loss(pred_noise[sel], noise[sel], latent_mask[sel])
            scale_loss_sum[sname] += float(loss_s.item()) * n
            scale_count[sname] += n

    if total_count == 0:
        raise RuntimeError("验证集评估为空，请检查数据集或 --max_batches")

    by_scale = {}
    for s in ["high", "mid", "low"]:
        cnt = scale_count[s]
        by_scale[s] = {
            "loss": float(scale_loss_sum[s] / max(1, cnt)),
            "samples": int(cnt),
        }

    return EvalResult(
        name=name,
        checkpoint_path=ckpt_path,
        step=int(ckpt.get("step", -1)) if isinstance(ckpt, dict) else -1,
        overall_loss=float(total_loss_sum / total_count),
        num_batches=num_batches,
        num_samples=total_count,
        by_scale=by_scale,
    )


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _format_compare_table(best: EvalResult, final: EvalResult) -> str:
    rows = [
        ("overall_loss", best.overall_loss, final.overall_loss),
        ("high_loss", best.by_scale["high"]["loss"], final.by_scale["high"]["loss"]),
        ("mid_loss", best.by_scale["mid"]["loss"], final.by_scale["mid"]["loss"]),
        ("low_loss", best.by_scale["low"]["loss"], final.by_scale["low"]["loss"]),
    ]

    lines = [
        "| metric | best | final | final-best |",
        "|---|---:|---:|---:|",
    ]
    for name, b, f in rows:
        lines.append(f"| {name} | {b:.6f} | {f:.6f} | {f - b:+.6f} |")
    return "\n".join(lines)


def _plot_val_curve(curve_rows: List[Dict], output_path: str):
    rows = [r for r in curve_rows if int(r.get("step", -1)) >= 0]
    if len(rows) < 2:
        return

    rows = sorted(rows, key=lambda x: int(x["step"]))
    steps = [int(r["step"]) for r in rows]
    overall = [float(r["overall_loss"]) for r in rows]
    high = [float(r["high_loss"]) for r in rows]
    mid = [float(r["mid_loss"]) for r in rows]
    low = [float(r["low_loss"]) for r in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, overall, label="val_overall", linewidth=2.2)
    plt.plot(steps, high, label="val_high", alpha=0.8)
    plt.plot(steps, mid, label="val_mid", alpha=0.8)
    plt.plot(steps, low, label="val_low", alpha=0.8)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Validation Curve")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _parse_train_curve_from_log(train_log_path: str) -> List[Dict[str, float]]:
    if not train_log_path or (not os.path.exists(train_log_path)):
        return []

    rows: List[Dict[str, float]] = []
    with open(train_log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("{"):
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if "step" in obj and "loss" in obj:
                try:
                    rows.append({"step": int(obj["step"]), "loss": float(obj["loss"])})
                except Exception:
                    continue

    # 去重并排序
    uniq = {}
    for r in rows:
        uniq[int(r["step"])] = float(r["loss"])
    return [{"step": k, "loss": uniq[k]} for k in sorted(uniq.keys())]


def _plot_train_curve(train_rows: List[Dict[str, float]], output_path: str):
    if len(train_rows) < 2:
        return

    steps = [int(r["step"]) for r in train_rows]
    losses = [float(r["loss"]) for r in train_rows]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label="train_loss", linewidth=1.8)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Train Curve (from log)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    ckpts = resolve_ckpts(args)

    target_latent_lens = {
        "high": args.latent_len_high,
        "mid": args.latent_len_mid,
        "low": args.latent_len_low,
    }
    collate = partial(
        collate_cached_multiscale_batch,
        pad_token=args.pad_token,
        target_latent_lens=target_latent_lens,
    )

    ds_val = CachedMultiscaleLatentDataset(cache_index_path=args.cache_index_path, split="val")
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
    )

    structural_builder = StructuralFeatureBuilder()

    results: List[EvalResult] = []
    for name, path in ckpts.items():
        print(f"Evaluating {name}: {path}")
        res = evaluate_checkpoint(
            name=name,
            ckpt_path=path,
            dl_val=dl_val,
            structural_builder=structural_builder,
            args=args,
            device=device,
        )
        results.append(res)
        print(json.dumps({
            "name": res.name,
            "step": res.step,
            "overall_loss": res.overall_loss,
            "num_batches": res.num_batches,
            "num_samples": res.num_samples,
            "by_scale": res.by_scale,
        }, ensure_ascii=False))

    output_dir = args.output_dir or (os.path.join(args.save_dir, "eval") if args.save_dir else "logs/multiscale_eval")
    os.makedirs(output_dir, exist_ok=True)

    # 为稳定可视化输出，按 step 排序（未知 step 放后）
    def _sort_key(x: EvalResult):
        return (10**12 if x.step < 0 else x.step, x.name)

    results = sorted(results, key=_sort_key)

    summary_rows = []
    scale_rows = []
    curve_rows = []
    for r in results:
        summary_rows.append({
            "name": r.name,
            "checkpoint_path": r.checkpoint_path,
            "step": r.step,
            "overall_loss": f"{r.overall_loss:.8f}",
            "num_batches": r.num_batches,
            "num_samples": r.num_samples,
        })

        for s in ["high", "mid", "low"]:
            scale_rows.append({
                "name": r.name,
                "scale": s,
                "loss": f"{r.by_scale[s]['loss']:.8f}",
                "samples": r.by_scale[s]["samples"],
            })

        # 验证曲线用：单行包含整体和三尺度 loss
        curve_rows.append({
            "name": r.name,
            "step": r.step,
            "overall_loss": f"{r.overall_loss:.8f}",
            "high_loss": f"{r.by_scale['high']['loss']:.8f}",
            "mid_loss": f"{r.by_scale['mid']['loss']:.8f}",
            "low_loss": f"{r.by_scale['low']['loss']:.8f}",
            "num_batches": r.num_batches,
            "num_samples": r.num_samples,
            "checkpoint_path": r.checkpoint_path,
        })

    _write_csv(
        os.path.join(output_dir, "checkpoint_summary.csv"),
        summary_rows,
        ["name", "checkpoint_path", "step", "overall_loss", "num_batches", "num_samples"],
    )
    _write_csv(
        os.path.join(output_dir, "scale_breakdown.csv"),
        scale_rows,
        ["name", "scale", "loss", "samples"],
    )
    _write_csv(
        os.path.join(output_dir, "val_curve.csv"),
        curve_rows,
        [
            "name",
            "step",
            "overall_loss",
            "high_loss",
            "mid_loss",
            "low_loss",
            "num_batches",
            "num_samples",
            "checkpoint_path",
        ],
    )

    compare_md = ""
    compare_rows = []
    by_name = {r.name: r for r in results}
    if "best" in by_name and "final" in by_name:
        best = by_name["best"]
        final = by_name["final"]
        compare_md = _format_compare_table(best, final)

        compare_rows = [
            {
                "metric": "overall_loss",
                "best": f"{best.overall_loss:.8f}",
                "final": f"{final.overall_loss:.8f}",
                "delta_final_minus_best": f"{(final.overall_loss - best.overall_loss):+.8f}",
            },
            {
                "metric": "high_loss",
                "best": f"{best.by_scale['high']['loss']:.8f}",
                "final": f"{final.by_scale['high']['loss']:.8f}",
                "delta_final_minus_best": f"{(final.by_scale['high']['loss'] - best.by_scale['high']['loss']):+.8f}",
            },
            {
                "metric": "mid_loss",
                "best": f"{best.by_scale['mid']['loss']:.8f}",
                "final": f"{final.by_scale['mid']['loss']:.8f}",
                "delta_final_minus_best": f"{(final.by_scale['mid']['loss'] - best.by_scale['mid']['loss']):+.8f}",
            },
            {
                "metric": "low_loss",
                "best": f"{best.by_scale['low']['loss']:.8f}",
                "final": f"{final.by_scale['low']['loss']:.8f}",
                "delta_final_minus_best": f"{(final.by_scale['low']['loss'] - best.by_scale['low']['loss']):+.8f}",
            },
        ]

        _write_csv(
            os.path.join(output_dir, "best_final_compare.csv"),
            compare_rows,
            ["metric", "best", "final", "delta_final_minus_best"],
        )

        with open(os.path.join(output_dir, "best_final_compare.md"), "w", encoding="utf-8") as f:
            f.write(compare_md + "\n")

    out_json = {
        "device": str(device),
        "cache_index_path": args.cache_index_path,
        "max_batches": args.max_batches,
        "results": [
            {
                "name": r.name,
                "checkpoint_path": r.checkpoint_path,
                "step": r.step,
                "overall_loss": r.overall_loss,
                "num_batches": r.num_batches,
                "num_samples": r.num_samples,
                "by_scale": r.by_scale,
            }
            for r in results
        ],
        "best_final_compare_markdown": compare_md,
    }

    with open(os.path.join(output_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    # 绘图输出
    val_curve_png = os.path.join(output_dir, "val_curve.png")
    _plot_val_curve(curve_rows, val_curve_png)

    train_curve_png = os.path.join(output_dir, "train_curve.png")
    train_rows = _parse_train_curve_from_log(args.train_log_path)
    if train_rows:
        _plot_train_curve(train_rows, train_curve_png)

    print(f"Saved summary JSON: {os.path.join(output_dir, 'eval_summary.json')}")
    print(f"Saved CSV: {os.path.join(output_dir, 'checkpoint_summary.csv')}")
    print(f"Saved CSV: {os.path.join(output_dir, 'scale_breakdown.csv')}")
    print(f"Saved CSV: {os.path.join(output_dir, 'val_curve.csv')}")
    if os.path.exists(val_curve_png):
        print(f"Saved PNG: {val_curve_png}")
    if train_rows and os.path.exists(train_curve_png):
        print(f"Saved PNG: {train_curve_png}")
    if compare_rows:
        print(f"Saved CSV: {os.path.join(output_dir, 'best_final_compare.csv')}")
        print("\nBest vs Final comparison:")
        print(compare_md)


if __name__ == "__main__":
    main()
