"""
评估“多尺度切分本身收益”的实验脚本（离线分析版）。

用途：
1) 统计三尺度采样分布是否均衡
2) 统计不同尺度下 token 密度、pedal 密度、时长分布
3) 对比 baseline（固定单尺度）和 multiscale 的数据覆盖差异

说明：
- 该脚本不依赖模型，仅分析数据切分收益。
- 如需模型侧收益，请结合训练日志对比（同训练预算下 val loss / 生成指标）。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from .segmenter import TimeBasedMultiScaleSegmenter
from .dataset import PairedTimeSegmentDataset


def summarize(dataset, max_samples: int = 500):
    stats = {
        "num_samples": 0,
        "scale_count": defaultdict(int),
        "duration_sec": defaultdict(list),
        "token_density": defaultdict(list),
        "pedal_density": defaultdict(list),
    }

    n = min(len(dataset), max_samples)
    for i in range(n):
        item = dataset[i]
        scale = item["scale"]
        dur = max(1e-6, float(item["window"]["duration_sec"]))
        token_count = len(item["midi_tokens"])
        pedal_count = float(item["event_stats"]["pedal_count"])

        stats["num_samples"] += 1
        stats["scale_count"][scale] += 1
        stats["duration_sec"][scale].append(dur)
        stats["token_density"][scale].append(token_count / dur)
        stats["pedal_density"][scale].append(pedal_count / dur)

    out = {
        "num_samples": stats["num_samples"],
        "scale_count": dict(stats["scale_count"]),
        "duration_sec": {},
        "token_density": {},
        "pedal_density": {},
    }

    for k in ["high", "mid", "low"]:
        durs = np.array(stats["duration_sec"].get(k, [0.0]), dtype=np.float32)
        toks = np.array(stats["token_density"].get(k, [0.0]), dtype=np.float32)
        peds = np.array(stats["pedal_density"].get(k, [0.0]), dtype=np.float32)
        out["duration_sec"][k] = {
            "mean": float(durs.mean()),
            "std": float(durs.std()),
            "p50": float(np.percentile(durs, 50)),
            "p90": float(np.percentile(durs, 90)),
        }
        out["token_density"][k] = {
            "mean": float(toks.mean()),
            "std": float(toks.std()),
            "p50": float(np.percentile(toks, 50)),
            "p90": float(np.percentile(toks, 90)),
        }
        out["pedal_density"][k] = {
            "mean": float(peds.mean()),
            "std": float(peds.std()),
            "p50": float(np.percentile(peds, 50)),
            "p90": float(np.percentile(peds, 90)),
        }

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_wav_dir", type=str, required=True)
    parser.add_argument("--midi_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--output_json", type=str, default="logs/scheme3_multiscale/data_coverage_eval.json")
    args = parser.parse_args()

    # 1) baseline: 固定 mid 尺度（可近似原始固定长度切片的结构覆盖）
    mid_segmenter = TimeBasedMultiScaleSegmenter(scale_bars={"high": 4, "mid": 4, "low": 4}, seed=42)
    baseline_ds = PairedTimeSegmentDataset(
        audio_dir=args.audio_wav_dir,
        midi_dir=args.midi_dir,
        segmenter=mid_segmenter,
        split="train",
        split_ratio=(0.8, 0.1, 0.1),
        seed=42,
    )
    baseline_stats = summarize(baseline_ds, max_samples=args.max_samples)

    # 2) multiscale: high/mid/low = 1/4/8
    multi_segmenter = TimeBasedMultiScaleSegmenter(scale_bars={"high": 1, "mid": 4, "low": 8}, seed=42)
    multiscale_ds = PairedTimeSegmentDataset(
        audio_dir=args.audio_wav_dir,
        midi_dir=args.midi_dir,
        segmenter=multi_segmenter,
        split="train",
        split_ratio=(0.8, 0.1, 0.1),
        seed=42,
    )
    multiscale_stats = summarize(multiscale_ds, max_samples=args.max_samples)

    result = {
        "baseline_fixed_mid": baseline_stats,
        "multiscale_1_4_8": multiscale_stats,
        "recommendation": {
            "check_scale_balance": "multiscale scale_count should be roughly balanced",
            "check_density_spread": "multiscale token_density/pedal_density should show wider but controllable coverage",
            "next_step": "run paired training and compare val loss + generation metrics",
        },
    }

    import os
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved to: {args.output_json}")


if __name__ == "__main__":
    main()
