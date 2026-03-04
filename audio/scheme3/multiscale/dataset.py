"""
配对样本（audio + MIDI）的时间多尺度切分数据集。

职责：
- 枚举可配对样本（按 basename）
- 每次 __getitem__ 调用时执行一次随机 multi-scale time-based segment
- 输出统一结构，直接可用于后续 VAE / MIDI 条件 / UNet
"""

from __future__ import annotations

from dataclasses import asdict
import glob
import os
from typing import Dict, List, Optional

from torch.utils.data import Dataset

from .segmenter import TimeBasedMultiScaleSegmenter, TimeScaleSegmentSample


class PairedTimeSegmentDataset(Dataset):
    def __init__(
        self,
        audio_dir: str,
        midi_dir: str,
        segmenter: Optional[TimeBasedMultiScaleSegmenter] = None,
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
        seed: int = 1234,
    ):
        self.audio_dir = audio_dir
        self.midi_dir = midi_dir
        self.segmenter = segmenter or TimeBasedMultiScaleSegmenter()

        pairs = self._collect_pairs(audio_dir, midi_dir)
        if len(pairs) == 0:
            raise ValueError(f"未找到可配对样本: audio_dir={audio_dir}, midi_dir={midi_dir}")

        import random
        rng = random.Random(seed)
        rng.shuffle(pairs)

        if len(split_ratio) != 3:
            raise ValueError(f"split_ratio 必须是长度为 3 的 tuple，当前为: {split_ratio}")

        train_r, val_r, test_r = split_ratio
        if min(train_r, val_r, test_r) < 0:
            raise ValueError(f"split_ratio 不能包含负值，当前为: {split_ratio}")
        total_r = train_r + val_r + test_r
        if total_r <= 0:
            raise ValueError(f"split_ratio 总和必须 > 0，当前为: {split_ratio}")
        train_r, val_r, test_r = train_r / total_r, val_r / total_r, test_r / total_r

        n = len(pairs)
        n_train = int(n * train_r)
        n_val = int(n * val_r)

        if split == "train":
            self.pairs = pairs[:n_train]
        elif split == "val":
            self.pairs = pairs[n_train:n_train + n_val]
        elif split == "test":
            self.pairs = pairs[n_train + n_val:]
        else:
            raise ValueError("split 必须是 train/val/test")

        if len(self.pairs) == 0:
            raise ValueError(f"split={split} 下没有样本，请检查数据规模与 split_ratio")

    @staticmethod
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
            # 避免 basename 重名导致错误配对（此前会被静默覆盖）
            if len(audio_map[base]) != 1 or len(midi_map[base]) != 1:
                continue
            pairs.append({"audio_path": audio_map[base][0], "midi_path": midi_map[base][0], "basename": base})

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]
        seg: TimeScaleSegmentSample = self.segmenter.sample_for_training_step(
            audio_path=pair["audio_path"],
            midi_path=pair["midi_path"],
        )

        # 统一输出结构（dict 化，便于 DataLoader collate 与训练端使用）
        return {
            "basename": pair["basename"],
            "audio_path": pair["audio_path"],
            "midi_path": pair["midi_path"],
            "scale": seg.scale,
            "sample_rate": seg.sample_rate,
            "audio_segment": seg.audio_segment,
            "midi_tokens": seg.midi_segment.midi_tokens,
            "midi_pretty": seg.midi_segment.pretty_midi_obj,
            "window": asdict(seg.window),
            "global_position": seg.global_position,
            "event_stats": {
                "note_count": seg.midi_segment.note_count,
                "pedal_count": seg.midi_segment.pedal_count,
                "program_count": seg.midi_segment.program_count,
            },
        }


def collate_time_segment_batch(
    batch: List[Dict],
    pad_token: int,
    target_latent_lens: Dict[str, int] | None = None,
) -> Dict:
    """
    将 variable-length 的 audio_segment / midi_tokens 对齐为 batch。

    输出字段：
    - audio_segments: (B, 1, T_max)
    - midi_tokens: (B, S_max)
    - midi_mask: (B, S_max) bool（True=有效）
    - scale_id: (B,) long, high/mid/low -> 0/1/2
    - target_latent_len: (B,) long
    - global_position, window, scale 等元信息按 list 返回
    """
    import torch

    scale_to_id = {"high": 0, "mid": 1, "low": 2}
    target_latent_lens = target_latent_lens or {"high": 64, "mid": 128, "low": 256}

    bsz = len(batch)
    max_audio_len = max(item["audio_segment"].shape[-1] for item in batch)
    max_midi_len = max(len(item["midi_tokens"]) for item in batch)

    audio_segments = torch.zeros((bsz, 1, max_audio_len), dtype=torch.float32)
    midi_tokens = torch.full((bsz, max_midi_len), fill_value=pad_token, dtype=torch.long)
    midi_mask = torch.zeros((bsz, max_midi_len), dtype=torch.bool)
    scale_id = torch.zeros((bsz,), dtype=torch.long)
    target_latent_len = torch.zeros((bsz,), dtype=torch.long)

    for i, item in enumerate(batch):
        a = item["audio_segment"]
        audio_segments[i, :, :a.shape[-1]] = a

        tokens = item["midi_tokens"]
        if len(tokens) > 0:
            midi_tokens[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            midi_mask[i, :len(tokens)] = True

        s = item["scale"]
        scale_id[i] = scale_to_id.get(s, 1)
        target_latent_len[i] = int(target_latent_lens.get(s, 128))

    return {
        "audio_segments": audio_segments,
        "midi_tokens": midi_tokens,
        "midi_mask": midi_mask,
        "scale_id": scale_id,
        "target_latent_len": target_latent_len,
        "scale": [item["scale"] for item in batch],
        "global_position": [item["global_position"] for item in batch],
        "window": [item["window"] for item in batch],
        "basename": [item["basename"] for item in batch],
        "audio_path": [item["audio_path"] for item in batch],
        "midi_path": [item["midi_path"] for item in batch],
        "event_stats": [item["event_stats"] for item in batch],
    }
