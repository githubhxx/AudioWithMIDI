"""
基于离线 latent cache 的多尺度数据集。

特点：
- 训练时不做音频切分/编码
- 在线仅做 MIDI 窗口切分 + token 编码
- 直接读取匹配的 latent
"""

from __future__ import annotations

import json
import os
import pickle
import random
from typing import Dict, List

import pretty_midi
import torch
from torch.utils.data import Dataset

from midi.midi_processor.processor import encode_pretty_midi


class CachedMultiscaleLatentDataset(Dataset):
    def __init__(
        self,
        cache_index_path: str,
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
        seed: int = 1234,
    ):
        if not os.path.exists(cache_index_path):
            raise FileNotFoundError(cache_index_path)

        with open(cache_index_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        self.meta = obj.get("meta", {})
        records = obj.get("records", [])
        if not records:
            raise ValueError("cache index records is empty")

        # 按 basename 分组，保证划分在曲目级，避免泄露
        by_base: Dict[str, List[Dict]] = {}
        for r in records:
            by_base.setdefault(r["basename"], []).append(r)

        bases = list(by_base.keys())
        rng = random.Random(seed)
        rng.shuffle(bases)

        if len(split_ratio) != 3:
            raise ValueError(f"split_ratio must have 3 elements, got: {split_ratio}")

        tr, va, te = split_ratio
        if min(tr, va, te) < 0:
            raise ValueError(f"split_ratio cannot contain negative values, got: {split_ratio}")
        s = tr + va + te
        if s <= 0:
            raise ValueError(f"split_ratio sum must be > 0, got: {split_ratio}")
        tr, va, te = tr / s, va / s, te / s

        n = len(bases)
        n_train = int(n * tr)
        n_val = int(n * va)

        if split == "train":
            sel = set(bases[:n_train])
        elif split == "val":
            sel = set(bases[n_train:n_train + n_val])
        elif split == "test":
            sel = set(bases[n_train + n_val:])
        else:
            raise ValueError("split must be train/val/test")

        self.records = [r for b in sel for r in by_base[b]]
        if not self.records:
            raise ValueError(f"split={split} has no records")

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _slice_midi_by_seconds(midi_obj: pretty_midi.PrettyMIDI, start_sec: float, end_sec: float) -> pretty_midi.PrettyMIDI:
        out_mid = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        for inst in midi_obj.instruments:
            new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)

            for note in inst.notes:
                if note.end <= start_sec or note.start >= end_sec:
                    continue
                st = max(note.start, start_sec) - start_sec
                ed = min(note.end, end_sec) - start_sec
                if ed <= st:
                    continue
                new_inst.notes.append(pretty_midi.Note(int(note.velocity), int(note.pitch), float(st), float(ed)))

            for cc in inst.control_changes:
                if cc.time < start_sec or cc.time >= end_sec:
                    continue
                new_inst.control_changes.append(
                    pretty_midi.ControlChange(number=int(cc.number), value=int(cc.value), time=float(cc.time - start_sec))
                )

            for pb in inst.pitch_bends:
                if pb.time < start_sec or pb.time >= end_sec:
                    continue
                new_inst.pitch_bends.append(pretty_midi.PitchBend(int(pb.pitch), float(pb.time - start_sec)))

            out_mid.instruments.append(new_inst)

        return out_mid

    def __getitem__(self, idx: int) -> Dict:
        r = self.records[idx]

        with open(r["latent_path"], "rb") as f:
            latent = pickle.load(f)
        latent_tensor = torch.from_numpy(latent).float()

        midi_obj = pretty_midi.PrettyMIDI(r["midi_path"])
        midi_seg = self._slice_midi_by_seconds(midi_obj, float(r["start_sec"]), float(r["end_sec"]))
        midi_tokens = encode_pretty_midi(midi_seg)

        pedal_count = 0
        note_count = 0
        program_count = 0
        for inst in midi_seg.instruments:
            program_count += 1
            note_count += len(inst.notes)
            pedal_count += sum(1 for cc in inst.control_changes if cc.number == 64)

        return {
            "basename": r["basename"],
            "audio_path": r["audio_path"],
            "midi_path": r["midi_path"],
            "scale": r["scale"],
            "window": {
                "start_sec": float(r["start_sec"]),
                "end_sec": float(r["end_sec"]),
                "duration_sec": float(r["duration_sec"]),
                "bar_start_index": int(r["bar_start_index"]),
                "bar_end_index": int(r["bar_end_index"]),
                "beat_start_index": int(r["beat_start_index"]),
                "beat_end_index": int(r["beat_end_index"]),
            },
            "global_position": {
                "bar_start_index": int(r["bar_start_index"]),
                "bar_end_index": int(r["bar_end_index"]),
                "beat_start_index": int(r["beat_start_index"]),
                "beat_end_index": int(r["beat_end_index"]),
            },
            "event_stats": {
                "note_count": note_count,
                "pedal_count": pedal_count,
                "program_count": program_count,
            },
            "midi_tokens": midi_tokens,
            "audio_latent": latent_tensor,
        }


def collate_cached_multiscale_batch(
    batch: List[Dict],
    pad_token: int,
    target_latent_lens: Dict[str, int] | None = None,
) -> Dict:
    target_latent_lens = target_latent_lens or {"high": 64, "mid": 128, "low": 256}
    scale_to_id = {"high": 0, "mid": 1, "low": 2}

    bsz = len(batch)
    max_midi_len = max(len(x["midi_tokens"]) for x in batch)
    max_latent_len = max(x["audio_latent"].shape[1] for x in batch)
    latent_dim = batch[0]["audio_latent"].shape[0]

    midi_tokens = torch.full((bsz, max_midi_len), fill_value=pad_token, dtype=torch.long)
    midi_mask = torch.zeros((bsz, max_midi_len), dtype=torch.bool)
    audio_latents = torch.zeros((bsz, latent_dim, max_latent_len), dtype=torch.float32)
    scale_id = torch.zeros((bsz,), dtype=torch.long)
    target_latent_len = torch.zeros((bsz,), dtype=torch.long)

    for i, x in enumerate(batch):
        t = x["midi_tokens"]
        if t:
            midi_tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)
            midi_mask[i, :len(t)] = True

        z = x["audio_latent"]
        audio_latents[i, :, :z.shape[1]] = z

        s = x["scale"]
        scale_id[i] = scale_to_id.get(s, 1)
        target_latent_len[i] = int(target_latent_lens.get(s, 128))

    return {
        "audio_latents": audio_latents,
        "midi_tokens": midi_tokens,
        "midi_mask": midi_mask,
        "scale_id": scale_id,
        "target_latent_len": target_latent_len,
        "scale": [x["scale"] for x in batch],
        "global_position": [x["global_position"] for x in batch],
        "window": [x["window"] for x in batch],
        "basename": [x["basename"] for x in batch],
        "audio_path": [x["audio_path"] for x in batch],
        "midi_path": [x["midi_path"] for x in batch],
        "event_stats": [x["event_stats"] for x in batch],
    }
