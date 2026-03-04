"""
Low / Mid / High 音乐时间（bar / beat）多尺度切分器。

设计目标：
1) 每个训练 step 可对一条配对样本（audio + midi）完成一次随机多尺度切分
2) 切分窗口基于音乐时间（bar / beat 对齐），而不是纯秒级随机裁剪
3) 输出统一结构，供后续：
   - VAE 编码（audio_segment）
   - MIDI 条件构建（midi_segment 与 midi_tokens）
   - UNet 训练（global_position）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import pretty_midi
import torch
import torchaudio

from midi.midi_processor.processor import encode_pretty_midi


@dataclass
class SegmentWindow:
    scale: str
    start_sec: float
    end_sec: float
    duration_sec: float
    bar_start_index: int
    bar_end_index: int
    beat_start_index: int
    beat_end_index: int


@dataclass
class MidiSegmentData:
    pretty_midi_obj: pretty_midi.PrettyMIDI
    midi_tokens: List[int]
    note_count: int
    pedal_count: int
    program_count: int


@dataclass
class TimeScaleSegmentSample:
    scale: str
    sample_rate: int
    audio_segment: torch.Tensor  # (1, T)
    midi_segment: MidiSegmentData
    window: SegmentWindow
    global_position: Dict[str, int]


class TimeBasedMultiScaleSegmenter:
    """
    基于音乐时间（bar / beat）的多尺度切分器。

    默认尺度定义（可通过 scale_bars 覆盖）：
      - high: 1 bar
      - mid: 4 bars
      - low: 8 bars
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        scale_bars: Optional[Dict[str, int]] = None,
        seed: Optional[int] = None,
    ):
        self.target_sample_rate = target_sample_rate
        self.scale_bars = scale_bars or {"high": 1, "mid": 4, "low": 8}
        self._rng = random.Random(seed)

        for required in ("low", "mid", "high"):
            if required not in self.scale_bars:
                raise ValueError(f"scale_bars 必须包含键: {required}")
            if self.scale_bars[required] <= 0:
                raise ValueError(f"scale_bars['{required}'] 必须 > 0")

    def sample_for_training_step(
        self,
        audio_path: str,
        midi_path: str,
        force_scale: Optional[str] = None,
    ) -> TimeScaleSegmentSample:
        """
        在一个训练 step 中完成：
        1) 选择尺度（low/mid/high）
        2) 采样 bar/beat 对齐窗口
        3) 映射到秒级并切 audio
        4) 过滤/截断 MIDI（notes、pedal、program）
        5) 生成全局位置索引
        6) 输出统一结构
        """
        waveform, sr = torchaudio.load(audio_path)
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.target_sample_rate)
            sr = self.target_sample_rate

        midi_obj = pretty_midi.PrettyMIDI(midi_path)
        if force_scale is not None:
            if force_scale not in self.scale_bars:
                raise ValueError(f"未知尺度: {force_scale}")
            scale = force_scale
        else:
            scale = self._rng.choice(["low", "mid", "high"])

        downbeats, beats = self._build_music_grids(midi_obj, total_audio_duration_sec=waveform.size(-1) / sr)
        window = self._sample_window(scale=scale, downbeats=downbeats, beats=beats)

        audio_segment = self._slice_audio_by_seconds(waveform, sr, window.start_sec, window.end_sec)
        midi_segment_obj, note_count, pedal_count, program_count = self._slice_midi_by_seconds(
            midi_obj,
            window.start_sec,
            window.end_sec,
        )
        midi_tokens = encode_pretty_midi(midi_segment_obj)

        return TimeScaleSegmentSample(
            scale=scale,
            sample_rate=sr,
            audio_segment=audio_segment,
            midi_segment=MidiSegmentData(
                pretty_midi_obj=midi_segment_obj,
                midi_tokens=midi_tokens,
                note_count=note_count,
                pedal_count=pedal_count,
                program_count=program_count,
            ),
            window=window,
            global_position={
                "bar_start_index": window.bar_start_index,
                "bar_end_index": window.bar_end_index,
                "beat_start_index": window.beat_start_index,
                "beat_end_index": window.beat_end_index,
            },
        )

    def _build_music_grids(
        self,
        midi_obj: pretty_midi.PrettyMIDI,
        total_audio_duration_sec: float,
    ) -> Tuple[List[float], List[float]]:
        beats = list(midi_obj.get_beats())
        downbeats = list(midi_obj.get_downbeats())

        # fallback：最少保证 beat/downbeat 网格可用
        if not beats:
            beats = [0.0, max(total_audio_duration_sec, midi_obj.get_end_time())]
        else:
            if beats[0] > 1e-6:
                beats = [0.0] + beats
            end_t = max(total_audio_duration_sec, midi_obj.get_end_time())
            if beats[-1] < end_t:
                beats.append(end_t)

        if not downbeats:
            # 若无 downbeat，按 time signature 分组 beats 推断 bar 起点
            ts_changes = midi_obj.time_signature_changes
            numerator = ts_changes[0].numerator if ts_changes else 4
            numerator = max(1, int(numerator))
            downbeats = [beats[i] for i in range(0, len(beats), numerator)]

        if downbeats[0] > 1e-6:
            downbeats = [0.0] + downbeats

        end_t = max(total_audio_duration_sec, midi_obj.get_end_time())
        if downbeats[-1] < end_t:
            downbeats.append(end_t)

        return downbeats, beats

    def _sample_window(self, scale: str, downbeats: List[float], beats: List[float]) -> SegmentWindow:
        bars_per_window = self.scale_bars[scale]
        total_bars = max(1, len(downbeats) - 1)

        # 如果乐曲过短，自动退化为可用 bar 数
        effective_bars = min(bars_per_window, total_bars)
        max_start_bar = max(0, total_bars - effective_bars)
        bar_start = self._rng.randint(0, max_start_bar)
        bar_end = bar_start + effective_bars

        start_sec = downbeats[bar_start]
        end_sec = downbeats[bar_end]
        if end_sec <= start_sec:
            # 极端异常保护
            end_sec = start_sec + 0.1

        beat_start = self._search_leq_index(beats, start_sec)
        beat_end = max(beat_start + 1, self._search_leq_index(beats, end_sec))

        return SegmentWindow(
            scale=scale,
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            duration_sec=float(end_sec - start_sec),
            bar_start_index=int(bar_start),
            bar_end_index=int(max(bar_start, bar_end - 1)),
            beat_start_index=int(max(0, beat_start)),
            beat_end_index=int(max(0, beat_end - 1)),
        )

    @staticmethod
    def _search_leq_index(sorted_times: List[float], t: float) -> int:
        # 返回 <= t 的最后一个索引
        idx = 0
        for i, v in enumerate(sorted_times):
            if v <= t:
                idx = i
            else:
                break
        return idx

    @staticmethod
    def _slice_audio_by_seconds(waveform: torch.Tensor, sr: int, start_sec: float, end_sec: float) -> torch.Tensor:
        n = waveform.size(-1)
        s = int(max(0, min(n, round(start_sec * sr))))
        e = int(max(s + 1, min(n, round(end_sec * sr))))
        return waveform[:, s:e]

    @staticmethod
    def _slice_midi_by_seconds(
        midi_obj: pretty_midi.PrettyMIDI,
        start_sec: float,
        end_sec: float,
    ) -> Tuple[pretty_midi.PrettyMIDI, int, int, int]:
        """
        过滤并截断 MIDI 事件到 [start_sec, end_sec]，并将时间平移到段内局部时间。
        """
        out_mid = pretty_midi.PrettyMIDI(initial_tempo=120.0)

        note_count = 0
        pedal_count = 0
        program_count = 0

        for inst in midi_obj.instruments:
            new_inst = pretty_midi.Instrument(
                program=inst.program,
                is_drum=inst.is_drum,
                name=inst.name,
            )
            program_count += 1

            # notes
            for note in inst.notes:
                if note.end <= start_sec or note.start >= end_sec:
                    continue
                clipped_start = max(note.start, start_sec) - start_sec
                clipped_end = min(note.end, end_sec) - start_sec
                if clipped_end <= clipped_start:
                    continue
                new_inst.notes.append(
                    pretty_midi.Note(
                        velocity=int(note.velocity),
                        pitch=int(note.pitch),
                        start=float(clipped_start),
                        end=float(clipped_end),
                    )
                )
                note_count += 1

            # control changes（重点保留 pedal: cc64；其余也保留，若落在窗口内）
            for cc in inst.control_changes:
                if cc.time < start_sec or cc.time >= end_sec:
                    continue
                new_inst.control_changes.append(
                    pretty_midi.ControlChange(
                        number=int(cc.number),
                        value=int(cc.value),
                        time=float(cc.time - start_sec),
                    )
                )
                if cc.number == 64:
                    pedal_count += 1

            # pitch bends（可选，但作为 MIDI 事件一起保留更完整）
            for pb in inst.pitch_bends:
                if pb.time < start_sec or pb.time >= end_sec:
                    continue
                new_inst.pitch_bends.append(
                    pretty_midi.PitchBend(
                        pitch=int(pb.pitch),
                        time=float(pb.time - start_sec),
                    )
                )

            out_mid.instruments.append(new_inst)

        # 尝试继承原始拍号和速度变化（并裁剪平移时间）
        out_mid.time_signature_changes = []
        for ts in getattr(midi_obj, "time_signature_changes", []):
            if ts.time < start_sec or ts.time >= end_sec:
                continue
            out_mid.time_signature_changes.append(
                pretty_midi.TimeSignature(ts.numerator, ts.denominator, float(ts.time - start_sec))
            )

        out_mid.key_signature_changes = []
        for ks in getattr(midi_obj, "key_signature_changes", []):
            if ks.time < start_sec or ks.time >= end_sec:
                continue
            out_mid.key_signature_changes.append(
                pretty_midi.KeySignature(int(ks.key_number), float(ks.time - start_sec))
            )

        # pretty_midi 的 tempo changes 是从 tick_scales 推导；直接复制私有结构不稳定。
        # 这里保持简洁，不强拷贝 tempo map，默认依赖 note/control 的相对秒时间进行条件建模。

        return out_mid, note_count, pedal_count, program_count
