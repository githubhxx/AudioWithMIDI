"""
单步演示：对一条配对样本执行 Low/Mid/High time-based multi-scale segment。

用法：
python -m audio.scheme3.demo_time_segment_step \
  --audio_path /path/to/sample.wav \
  --midi_path /path/to/sample.mid
"""

import argparse

from .multiscale.segmenter import TimeBasedMultiScaleSegmenter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--midi_path", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()

    segmenter = TimeBasedMultiScaleSegmenter(target_sample_rate=args.sample_rate)

    for scale in ["high", "mid", "low"]:
        seg = segmenter.sample_for_training_step(
            audio_path=args.audio_path,
            midi_path=args.midi_path,
            force_scale=scale,
        )
        print("=" * 60)
        print(f"scale: {seg.scale}")
        print(f"audio_segment shape: {tuple(seg.audio_segment.shape)}, sr={seg.sample_rate}")
        print(f"window(sec): [{seg.window.start_sec:.3f}, {seg.window.end_sec:.3f}] dur={seg.window.duration_sec:.3f}")
        print(
            f"bar idx: {seg.window.bar_start_index}~{seg.window.bar_end_index}, "
            f"beat idx: {seg.window.beat_start_index}~{seg.window.beat_end_index}"
        )
        print(
            f"midi events: tokens={len(seg.midi_segment.midi_tokens)}, "
            f"notes={seg.midi_segment.note_count}, pedal={seg.midi_segment.pedal_count}, "
            f"program={seg.midi_segment.program_count}"
        )


if __name__ == "__main__":
    main()
