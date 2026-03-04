"""
三尺度时间切分的结构特征构建工具。

目标：
1) 从 time-based segment 的元信息中提取结构条件
2) 提供统一 tensor 格式，供结构条件头使用

当前特征（轻量版）：
- bar_start_index
- beat_start_index
- local_tempo_bpm（窗口内估计）
- pedal_density（窗口内 pedal 事件密度）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


DEFAULT_TEMPO_BPM = 120.0


class StructuralFeatureBuilder:
    def __init__(
        self,
        scale_bars: Optional[Dict[str, int]] = None,
        tempo_clip: Tuple[float, float] = (20.0, 300.0),
        max_bar_index: int = 2048,
        max_beat_index: int = 8192,
    ):
        self.scale_bars = scale_bars or {"high": 1, "mid": 4, "low": 8}
        self.tempo_clip = tempo_clip
        self.max_bar_index = max_bar_index
        self.max_beat_index = max_beat_index

    def build(self, batch_meta: Dict) -> torch.Tensor:
        """
        输入：collate_time_segment_batch 输出字典（含 scale/window/global_position/event_stats）
        输出：结构特征张量 (B, 4)，顺序为
            [bar_start_index, beat_start_index, local_tempo_bpm, pedal_density]
        """
        scales: List[str] = batch_meta["scale"]
        windows: List[Dict] = batch_meta["window"]
        global_positions: List[Dict] = batch_meta["global_position"]
        event_stats: List[Dict] = batch_meta["event_stats"]

        rows = []
        for scale, win, gpos, stats in zip(scales, windows, global_positions, event_stats):
            bars = max(1, int(self.scale_bars.get(scale, 1)))
            dur = max(1e-4, float(win.get("duration_sec", 0.0)))

            # 局部速度（窗口估计）：bars / duration * 4 * 60
            # 假设每小节约 4 拍（与大多数钢琴数据近似一致）
            local_tempo = (bars / dur) * 4.0 * 60.0
            local_tempo = max(self.tempo_clip[0], min(self.tempo_clip[1], local_tempo))

            bar_start_index = int(gpos.get("bar_start_index", 0))
            beat_start_index = int(gpos.get("beat_start_index", 0))

            bar_start_index = max(0, min(self.max_bar_index, bar_start_index))
            beat_start_index = max(0, min(self.max_beat_index, beat_start_index))

            pedal_count = float(stats.get("pedal_count", 0))
            pedal_density = pedal_count / dur

            rows.append([
                float(bar_start_index),
                float(beat_start_index),
                float(local_tempo),
                float(pedal_density),
            ])

        return torch.tensor(rows, dtype=torch.float32)
