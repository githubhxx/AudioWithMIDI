"""Shared mel/STFT config for scheme3 multiscale train/infer/eval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MelConfig:
    n_fft: int = 1024
    hop_length: int = 160
    win_length: int = 1024
    n_mels: int = 80
    sampling_rate: int = 16000
    fmin: float = 0.0
    fmax: float = 8000.0
    center: bool = True


DEFAULT_MEL_CONFIG = MelConfig()
