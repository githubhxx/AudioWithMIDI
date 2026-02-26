"""
MIDI处理模块 - 方案三专用
包含MIDI事件表示和Transformer训练
"""

from .model import MusicTransformer
from .train import *
from .preprocess import preprocess_midi, preprocess_midi_files_under
from .data import Data

__all__ = [
    'MusicTransformer',
    'preprocess_midi',
    'preprocess_midi_files_under',
    'Data',
]

