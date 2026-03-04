"""Scheme3 多尺度子模块。"""

from .segmenter import TimeBasedMultiScaleSegmenter, TimeScaleSegmentSample, SegmentWindow, MidiSegmentData
from .dataset import PairedTimeSegmentDataset, collate_time_segment_batch
from .cached_dataset import CachedMultiscaleLatentDataset, collate_cached_multiscale_batch
from .structural_features import StructuralFeatureBuilder
from .model import MultiScaleLatentSpaceConditionalModel, StructuralConditionHead
from .latent_cache import build_multiscale_latent_cache
from .train_utils import resize_latent_to_target, resize_latent_per_sample, masked_mse_loss

__all__ = [
    "TimeBasedMultiScaleSegmenter",
    "TimeScaleSegmentSample",
    "SegmentWindow",
    "MidiSegmentData",
    "PairedTimeSegmentDataset",
    "collate_time_segment_batch",
    "CachedMultiscaleLatentDataset",
    "collate_cached_multiscale_batch",
    "StructuralFeatureBuilder",
    "MultiScaleLatentSpaceConditionalModel",
    "StructuralConditionHead",
    "build_multiscale_latent_cache",
    "resize_latent_to_target",
    "resize_latent_per_sample",
    "masked_mse_loss",
]
