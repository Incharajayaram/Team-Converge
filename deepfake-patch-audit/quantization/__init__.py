"""Quantization tools for model compression and edge deployment."""

from .advanced_quantization import (
    QuantizationAwareTraining,
    MixedPrecisionQuantization,
    PostTrainingQuantization,
    KnowledgeDistillationQuantization,
)

try:
    from .improved_dynamic_range import ImprovedDynamicRangeQuantizer
except ImportError:
    ImprovedDynamicRangeQuantizer = None

__all__ = [
    "QuantizationAwareTraining",
    "MixedPrecisionQuantization",
    "PostTrainingQuantization",
    "KnowledgeDistillationQuantization",
    "ImprovedDynamicRangeQuantizer",
]
