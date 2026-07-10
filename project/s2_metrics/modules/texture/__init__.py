"""Texture V2 modules."""

from .extractor_v2 import TextureExtractorV2
from .classifier import TextureSkinClassifierV2
from .catalog import TEXTURE_CORE_METRICS, PHYSICAL_AUX_METRICS, load_texture_metric_catalog

__all__ = [
    "TextureExtractorV2",
    "TextureSkinClassifierV2",
    "TEXTURE_CORE_METRICS",
    "PHYSICAL_AUX_METRICS",
    "load_texture_metric_catalog",
]

# Backward compatibility
TextureExtractor = TextureExtractorV2
TextureSkinClassifier = TextureSkinClassifierV2