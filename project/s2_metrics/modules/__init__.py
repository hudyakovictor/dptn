"""Модули этапа 2."""

from .geometry_extractor import GeometryExtractor
from .texture.extractor_v2 import TextureExtractorV2 as TextureExtractor
from .zone_analyzer import ZoneAnalyzer
from .geometry import GEOMETRY_CORE_METRICS, GeometryIdentityResolver, load_geometry_metric_catalog
from .texture.classifier import TextureSkinClassifierV2 as TextureSkinClassifier
from .texture.catalog import TEXTURE_CORE_METRICS, load_texture_metric_catalog

__all__ = [
    "GeometryExtractor",
    "TextureExtractor",
    "ZoneAnalyzer",
    "GEOMETRY_CORE_METRICS",
    "GeometryIdentityResolver",
    "TextureSkinClassifier",
    "load_geometry_metric_catalog",
    "load_texture_metric_catalog",
]
