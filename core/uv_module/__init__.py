"""
uv_module — генерация HD UV-текстур лица для 3DDFA_V3.
"""
from .hd_uv_generator import HDUVTextureGenerator, HDUVConfig
from .uvio import UVIOExporter, ObjData

__all__ = [
    "HDUVTextureGenerator",
    "HDUVConfig",
    "UVIOExporter",
    "ObjData",
]