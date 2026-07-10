from __future__ import annotations

from importlib import import_module
from types import ModuleType

from .types import MetricSpec
from .policy import spec_allowed_for_bucket

MODULE_NAMES = [
    "existing_backend",
    "shape_3dmm",
    "distances",
    "angles",
    "triangles",
    "quads",
    "cross_sections",
    "curvature_normals",
    "area_volume_convexity",
    "geodesics",
    "mirror_asymmetry",
    "orbit_special",
    "brow_lid",
    "nose_bridge",
    "mandible",
    "zygomatic_temporal",
    "periocular_lid",
    "texture_roi",
    "zone_morphology",
    "zone_relations",
    "spectral_zone",
    "interorbital_bridge",
    "palpebral_aperture",
    "eye_mask_metrics",
    "midface_profile",
    # Pair/chronology modules are registered for pair runner integration.
    "dense_residuals",
    "pair_zone_residuals",
    # temporal_chronology: scope=chronology only, not loaded for single-photo metrics
]


def load_modules() -> list[ModuleType]:
    pkg = __name__.rsplit(".", 1)[0]
    modules: list[ModuleType] = []
    for name in MODULE_NAMES:
        modules.append(import_module(f"{pkg}.{name}"))
    return modules


def all_specs() -> list[MetricSpec]:
    out: list[MetricSpec] = []
    for module in load_modules():
        if hasattr(module, "specs"):
            out.extend(module.specs())
    return out


def specs_for_bucket(
    bucket: str,
    *,
    scope: str = "single",
    yaw_deg: float | None = None,
) -> list[MetricSpec]:
    return [
        s
        for s in all_specs()
        if s.scope == scope and spec_allowed_for_bucket(s, bucket, yaw_deg=yaw_deg)
    ]
