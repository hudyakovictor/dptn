from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

import pandas as pd

GEOMETRY_CORE_METRICS = [
    "quality",
    "chin_surface_area_ratio",
    "chin_bbox_area_ratio",
    "chin_bbox_volume_ratio",
    "zone_chin_normal_mean_x",
    "zone_chin_normal_mean_y",
    "zone_chin_normal_mean_z",
    "zone_chin_normal_variance",
    "zone_chin_span_vertical_ratio",
    "zone_chin_radial_dispersion_ratio",
    "zone_chin_plane_residual_std_ratio",
    "zone_chin_plane_residual_p95_ratio",
    "brow_ridge_L_convexity_index",
    "brow_ridge_R_convexity_index",
    "brow_ridge_L_bbox_volume_ratio",
    "brow_ridge_R_bbox_volume_ratio",
    "zone_brow_ridge_L_depth_std_ratio",
    "zone_brow_ridge_R_depth_std_ratio",
    "zone_brow_ridge_L_span_vertical_ratio",
    "zone_brow_ridge_R_span_vertical_ratio",
    "zone_brow_ridge_L_normal_mean_x",
    "zone_brow_ridge_R_normal_mean_x",
    "zone_brow_ridge_L_normal_mean_y",
    "zone_brow_ridge_R_normal_mean_y",
    "zone_brow_ridge_L_normal_variance",
    "zone_brow_ridge_R_normal_variance",
    "zone_rel_brow_ridge_L_to_jaw_angle_L_normal_angle_deg",
    "zone_rel_brow_ridge_R_to_jaw_angle_R_normal_angle_deg",
    "zone_rel_brow_ridge_L_to_jaw_angle_L_distance_ratio",
    "zone_rel_brow_ridge_R_to_jaw_angle_R_distance_ratio",
    "zone_rel_jaw_L_to_brow_ridge_L_normal_angle_deg",
    "zone_rel_jaw_R_to_brow_ridge_R_normal_angle_deg",
    "zone_rel_jaw_R_to_orbit_R_distance_ratio",
    "zone_rel_cheekbone_L_to_temporal_L_distance_ratio",
    "zone_rel_cheekbone_R_to_temporal_R_distance_ratio",
    "zone_rel_cheekbone_R_to_orbit_R_distance_ratio",
    "jaw_L_surface_area_ratio",
    "jaw_R_surface_area_ratio",
    "jaw_L_convexity_index",
    "jaw_R_convexity_index",
    "zone_jaw_L_span_lateral_ratio",
    "zone_jaw_R_span_lateral_ratio",
    "zone_jaw_L_span_depth_ratio",
    "zone_jaw_R_span_depth_ratio",
    "zone_jaw_L_depth_std_ratio",
    "zone_jaw_R_depth_std_ratio",
    "zone_jaw_L_normal_mean_y",
    "zone_jaw_R_normal_mean_y",
    "zone_jaw_L_normal_mean_z",
    "zone_jaw_R_normal_mean_z",
    "zone_jaw_angle_L_normal_mean_x",
    "zone_jaw_angle_L_normal_mean_y",
    "zone_jaw_angle_L_normal_mean_z",
    "zone_jaw_angle_R_normal_mean_x",
    "zone_jaw_angle_R_normal_mean_y",
    "zone_jaw_angle_R_normal_mean_z",
    "zone_jaw_angle_L_normal_variance",
    "zone_jaw_angle_R_normal_variance",
    "zone_jaw_angle_L_bbox_volume_ratio",
    "zone_jaw_angle_R_bbox_volume_ratio",
    "zone_jaw_angle_L_bbox_area_ratio",
    "zone_jaw_angle_R_bbox_area_ratio",
    "zone_jaw_angle_L_span_vertical_ratio",
    "zone_jaw_angle_R_span_vertical_ratio",
    "zone_jaw_angle_L_span_lateral_ratio",
    "zone_jaw_angle_R_span_lateral_ratio",
    "zone_jaw_angle_L_span_depth_ratio",
    "zone_jaw_angle_R_span_depth_ratio",
    "zone_jaw_angle_L_depth_std_ratio",
    "zone_jaw_angle_R_depth_std_ratio",
    "zone_jaw_angle_L_plane_residual_std_ratio",
    "zone_jaw_angle_R_plane_residual_std_ratio",
    "zone_jaw_angle_L_plane_residual_p95_ratio",
    "zone_jaw_angle_R_plane_residual_p95_ratio",
    "zone_jaw_angle_L_radial_dispersion_ratio",
    "zone_jaw_angle_R_radial_dispersion_ratio",
    "zone_ligament_orbital_L_normal_variance",
    "zone_ligament_orbital_R_normal_variance",
    "zone_ligament_orbital_L_normal_mean_y",
    "zone_ligament_orbital_R_normal_mean_y",
    "zone_ligament_orbital_L_span_depth_ratio",
    "zone_ligament_orbital_R_span_depth_ratio",
    "zone_ligament_orbital_L_span_lateral_ratio",
    "zone_ligament_orbital_R_span_lateral_ratio",
    "zone_ligament_orbital_L_depth_std_ratio",
    "zone_ligament_orbital_R_depth_std_ratio",
    "zone_ligament_zygomatic_L_normal_variance",
    "zone_ligament_zygomatic_R_normal_variance",
    "zone_ligament_zygomatic_L_normal_mean_y",
    "zone_ligament_zygomatic_R_normal_mean_y",
    "zone_ligament_zygomatic_L_span_lateral_ratio",
    "zone_ligament_zygomatic_R_span_lateral_ratio",
    "zone_ligament_zygomatic_L_span_depth_ratio",
    "zone_ligament_zygomatic_R_span_depth_ratio",
    "zone_ligament_zygomatic_L_depth_std_ratio",
    "zone_ligament_zygomatic_R_depth_std_ratio",
    "zone_nose_bridge_tip_normal_mean_x",
    "zone_nose_bridge_tip_normal_mean_z",
    "zone_nose_bridge_tip_normal_variance",
    "zone_nose_bridge_tip_depth_std_ratio",
    "zone_nose_bridge_tip_span_depth_ratio",
    "zone_nose_bridge_tip_bbox_area_ratio",
    "zone_nose_bridge_tip_plane_residual_std_ratio",
    "zone_nose_bridge_tip_plane_residual_p95_ratio",
    "nose_bridge_tip_bbox_volume_ratio",
    "nose_bridge_tip_bbox_area_ratio",
    "zone_nose_bridge_tip_bbox_volume_ratio",
    "nose_bridge_tip_convexity_index",
    "zone_nose_wing_L_normal_mean_y",
    "zone_nose_wing_R_normal_mean_y",
    "zone_nose_wing_L_normal_mean_z",
    "zone_nose_wing_R_normal_mean_z",
    "zone_nose_wing_L_span_lateral_ratio",
    "zone_nose_wing_R_span_lateral_ratio",
    "zone_nose_wing_L_span_depth_ratio",
    "zone_nose_wing_R_span_depth_ratio",
    "zone_nose_wing_L_depth_std_ratio",
    "zone_nose_wing_R_depth_std_ratio",
    "zone_nose_wing_L_radial_dispersion_ratio",
    "zone_nose_wing_R_radial_dispersion_ratio",
    "zone_nose_wing_L_plane_residual_std_ratio",
    "zone_nose_wing_R_plane_residual_std_ratio",
    "zone_cheekbone_L_normal_mean_x",
    "zone_cheekbone_L_normal_mean_y",
    "zone_cheekbone_R_normal_mean_x",
    "zone_cheekbone_R_normal_mean_y",
    "zone_cheekbone_L_normal_variance",
    "zone_cheekbone_R_normal_variance",
    "zone_cheekbone_L_span_lateral_ratio",
    "zone_cheekbone_R_span_lateral_ratio",
    "zone_cheekbone_L_span_vertical_ratio",
    "zone_cheekbone_R_span_vertical_ratio",
    "zone_cheekbone_L_depth_std_ratio",
    "zone_cheekbone_R_depth_std_ratio",
    "zone_cheekbone_L_plane_residual_std_ratio",
    "zone_cheekbone_R_plane_residual_std_ratio",
    "zone_cheekbone_L_plane_residual_p95_ratio",
    "zone_cheekbone_R_plane_residual_p95_ratio",
    "cheekbone_L_bbox_volume_ratio",
    "cheekbone_R_bbox_volume_ratio",
    "cheekbone_L_surface_area_ratio",
    "cheekbone_R_surface_area_ratio",
    "cheekbone_L_convexity_index",
    "cheekbone_R_convexity_index",
    "zone_orbit_L_normal_mean_x",
    "zone_orbit_R_normal_mean_x",
    "zone_orbit_L_normal_mean_y",
    "zone_orbit_R_normal_mean_y",
    "zone_orbit_L_normal_variance",
    "zone_orbit_R_normal_variance",
    "zone_orbit_L_span_vertical_ratio",
    "zone_orbit_R_span_vertical_ratio",
    "zone_orbit_L_span_depth_ratio",
    "zone_orbit_R_span_depth_ratio",
    "zone_orbit_L_span_lateral_ratio",
    "zone_orbit_R_span_lateral_ratio",
    "zone_orbit_L_depth_std_ratio",
    "zone_orbit_R_depth_std_ratio",
    "zone_orbit_L_radial_dispersion_ratio",
    "zone_orbit_R_radial_dispersion_ratio",
    "zone_orbit_L_bbox_area_ratio",
    "zone_orbit_R_bbox_area_ratio",
    "zone_orbit_L_bbox_volume_ratio",
    "zone_orbit_R_bbox_volume_ratio",
    "orbit_L_bbox_area_ratio",
    "orbit_R_bbox_area_ratio",
    "orbit_L_bbox_volume_ratio",
    "orbit_R_bbox_volume_ratio",
    "orbit_L_ellipse_area_ratio",
    "orbit_R_ellipse_area_ratio",
    "orbit_L_ellipse_minor_ratio",
    "orbit_R_ellipse_minor_ratio",
    "orbit_L_surface_area_ratio",
    "orbit_R_surface_area_ratio",
    "orbit_L_orbital_bowl_volume_proxy",
    "orbit_R_orbital_bowl_volume_proxy",
    "orbit_L_convexity_index",
    "orbit_R_convexity_index",
    "zone_orbit_L_plane_residual_p95_ratio",
    "zone_orbit_R_plane_residual_p95_ratio",
    "zone_temporal_L_normal_mean_x",
    "zone_temporal_L_normal_mean_y",
    "zone_temporal_L_normal_mean_z",
    "zone_temporal_R_normal_mean_x",
    "zone_temporal_R_normal_mean_y",
    "zone_temporal_R_normal_mean_z",
    "zone_temporal_L_normal_variance",
    "zone_temporal_R_normal_variance",
    "zone_temporal_L_span_vertical_ratio",
    "zone_temporal_R_span_vertical_ratio",
    "zone_temporal_L_span_lateral_ratio",
    "zone_temporal_R_span_lateral_ratio",
    "zone_temporal_L_span_depth_ratio",
    "zone_temporal_R_span_depth_ratio",
    "zone_rel_temporal_L_to_orbit_L_distance_ratio",
    "zone_rel_temporal_R_to_orbit_R_distance_ratio",
    "zone_rel_temporal_L_to_orbit_L_normal_angle_deg",
    "zone_rel_temporal_R_to_orbit_R_normal_angle_deg",
    "zone_rel_cheekbone_R_to_temporal_R_distance_ratio",
    "zone_rel_cheekbone_R_to_temporal_R_normal_angle_deg",
    "zone_forehead_normal_mean_y",
    "zone_forehead_span_depth_ratio",
    "zone_forehead_span_vertical_ratio",
    "zone_forehead_bbox_volume_ratio",
    "zone_forehead_bbox_area_ratio",
    "zone_forehead_plane_residual_p95_ratio",
    "zone_forehead_plane_residual_std_ratio",
    "zone_forehead_depth_std_ratio",
    "zone_forehead_normal_variance",
    "forehead_bbox_volume_ratio",
    "forehead_bbox_area_ratio",
    "forehead_surface_area_ratio",
    "forehead_convexity_index",
    "chin_convexity_index",
    "L_brow_lid_depth_gap_ratio",
    "L_brow_lid_vertical_gap_ratio",
    "L_brow_overhang_proxy",
    "L_eye_socket_bowl_depth",
    "L_gonion_to_chin_curve_sharpness",
    "L_jaw_depth_span_ratio",
    "L_jawline_arc_length_ratio",
    "L_lid_surface_curvature",
    "L_malar_peak_height_ratio",
    "L_ramus_height_proxy_ratio",
    "L_submalar_hollow_proxy",
    "L_temporal_concavity_proxy",
    "L_temporal_to_orbit_depth_gradient",
    "L_temporal_to_zygoma_step_ratio",
    "R_brow_lid_depth_gap_ratio",
    "R_brow_lid_vertical_gap_ratio",
    "R_brow_overhang_proxy",
    "R_eye_socket_bowl_depth",
    "R_lid_surface_curvature",
    "R_malar_peak_height_ratio",
]

_FALLBACK_GEOMETRY_CORE_METRICS = tuple(GEOMETRY_CORE_METRICS)


def _test_personas_root() -> Path:
    return Path(__file__).resolve().parents[4] / "test_personas"


def _load_test_personas_geometry_metrics() -> dict[str, set[str]]:
    root = _test_personas_root()
    buckets: dict[str, set[str]] = {}
    if not root.exists():
        return buckets
    for metrics_path in root.rglob("metrics.csv"):
        summary_path = metrics_path.with_name("summary.json")
        bucket = "unclassified"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                bucket = str((summary.get("pose") or {}).get("bucket", "unclassified"))
            except Exception:
                bucket = "unclassified"
        try:
            with metrics_path.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        bucket_set = buckets.setdefault(bucket, set())
        for row in rows:
            metric_name = str(row.get("metric_name") or row.get("метрика") or "").strip()
            if not metric_name or metric_name == "nan":
                continue
            if metric_name == "eye_mask_silicone_prob":
                continue
            bucket_set.add(metric_name)
    return buckets


_TEST_PERSONAS_GEOMETRY_BUCKETS = _load_test_personas_geometry_metrics()
GEOMETRY_CORE_METRICS = tuple(
    sorted({name for names in _TEST_PERSONAS_GEOMETRY_BUCKETS.values() for name in names})
    or _FALLBACK_GEOMETRY_CORE_METRICS
)
GEOMETRY_BUCKET_METRICS = {
    bucket: tuple(sorted(names)) for bucket, names in _TEST_PERSONAS_GEOMETRY_BUCKETS.items() if names
}


def load_geometry_metric_catalog(path: str | Path | None = None) -> list[dict[str, Any]]:
    if path is None:
        return [{"metric_name": name, "priority": idx + 1} for idx, name in enumerate(GEOMETRY_CORE_METRICS)]
    p = Path(path)
    if not p.exists():
        return [{"metric_name": name, "priority": idx + 1} for idx, name in enumerate(GEOMETRY_CORE_METRICS)]
    df = pd.read_csv(p)
    out = []
    for idx, row in df.head(20).iterrows():
        out.append(
            {
                "metric_name": str(row.get("metric_name", "")),
                "full_name_ru": str(row.get("full_name_ru", "")),
                "metric_family": str(row.get("metric_family", "")),
                "metric_group": str(row.get("metric_group", "")),
                "anatomical_zone": str(row.get("anatomical_zone", "")),
                "pose_buckets": str(row.get("pose_buckets", "")),
                "priority": int(idx) + 1,
            }
        )
    return out
