from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

FIELDNAMES = [
    "schema_version", "kind", "photo_id_a", "photo_id_b", "pair_id", "pose_bucket",
    "metric_name", "value", "source_space", "scope", "metric_family", "metric_group",
    "anatomical_zone", "side", "unit", "generated_at",
]

KNOWN_PAIR_ZONE_PREFIXES = [
    "forehead", "brow_ridge_L", "brow_ridge_R", "orbit_L", "orbit_R", "nose_bridge_tip",
    "nose_wing_L", "nose_wing_R", "cheekbone_L", "cheekbone_R", "cheek_soft_L", "cheek_soft_R",
    "temporal_L", "temporal_R", "jaw_L", "jaw_R", "jaw_angle_L", "jaw_angle_R", "chin",
    "ligament_orbital_L", "ligament_orbital_R", "ligament_zygomatic_L", "ligament_zygomatic_R",
]
KNOWN_F9_PREFIXES = [
    "global", "orbit_L", "orbit_R", "brow_ridge_L", "brow_ridge_R", "nose_bridge_tip",
    "cheekbone_L", "cheekbone_R", "temporal_L", "temporal_R", "jaw_L", "jaw_R", "chin",
]


def _side_from_zone(zone: str) -> str:
    if zone.endswith("_L"):
        return "L"
    if zone.endswith("_R"):
        return "R"
    return "NA"


def _infer_meta(name: str) -> dict[str, str]:
    if name.startswith("pair_zone_"):
        core = name[len("pair_zone_"):]
        for zone in sorted(KNOWN_PAIR_ZONE_PREFIXES, key=len, reverse=True):
            if core.startswith(zone + "_"):
                return {
                    "family": "F_pair_zone",
                    "group": "generic_pair_zone_residuals",
                    "zone": zone,
                    "side": _side_from_zone(zone),
                }
        return {"family": "F_pair_zone", "group": "generic_pair_zone_residuals", "zone": "unknown", "side": "NA"}

    for zone in sorted(KNOWN_F9_PREFIXES, key=len, reverse=True):
        if name.startswith(zone + "_"):
            return {
                "family": "F9",
                "group": "dense_residuals",
                "zone": zone,
                "side": _side_from_zone(zone),
            }
    return {"family": "F9", "group": "dense_residuals", "zone": "global", "side": "NA"}


def append_pair_metric_candidates_csv(
    path: Path,
    *,
    kind: str,
    photo_id_a: str,
    photo_id_b: str,
    bucket: str,
    candidates: dict[str, Any],
    generated_at: str,
) -> int:
    if not candidates:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    pair_id = f"{photo_id_a}__{photo_id_b}"
    n = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        for name, value in sorted(candidates.items()):
            try:
                v = float(value)
            except Exception:
                continue
            meta = _infer_meta(name)
            w.writerow({
                "schema_version": 1,
                "kind": kind,
                "photo_id_a": photo_id_a,
                "photo_id_b": photo_id_b,
                "pair_id": pair_id,
                "pose_bucket": bucket,
                "metric_name": name,
                "value": v,
                "source_space": "pair_umeyama",
                "scope": "pair",
                "metric_family": meta["family"],
                "metric_group": meta["group"],
                "anatomical_zone": meta["zone"],
                "side": meta["side"],
                "unit": "unit_scale",
                "generated_at": generated_at,
            })
            n += 1
    return n
