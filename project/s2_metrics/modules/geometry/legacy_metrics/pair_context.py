from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .types import PairMetricContext


def _photo_id(recon: Any) -> str:
    p = getattr(recon, "image_path", None)
    if p is None:
        return "unknown"
    try:
        return Path(p).stem
    except Exception:
        return str(p)


def build_pair_metric_context_from_prep(
    recon_a: Any,
    recon_b: Any,
    prep: dict[str, Any],
    alignment: Any,
) -> PairMetricContext:
    shared_idx = np.asarray(prep["shared_idx"], dtype=np.int64)
    pts_a = np.asarray(prep["points_a_unit"], dtype=float)
    pts_b = np.asarray(prep["points_b_unit"], dtype=float)
    # Parity with mesh_zone: A→B Umeyama on shared → compare source_aligned vs B (not B transformed wrong-way).
    vertices_a = pts_a.copy()
    try:
        source_aligned = np.asarray(alignment.source_aligned, dtype=float)
        if source_aligned.shape[0] == shared_idx.shape[0]:
            vertices_a[shared_idx] = source_aligned
        else:
            vertices_a[shared_idx] = (
                alignment.scale * (pts_a[shared_idx] @ alignment.rotation) + alignment.translation
            )
    except Exception:
        vertices_a = pts_a
    aligned_b = pts_b
    aid, bid = _photo_id(recon_a), _photo_id(recon_b)
    return PairMetricContext(
        photo_id_a=aid,
        photo_id_b=bid,
        pair_id=f"{aid}__{bid}",
        pose_bucket_a=str(prep.get("view_group_a") or getattr(recon_a, "pose_bucket", "frontal")),
        pose_bucket_b=str(prep.get("view_group_b") or getattr(recon_b, "pose_bucket", "frontal")),
        yaw_a_deg=float(np.asarray(recon_a.angles_deg).reshape(-1)[1]) if getattr(recon_a, "angles_deg", None) is not None else 0.0,
        yaw_b_deg=float(np.asarray(recon_b.angles_deg).reshape(-1)[1]) if getattr(recon_b, "angles_deg", None) is not None else 0.0,
        shared_idx=shared_idx,
        vertices_a_unit=vertices_a,
        vertices_b_unit_aligned=aligned_b,
        normals_a=getattr(recon_a, "normals_world", None),
        triangles=np.asarray(getattr(recon_a, "triangles", np.zeros((0, 3))), dtype=np.int64),
        macro_indices={},
        visibility_weights=np.asarray(prep.get("weights"), dtype=float) if prep.get("weights") is not None else None,
        alignment=alignment,
    )
