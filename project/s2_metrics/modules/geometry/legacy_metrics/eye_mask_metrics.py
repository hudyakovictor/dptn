from __future__ import annotations

from .catalog_specs import specs_for_module
from .common import emit

IMPLEMENTATION = "eye_mask_metrics.py"


def specs():
    # F_mask metrics are forensic / artifact diagnostics only.
    # They must stay out of identity scoring.
    return specs_for_module(IMPLEMENTATION, families={"F_mask"})


def compute(ctx, specs_):
    out = []
    spec_by = {s.name: s for s in specs_}

    # Минимальный сигнал для discrimination: используем уже посчитанный texture_silicone_prob как proxy.
    tf = getattr(ctx, "texture_forensics", None) or {}
    prob = tf.get("texture_silicone_prob")
    if "eye_mask_silicone_prob" in spec_by and prob is not None:
        if mv := emit(spec_by["eye_mask_silicone_prob"], float(prob), confidence=0.55):
            out.append(mv)
    return out

