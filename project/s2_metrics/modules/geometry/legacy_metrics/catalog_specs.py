from __future__ import annotations
from .metric_catalog_full import generate_rows
from .types import MetricSpec
from .common import ALL_BUCKETS

CODE_TO_BUCKET = {
    "FR": "frontal",
    "LTL": "left_threequarter_light",
    "RTL": "right_threequarter_light",
    "LTM": "left_threequarter_mid",
    "RTM": "right_threequarter_mid",
    "LTD": "left_threequarter_deep",
    "RTD": "right_threequarter_deep",
    "LP": "left_profile",
    "RP": "right_profile",
}
VALID_SPACES = {"raw", "canon_bucket", "shape_neutral", "unit_template", "pair_umeyama", "pair_icp"}


def _buckets_from_views(views: str) -> tuple[str, ...]:
    text = (views or "").replace("PAIR:", "")
    buckets: list[str] = []
    for token in text.replace(";", ",").split(","):
        code = token.strip().split()[0] if token.strip() else ""
        if code in CODE_TO_BUCKET:
            buckets.append(CODE_TO_BUCKET[code])
    return tuple(dict.fromkeys(buckets)) or ALL_BUCKETS


def _spaces(raw: str) -> tuple[str, ...]:
    out = tuple(x.strip() for x in (raw or "").split(",") if x.strip() in VALID_SPACES)
    return out or ("canon_bucket",)


def specs_for_module(module_name: str, *, families: set[str] | None = None, scope: str = "single") -> list[MetricSpec]:
    out=[]
    for r in generate_rows():
        if r['module'] != module_name: continue
        if families and r['family'] not in families: continue
        if r['scope'] != scope: continue
        side = r['side'] if r['side'] in {'L','R','B','NA'} else 'NA'
        out.append(MetricSpec(
            name=r['metric_name'], family=r['family'], group=r['group'], zone=r['zone'], side=side,
            buckets=_buckets_from_views(r.get('views','')), source_spaces=_spaces(r.get('source_spaces','')),
            scope=r['scope'], implementation=module_name, unit='ratio', normalization='face_scale', tags=(r['status'],)
        ))
    return out
