from __future__ import annotations

"""Active metric catalog: only metrics with stability median >=80% (baseline cal_poses)."""

import csv
from pathlib import Path

ALL9 = "FR,LTL,RTL,LTM,RTM,LTD,RTD,LP,RP"
FRONTAL_LIGHT = "FR,LTL,RTL"
VISIBLE_ALL = "FR,LTL,RTL,LTM,RTM,LTD,RTD,LP,RP visible-side gated"
PROFILE_SAFE = "LTM,RTM,LTD,RTD,LP,RP profile-safe"
PAIR_ALL9 = "PAIR:FR,LTL,RTL,LTM,RTM,LTD,RTD,LP,RP shared-visibility"

FIELDS = [
    "family",
    "group",
    "module",
    "metric_name",
    "zone",
    "side",
    "views",
    "source_spaces",
    "scope",
    "status",
    "catalog_role",
    "function_template",
    "notes",
]

_ACTIVE_CSV = Path(__file__).with_name("metric_catalog_active.csv")


def row(
    family,
    group,
    module,
    name,
    zone,
    side="NA",
    views=ALL9,
    spaces="canon_bucket",
    scope="single",
    status="planned",
    catalog_role="both",
    func="",
    notes="",
):
    return {
        "family": family,
        "group": group,
        "module": module,
        "metric_name": name,
        "zone": zone,
        "side": side,
        "views": views,
        "source_spaces": spaces,
        "scope": scope,
        "status": status,
        "catalog_role": catalog_role,
        "function_template": func,
        "notes": notes,
    }


def generate_rows() -> list[dict]:
    if not _ACTIVE_CSV.is_file():
        raise FileNotFoundError(f"active metric catalog missing: {_ACTIVE_CSV}")
    with _ACTIVE_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_catalog(path: Path) -> list[dict]:
    rows = generate_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def _write_role_catalogs(rows: list[dict]) -> None:
    def role(r: dict) -> str:
        return str(r.get("catalog_role") or "").strip().lower()

    by = {
        "metric_catalog_stability.csv": {"stability", "both"},
        "metric_catalog_discrimination.csv": {"discrimination", "both"},
    }
    for name, allowed in by.items():
        out = Path(__file__).with_name(name)
        kept = [r for r in rows if role(r) in allowed]
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(kept)


def main() -> int:
    out = Path(__file__).with_name("metric_catalog_full.csv")
    rows = write_catalog(out)
    _write_role_catalogs(rows)
    print(f"wrote {out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
