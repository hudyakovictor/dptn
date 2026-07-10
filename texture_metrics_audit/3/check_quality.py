#!/usr/bin/env python3
import json, numpy as np
with open("/home/user/dptn/analysis_raw.json") as f:
    data = json.load(f)
real = data["real"]
def vals(k): return np.array([x.get(k, np.nan) for x in real])
for k in ["noise_level","sharpness_score","overall_quality"]:
    v = vals(k); v=v[~np.isnan(v)]
    print(k, f"mean {v.mean():.2f} median {np.median(v):.2f} p10 {np.percentile(v,10):.2f} p90 {np.percentile(v,90):.2f} min {v.min():.2f} max {v.max():.2f}")
    # count failing current thresholds
    if k=="noise_level":
        print("  >25:", (v>25).sum(), "/", len(v))
    if k=="sharpness_score":
        print("  <50:", (v<50).sum())
    if k=="overall_quality":
        print("  <0.4:", (v<0.4).sum())
# check year correlation? paths include year
year_buckets = {}
for r in real:
    name = r["path"]
    # real_1999_08_12.png
    try:
        year = int(name.split("_")[1])
    except: year=0
    year_buckets.setdefault(year, []).append(r)
for y in sorted(year_buckets):
    rows = year_buckets[y]
    s = np.mean([x.get("sharpness_score",0) for x in rows])
    n = np.mean([x.get("noise_level",0) for x in rows])
    oq = np.mean([x.get("overall_quality",0) for x in rows])
    print(y, len(rows), f"sharp {s:.1f} noise {n:.2f} oq {oq:.2f}")
