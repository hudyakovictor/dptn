"""Build minimal test dataset for GitHub."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

SRC = Path("/Volumes/SDCARD/skin_dataset")
DST = Path("/Users/victorkhudyakov/dutin/newapp/deeputin/test_dataset")
DST.mkdir(exist_ok=True)

with open(SRC / "poses.json") as f:
    all_poses = json.load(f)
poses_map = {p["photo_id"]: p for p in all_poses}

SELECTION = {
    "real": [
        "real_calibration_y-6p3r0.png",
        "real_calibration_y40p6r11.png",
        "real_calibration_y-73p29r29.png",
        "real_calibration_y-77p-11r-15.png",
        "real_2000_05_09.png",
        "real_2002_04_18.png",
        "real_2006_06_09.png",
        "real_2013_03_07.png",
        "real_2018_07_16(1).png",
    ],
    "silicone": [
        "silicone_2010_09_01.png",
        "silicone_2012_08_28.png",
        "silicone_2015_05_08.png",
        "silicone_2017_08_20(2).png",
        "silicone_2018_06_20.png",
        "silicone_2018_07_29.png",
        "silicone_2019_01_06.png",
        "silicone_2017_10_21.png",
        "silicone_2023_06_09.png",
        "silicone_2023_10_06.png",
        "silicone_2024_03_28.png",
        "silicone_2024_08_21.png",
        "silicone_2025_05_08.png",
    ],
}

total_size = 0
copied = {"real": [], "silicone": []}

for label, files in SELECTION.items():
    src_dir = SRC / label
    dst_dir = DST / label
    dst_dir.mkdir(exist_ok=True)
    for fname in files:
        src_file = src_dir / fname
        dst_file = dst_dir / fname
        if not src_file.exists():
            print(f"  MISSING: {label}/{fname}")
            continue
        shutil.copy2(src_file, dst_file)
        size = src_file.stat().st_size
        total_size += size
        copied[label].append(fname)
        print(f"  OK: {label}/{fname} ({size//1024}KB)")

filtered_poses = [poses_map[f] for f in copied["silicone"] if f in poses_map]
for f in copied["real"]:
    m = re.search(r'y(-?\d+)p(-?\d+)r(-?\d+)', f)
    if m:
        yaw, pitch, roll = int(m.group(1)), int(m.group(2)), int(m.group(3))
        filtered_poses.append({
            "photo_id": f, "source": f"calibration/{f}",
            "bucket": "frontal" if abs(yaw) < 15 else ("left_threequarter_light" if yaw < -30 else "right_threequarter_light"),
            "yaw": float(yaw), "pitch": float(pitch), "roll": float(roll),
        })

with open(DST / "poses.json", "w") as f:
    json.dump(filtered_poses, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"Real: {len(copied['real'])} | Silicone: {len(copied['silicone'])} | Total: {len(copied['real'])+len(copied['silicone'])}")
print(f"Size: {total_size/1024/1024:.1f} MB | Poses: {len(filtered_poses)}")
print(f"{'='*60}")
