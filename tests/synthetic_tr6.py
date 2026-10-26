"""A tiny fake TR-6 with the same folder layout, file names and manifest columns as the real dataset.

Used by the tests and by ``scripts/check_setup.py --synthetic`` so the code can be exercised without
the dataset. Images are small random JPEGs; values carry no meaning.
"""
from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

from PIL import Image

FRUITS = ["Banana", "Carrot", "Guava", "Indian_Gooseberry", "Mango", "Tomato"]
SESSION_TIMES = {"morning": "090000", "afternoon": "140000", "evening": "170000"}
COLUMNS = (
    "fruit,label,actual_date,corrected_day_index,local_day_index,ir_morning_count,ir_afternoon_count,"
    "ir_evening_count,ir_total_sessions,ir_t_min_mean,ir_t_max_mean,ir_t_range_mean,srgb_morning_count,"
    "srgb_afternoon_count,srgb_evening_count,srgb_total_sessions,methane_morning_present,"
    "methane_afternoon_present,methane_evening_present,methane_total_sessions,methane_morning_ppm,"
    "methane_afternoon_ppm,methane_evening_ppm,methane_day_mean_ppm,sequence_break,days_until_spoilage,"
    "signed_days_from_spoilage,exclude,notes,ir_morning_tmin,ir_morning_tmax,ir_morning_trange,"
    "ir_afternoon_tmin,ir_afternoon_tmax,ir_afternoon_trange,ir_evening_tmin,ir_evening_tmax,"
    "ir_evening_trange,methane_morning_std,methane_morning_left,methane_morning_right,methane_afternoon_std,"
    "methane_afternoon_left,methane_afternoon_right,methane_evening_std,methane_evening_left,"
    "methane_evening_right,split"
).split(",")


def _jpeg(path: Path, rng: random.Random, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), tuple(rng.randrange(256) for _ in range(3)))
    img.save(path, quality=70)


def make_synthetic_tr6(root: Path, n_days: int = 6, seed: int = 0):
    """Write ``root/TR-6`` and ``root/master_manifest.csv``; return (data_root, manifest_path)."""
    rng = random.Random(seed)
    data_root = root / "TR-6"
    rows = []
    splits = ["train", "train", "train", "val", "test", "train"]
    for fruit in FRUITS:
        for label, folder in (("not_spoiled", "Not_spoiled"), ("spoiled", "Spoiled")):
            base = data_root / "Classified" / fruit / folder
            start = date(2025, 9, 11)
            for d in range(n_days):
                day = (start + timedelta(days=d)).strftime("%Y%m%d")
                row = {c: "" for c in COLUMNS}
                row.update(fruit=fruit, label=label, actual_date=day, corrected_day_index=d + 1,
                           local_day_index=d + 1, exclude="FALSE", sequence_break="FALSE",
                           days_until_spoilage=max(n_days - 1 - d, 0),
                           signed_days_from_spoilage=n_days - 1 - d, split=splits[d % len(splits)])
                for s, hhmmss in SESSION_TIMES.items():
                    has_rgb, has_ir, has_gas = rng.random() > 0.2, rng.random() > 0.2, rng.random() > 0.3
                    if has_rgb:
                        _jpeg(base / "sRGB_images" / f"{day}_{hhmmss}.jpg", rng)
                    if has_ir:
                        _jpeg(base / "IR_fusion_images" / f"{day}_{hhmmss}_28.5C_33.1C.jpg", rng)
                    ppm = rng.uniform(5, 40) * (1.5 if label == "spoiled" else 1.0)
                    row.update({f"srgb_{s}_count": int(has_rgb), f"ir_{s}_count": int(has_ir),
                                f"methane_{s}_present": "TRUE" if has_gas else "FALSE",
                                f"methane_{s}_ppm": round(ppm, 3) if has_gas else "",
                                f"methane_{s}_std": round(rng.uniform(0, 2), 3) if has_gas else "",
                                f"methane_{s}_left": round(ppm + rng.uniform(-2, 2), 3) if has_gas else "",
                                f"methane_{s}_right": round(ppm + rng.uniform(-2, 2), 3) if has_gas else ""})
                rows.append(row)
    manifest = root / "master_manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return data_root, manifest
