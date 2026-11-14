"""
Scan all matched Sentinel-1 <-> MODIS LandCover patch pairs in the SEN12MS
download and record the IGBP (LC band 1) class pixel counts per patch.

Output: index.parquet with one row per matched pair:
    season, scene, patch, s1_path, lc_path, px_1 ... px_17 (pixel counts)

Run:  python scan_lc.py
"""
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import rasterio

# --- dataset layout (matched S1 + LC roots per season) ---
SEASONS = {
    "spring": {
        "s1": r"D:\DATA\ROIs1158_spring",
        "lc": r"D:\DATA\m1474000\ROIs1158_spring",
    },
    "winter": {
        "s1": r"D:\DATA\m1474000\ROIs2017_winter 2",
        "lc": r"D:\DATA\m1474000\ROIs2017_winter",
    },
}

OUT = os.path.join(os.path.dirname(__file__), "index.parquet")
N_CLASSES = 17  # IGBP 1..17


def list_pairs():
    """Enumerate every S1 patch that has a matching LC patch."""
    pairs = []
    for season, roots in SEASONS.items():
        s1_root, lc_root = roots["s1"], roots["lc"]
        for d in sorted(os.listdir(s1_root)):
            if not d.startswith("s1_"):
                continue
            scene = d.split("_", 1)[1]           # e.g. "1"
            lc_dir = os.path.join(lc_root, "lc_" + scene)
            if not os.path.isdir(lc_dir):
                continue
            for s1_path in glob.glob(os.path.join(s1_root, d, "*.tif")):
                fn = os.path.basename(s1_path)
                lc_path = os.path.join(lc_dir, fn.replace("_s1_", "_lc_"))
                if os.path.exists(lc_path):
                    patch = fn.rsplit("_p", 1)[1].replace(".tif", "")
                    pairs.append((season, scene, patch, s1_path, lc_path))
    return pairs


def count_patch(args):
    """Read LC band 1 (IGBP) and return per-class pixel counts."""
    season, scene, patch, s1_path, lc_path = args
    try:
        with rasterio.open(lc_path) as ds:
            igbp = ds.read(1)
        counts = np.bincount(igbp.ravel(), minlength=N_CLASSES + 1)[1:N_CLASSES + 1]
    except Exception:
        counts = np.zeros(N_CLASSES, dtype=np.int64)
    return (season, scene, patch, s1_path, lc_path, counts)


def main():
    t0 = time.time()
    pairs = list_pairs()
    print(f"Matched pairs: {len(pairs):,}")

    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=10) as ex:
        for res in ex.map(count_patch, pairs, chunksize=200):
            season, scene, patch, s1_path, lc_path, counts = res
            row = {
                "season": season,
                "scene": scene,
                "patch": patch,
                "s1_path": s1_path,
                "lc_path": lc_path,
            }
            for c in range(N_CLASSES):
                row[f"px_{c + 1}"] = int(counts[c])
            rows.append(row)
            done += 1
            if done % 5000 == 0:
                print(f"  {done:,}/{len(pairs):,}  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print(f"Wrote {OUT}  ({len(df):,} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
