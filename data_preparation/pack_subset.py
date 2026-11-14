"""
Pack the selected subset (subset_index.csv) into compact arrays for Colab.

Per split writes:
    s1_<split>.npy   float16 (N, 2, 256, 256)   VV, VH backscatter in dB
    lbl_<split>.npy  uint8   (N, 256, 256)       labels 0..14, 255 = ignore
    meta_<split>.csv paths + dominant label

Label remapping (IGBP -> 0..14, with absent classes 3 & 15 -> 255) is baked in,
so the Colab notebook needs no rasterio and no remapping.

Run:  python pack_subset.py
"""
import os
import time
import numpy as np
import pandas as pd
import rasterio
from numpy.lib.format import open_memmap
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(__file__)
OUTDIR = r"D:\sen12ms_subset"
os.makedirs(OUTDIR, exist_ok=True)

idx = pd.read_csv(os.path.join(HERE, "subset_index.csv"))
cmap = pd.read_csv(os.path.join(HERE, "class_mapping.csv"))

# IGBP id (0..17) -> train label, default 255 (ignore)
LUT = np.full(18, 255, dtype=np.uint8)
for _, r in cmap.iterrows():
    if r.train_label != 255:
        LUT[int(r.igbp_id)] = int(r.train_label)


def pack_split(split):
    rows = idx[idx.split == split].reset_index(drop=True)
    n = len(rows)
    s1 = open_memmap(os.path.join(OUTDIR, f"s1_{split}.npy"), mode="w+",
                     dtype=np.float16, shape=(n, 2, 256, 256))
    lbl = open_memmap(os.path.join(OUTDIR, f"lbl_{split}.npy"), mode="w+",
                      dtype=np.uint8, shape=(n, 256, 256))

    def fill(i):
        row = rows.iloc[i]
        with rasterio.open(row.s1_path) as ds:
            s1[i] = ds.read().astype(np.float16)      # (2,256,256)
        with rasterio.open(row.lc_path) as ds:
            igbp = ds.read(1)
        lbl[i] = LUT[igbp]

    t = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(fill, range(n)))
    s1.flush(); lbl.flush()
    rows[["split", "season", "scene", "patch", "s1_path", "lc_path"]].to_csv(
        os.path.join(OUTDIR, f"meta_{split}.csv"), index=False)
    print(f"{split}: {n:,} patches packed ({time.time()-t:.0f}s)")
    return n


def main():
    total = 0
    for split in ["train", "val", "test"]:
        total += pack_split(split)
    cmap.to_csv(os.path.join(OUTDIR, "class_mapping.csv"), index=False)
    # quick sanity on train labels
    lbl = np.load(os.path.join(OUTDIR, "lbl_train.npy"), mmap_mode="r")
    vals, cnts = np.unique(np.asarray(lbl[:200]), return_counts=True)
    print("train label values (first 200 patches):", dict(zip(vals.tolist(), cnts.tolist())))
    print(f"\nPacked {total:,} patches -> {OUTDIR}")


if __name__ == "__main__":
    main()
