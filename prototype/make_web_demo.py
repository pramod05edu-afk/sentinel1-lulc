"""Precompute REAL model output for the static web demo (webui/).

The web demo is a static page (GitHub Pages), so it cannot call Earth Engine or
run PyTorch itself. This script runs the full pipeline -- GEE Sentinel-1 monthly
composites -> trained U-Net -> per-month LULC maps -> ESA WorldCover check --
for a few fixed locations and writes one small JSON file per location into
webui/data/, plus an index.json the page reads.

Run where Earth Engine and the checkpoint are available (e.g. Colab):

    python make_web_demo.py --weights /path/best_unet_resnet34_v2.pth \
        --ee-project YOUR_GCP_PROJECT --out ../webui/data

Each location takes a few minutes (12 monthly downloads + inference).
"""
import argparse
import base64
import json
import os
from datetime import date

import numpy as np

from classes import N_CLASSES, LABEL2SUPER

LOCATIONS = [
    dict(name="munich", label="Munich", kind="urban", lat=48.1372, lon=11.5756),
    dict(name="mannheim", label="Mannheim", kind="city + Rhine", lat=49.4875, lon=8.4660),
    dict(name="magdeburg_boerde", label="Magdeburger Börde", kind="farmland",
         lat=52.0500, lon=11.2500),
    dict(name="black_forest", label="Black Forest", kind="forest", lat=48.0500, lon=8.1500),
]

WEB_DIM = 128   # maps are mode-pooled to WEB_DIM x WEB_DIM for the browser


def mode_pool(label_map, dim=WEB_DIM, n_classes=N_CLASSES + 1):
    """Downsample a label map by majority vote in each block (no class blending)."""
    h = label_map.shape[0]
    f = h // dim
    a = label_map[:f * dim, :f * dim].reshape(dim, f, dim, f).transpose(0, 2, 1, 3)
    a = a.reshape(dim, dim, f * f)
    counts = np.stack([(a == k).sum(-1) for k in range(n_classes)], -1)
    return counts.argmax(-1).astype(np.uint8)


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).decode()


def build_record(loc, months, labels, esa_super=None, area_km=None):
    """Turn full-resolution model output into the compact JSON the page renders.

    labels    : (12, S, S) uint8 model labels (0..14)
    esa_super : (S, S) int ESA WorldCover super-classes (255 = unmapped) or None
    Percentages and agreement are computed at FULL resolution; only the pictures
    are downsampled.
    """
    pct = []
    for m in labels:
        counts = np.bincount(m.ravel(), minlength=N_CLASSES)[:N_CLASSES]
        pct.append([round(float(v), 3) for v in 100.0 * counts / counts.sum()])

    rec = dict(
        name=loc["name"], label=loc["label"], kind=loc["kind"],
        lat=loc["lat"], lon=loc["lon"],
        area_km=area_km, months=[[int(y), int(m)] for y, m in months],
        dim=WEB_DIM, pct=pct,
        maps=[b64(mode_pool(m)) for m in labels],
        generated=date.today().isoformat(),
    )
    if esa_super is not None:
        model_super = LABEL2SUPER[labels[0]]
        valid = esa_super != 255
        rec["agreement"] = round(float((model_super[valid] == esa_super[valid]).mean()), 4)
        ref = np.where(valid, esa_super, 8).astype(np.uint8)      # 8 = "unmapped" for display
        rec["ref_super"] = b64(mode_pool(ref, n_classes=9))
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", required=True)
    p.add_argument("--ee-project", required=True)
    p.add_argument("--start", default="2023-01", help="start month YYYY-MM")
    p.add_argument("--tiles", type=int, default=2, help="NxN tiles of 256 px (10 m)")
    p.add_argument("--out", default=os.path.join("..", "webui", "data"))
    p.add_argument("--only", nargs="*", help="subset of location names to run")
    args = p.parse_args()

    import gee_fetch
    import sanity_check
    from model_infer import LULCModel

    os.makedirs(args.out, exist_ok=True)
    gee_fetch.init_ee(args.ee_project)
    model = LULCModel(args.weights)
    year, month = (int(x) for x in args.start.split("-"))
    size_px = 256 * args.tiles

    index = []
    for loc in LOCATIONS:
        if args.only and loc["name"] not in args.only:
            continue
        print(f"\n=== {loc['label']} ({loc['lat']}, {loc['lon']}) ===")
        months, stack = gee_fetch.fetch_year(loc["lat"], loc["lon"], year, month, size_px)
        if np.isnan(stack).all(axis=(1, 2, 3)).any():
            print("  WARNING: some months had no imagery before the first available one")
        labels = model.predict_series(stack)
        esa = gee_fetch.fetch_esa_worldcover(loc["lat"], loc["lon"], size_px)
        esa_super = sanity_check.esa_to_super(esa)
        rec = build_record(loc, months, labels, esa_super, area_km=size_px * 10 / 1000)
        with open(os.path.join(args.out, f"{loc['name']}.json"), "w") as f:
            json.dump(rec, f, separators=(",", ":"))
        print(f"  agreement with ESA WorldCover: {rec['agreement']:.1%}")
        index.append({k: rec[k] for k in ("name", "label", "kind", "lat", "lon", "agreement")})

    # merge with any locations already exported earlier
    idx_path = os.path.join(args.out, "index.json")
    if os.path.exists(idx_path):
        old = {d["name"]: d for d in json.load(open(idx_path))}
        old.update({d["name"]: d for d in index})
        index = [old[l["name"]] for l in LOCATIONS if l["name"] in old]
    json.dump(index, open(idx_path, "w"), indent=1)
    print(f"\nDone. Wrote {len(index)} location(s) to {args.out}")


if __name__ == "__main__":
    main()
