"""End-to-end LULC prototype.

Input : latitude/longitude + start month.
Steps : fetch 12 monthly Sentinel-1 composites (GEE) -> run the trained model
        -> 12 monthly LULC maps -> class %/change -> figures -> WorldCover check.

Example:
    python run.py --lat 48.1372 --lon 11.5756 --start 2023-01 \
        --weights best_unet_resnet34.pth --ee-project my-gcp-project \
        --tiles 2 --name munich
"""
import os
import json
import argparse
import numpy as np

import gee_fetch
import analyze
import visualize
import sanity_check
from classes import SUPER


def parse_args():
    p = argparse.ArgumentParser(description="S1-only LULC 12-month prototype")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--start", required=True, help="start month YYYY-MM")
    p.add_argument("--weights", required=True, help="trained checkpoint .pth")
    p.add_argument("--ee-project", required=True, help="Google Cloud project id for EE")
    p.add_argument("--tiles", type=int, default=2, help="NxN tiles of 256 px (10 m) -> area")
    p.add_argument("--name", default="run", help="output folder name")
    p.add_argument("--out", default="outputs")
    p.add_argument("--no-sanity", action="store_true", help="skip WorldCover check")
    return p.parse_args()


def main():
    args = parse_args()
    year, month = (int(x) for x in args.start.split("-"))
    size_px = 256 * args.tiles
    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Location ({args.lat},{args.lon})  start {year}-{month:02d}  "
          f"area {size_px*10/1000:.2f} km  -> {out_dir}")

    # 1. Earth Engine: 12 monthly S1 composites
    gee_fetch.init_ee(args.ee_project)
    print("Fetching Sentinel-1 monthly composites ...")
    months, stack = gee_fetch.fetch_year(args.lat, args.lon, year, month, size_px)
    np.save(os.path.join(out_dir, "s1_stack_db.npy"), stack)

    # 2. Inference (import here so GEE works even without torch installed)
    from model_infer import LULCModel
    print(f"Running model on {args.weights} ...")
    model = LULCModel(args.weights)
    labels = model.predict_series(stack)                 # (12, S, S)
    np.save(os.path.join(out_dir, "lulc_maps.npy"), labels)

    # 3. Percentages + change
    pct = analyze.class_percentages(labels, months)
    m2m, total = analyze.change_tables(pct)
    pct.to_csv(os.path.join(out_dir, "class_percentages.csv"))
    m2m.to_csv(os.path.join(out_dir, "month_to_month_change_pp.csv"))
    total.to_csv(os.path.join(out_dir, "total_change_pp.csv"))
    print("\n" + analyze.summarize(total))

    # 4. Visualizations
    visualize.maps_grid(labels, months, os.path.join(out_dir, "lulc_maps.png"),
                        title=f"Monthly LULC - {args.name}")
    visualize.share_over_time(pct, os.path.join(out_dir, "class_share.png"),
                              title=f"Class share over 12 months - {args.name}")

    # 5. Sanity check vs ESA WorldCover (independent, not used in training)
    report = {"location": [args.lat, args.lon], "start": args.start,
              "area_km": size_px * 10 / 1000, "months": months}
    if not args.no_sanity:
        print("\nSanity check vs ESA WorldCover ...")
        esa = gee_fetch.fetch_esa_worldcover(args.lat, args.lon, size_px)
        model_super = sanity_check.model_to_super(labels[0])
        esa_super = sanity_check.esa_to_super(esa)
        overall, cmp_df = sanity_check.agreement(model_super, esa_super)
        cmp_df.to_csv(os.path.join(out_dir, "worldcover_comparison.csv"), index=False)
        visualize.comparison_maps(labels[0], esa_super, model_super, SUPER,
                                  os.path.join(out_dir, "worldcover_comparison.png"))
        report["worldcover_overall_agreement"] = overall
        print(f"Overall super-class agreement (month 1): {overall:.3f}")
        print(cmp_df.to_string(index=False))

    json.dump(report, open(os.path.join(out_dir, "run_report.json"), "w"), indent=2, default=str)
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
