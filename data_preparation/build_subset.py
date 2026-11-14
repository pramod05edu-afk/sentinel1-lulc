"""
Build a class-balanced, scene-stratified subset index from index.parquet.

- 15-class scheme: full IGBP minus the two absent classes (3 Deciduous
  Needleleaf, 15 Snow & Ice), which are mapped to ignore (255).
- Split is by SCENE (season+scene), so no scene appears in two splits.
- Within each split, patches are balanced by dominant class (rare classes
  taken in full, common classes capped) and topped up to the target size.

Outputs:
    class_mapping.csv   IGBP id -> train label (0..14) + name
    subset_index.csv    selected patches: split, paths, dominant label
Run:  python build_subset.py
"""
import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
INDEX = os.path.join(HERE, "index.parquet")
SEED = 42

IGBP_NAME = {
    1: "Evergreen Needleleaf Forest", 2: "Evergreen Broadleaf Forest",
    3: "Deciduous Needleleaf Forest", 4: "Deciduous Broadleaf Forest",
    5: "Mixed Forest", 6: "Closed Shrublands", 7: "Open Shrublands",
    8: "Woody Savannas", 9: "Savannas", 10: "Grasslands",
    11: "Permanent Wetlands", 12: "Croplands", 13: "Urban/Built-up",
    14: "Cropland/Natural Vegetation Mosaic", 15: "Snow & Ice",
    16: "Barren", 17: "Water",
}
DROP = [3, 15]                                     # absent -> ignore
KEEP = [c for c in range(1, 18) if c not in DROP]  # 15 classes
IGBP2LABEL = {c: i for i, c in enumerate(KEEP)}    # 0..14

# split and subset sizes
VAL_FRAC, TEST_FRAC = 0.15, 0.15
TARGETS = {"train": 8000, "val": 1500, "test": 2000}
CAPS = {"train": 650, "val": 130, "test": 170}     # per-dominant-class cap


def kept_px(df):
    return df[[f"px_{c}" for c in KEEP]].values


def dominant_label(df):
    """Train label (0..14) of the most frequent KEPT class in each patch."""
    px = kept_px(df)
    return px.argmax(1)   # index into KEEP == train label


def split_scenes(df):
    rng = np.random.default_rng(SEED)
    df = df.copy()
    df["skey"] = df.season + "_" + df.scene
    split = pd.Series("train", index=df.index)
    for season, g in df.groupby("season"):
        scenes = g.skey.unique()
        rng.shuffle(scenes)
        n = len(scenes)
        n_val = max(1, int(round(n * VAL_FRAC)))
        n_test = max(1, int(round(n * TEST_FRAC)))
        val_s = set(scenes[:n_val])
        test_s = set(scenes[n_val:n_val + n_test])
        split.loc[df.skey.isin(val_s)] = "val"
        split.loc[df.skey.isin(test_s)] = "test"
    df["split"] = split.values
    return df


def balanced_select(pool, target, cap):
    pool = pool.copy()
    pool["dom"] = dominant_label(pool)
    order = pool["dom"].value_counts().sort_values().index.tolist()  # rare first
    chosen = []
    for cls in order:
        sub = pool[pool.dom == cls]
        take = min(len(sub), cap)
        chosen.append(sub.sample(take, random_state=SEED))
    sel = pd.concat(chosen)
    if len(sel) > target:
        sel = sel.sample(target, random_state=SEED)
    elif len(sel) < target:
        remain = pool.drop(sel.index)
        extra = remain.sample(min(target - len(sel), len(remain)), random_state=SEED)
        sel = pd.concat([sel, extra])
    return sel


def main():
    df = pd.read_parquet(INDEX)
    df = split_scenes(df)

    # class mapping file
    rows = [{"igbp_id": c, "igbp_name": IGBP_NAME[c], "train_label": IGBP2LABEL[c]}
            for c in KEEP]
    rows += [{"igbp_id": c, "igbp_name": IGBP_NAME[c], "train_label": 255} for c in DROP]
    pd.DataFrame(rows).sort_values("igbp_id").to_csv(
        os.path.join(HERE, "class_mapping.csv"), index=False)

    parts = []
    print("scenes per split:")
    print(df.groupby("split")["season"].count(), "\n")
    for split in ["train", "val", "test"]:
        pool = df[df.split == split]
        sel = balanced_select(pool, TARGETS[split], CAPS[split])
        sel = sel.assign(split=split)
        parts.append(sel)
        print(f"{split}: pool {len(pool):,} -> selected {len(sel):,}")

    out = pd.concat(parts)[
        ["split", "season", "scene", "patch", "s1_path", "lc_path"] +
        [f"px_{c}" for c in KEEP]
    ]
    out.to_csv(os.path.join(HERE, "subset_index.csv"), index=False)

    # report per-class PIXEL coverage in the subset (presence across classes)
    print("\nper-class pixel share in selected subset:")
    for split in ["train", "val", "test"]:
        s = out[out.split == split]
        px = s[[f"px_{c}" for c in KEEP]].values.sum(0)
        tot = px.sum()
        print(f"\n[{split}] {len(s):,} patches")
        for i, c in enumerate(KEEP):
            npatch = (s[f"px_{c}"] > 0).sum()
            print("  lbl %2d %-34s %6.2f%%  (in %d patches)"
                  % (IGBP2LABEL[c], IGBP_NAME[c], 100 * px[i] / tot, npatch))

    print(f"\nWrote subset_index.csv ({len(out):,} patches), class_mapping.csv")


if __name__ == "__main__":
    main()
