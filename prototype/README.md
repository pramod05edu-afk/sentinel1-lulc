# Sentinel-1 LULC Prototype

End-to-end prototype for **Land Use / Land Cover (LULC) semantic segmentation
from Sentinel-1 radar only**. Given a location and a start month, it pulls 12
monthly Sentinel-1 composites, runs the trained deep-learning model, and reports
the monthly LULC maps, per-class area percentages, and the month-to-month and
total change.

## What it does

1. **Input** — latitude/longitude + start month (`YYYY-MM`).
2. **Fetch** — for each of 12 consecutive months, builds a mean Sentinel-1
   (VV+VH, dB) composite from Google Earth Engine (`COPERNICUS/S1_GRD`).
3. **Infer** — runs the trained model (3-channel `[VV, VH, VV−VH]`, the same
   standardization used in training, stored inside the checkpoint) to produce
   one LULC map per month.
4. **Calculate** — per-month percentage of area per class; month-to-month change
   and total first→last change, both in **percentage points**.
5. **Visualize** — a 12-panel monthly LULC map grid and a class-share-over-time
   chart.
6. **Sanity check** — compares month 1 against **ESA WorldCover** (independent,
   not used in training), reduced to common super-classes.

## Requirements

- Python 3.10+
- Packages in `requirements.txt` (`pip install -r requirements.txt`)
- **A trained checkpoint** `best_<arch>_<encoder>.pth` from the training
  notebook (holds the model weights **and** `ch_mean`/`ch_std`).

## Accounts / API keys

- **Google Earth Engine** account (free, non-commercial): sign up at
  <https://earthengine.google.com/> and create/attach a **Google Cloud project**.
  The first `init_ee()` call opens a browser for one-time authentication; pass
  the Cloud **project id** via `--ee-project`.
- No other API keys. All data (Sentinel-1, ESA WorldCover) comes from GEE.

## Usage

```bash
pip install -r requirements.txt

python run.py \
  --lat 48.1372 --lon 11.5756 \
  --start 2023-01 \
  --weights /path/to/best_unet_resnet34.pth \
  --ee-project my-gcp-project \
  --tiles 2 --name munich
```

`--tiles N` sets the analysis window to an N×N grid of 256-pixel (10 m) tiles
(`N=2` → 5.12 km). Add `--no-sanity` to skip the WorldCover comparison.

For an interactive run (recommended for the screen recording), use
`prototype_demo.ipynb` on Google Colab.

## Web UI (Gradio)

`app_gradio.py` is a browser UI wired to the real pipeline: enter a lat/lon and
start month and it runs the 12-month Sentinel-1 -> model -> maps analysis live,
with the class-share chart and the ESA WorldCover sanity check.

Run locally:

```bash
EE_PROJECT=my-gcp-project WEIGHTS=/path/best_unet_resnet34.pth python app_gradio.py
```

In Colab (reuses the already-authenticated Earth Engine session and avoids a
second auth), run in a cell:

```python
!pip -q install gradio
from app_gradio import build_ui
build_ui(WEIGHTS).launch(share=True)   # prints a public *.gradio.live link
```

## Outputs (`outputs/<name>/`)

| File | Contents |
|------|----------|
| `lulc_maps.png` | 12-month LULC map grid |
| `class_share.png` | stacked class share over 12 months |
| `class_percentages.csv` | % area per class, per month |
| `month_to_month_change_pp.csv` | consecutive-month change (pp) |
| `total_change_pp.csv` | first→last change per class (pp) |
| `worldcover_comparison.{png,csv}` | model vs ESA WorldCover (super-classes) |
| `lulc_maps.npy`, `s1_stack_db.npy` | raw label maps and input composites |
| `run_report.json` | run metadata + overall agreement |

## Files

- `run.py` — CLI orchestration
- `gee_fetch.py` — Earth Engine access + monthly compositing + tile download
- `model_infer.py` — checkpoint loading, standardization, tiled inference
- `analyze.py` — percentages and change tables
- `visualize.py` — map grid, class-share chart, comparison figure
- `sanity_check.py` — ESA WorldCover super-class agreement
- `classes.py` — 15-class scheme, colors, cross-taxonomy mapping

## Notes / caveats (for the report)

- GEE's `S1_GRD` is already sigma-nought in **dB**, so no extra
  calibration/terrain-correction is applied here; monthly compositing averages
  in **linear power** then converts back to dB.
- Months with no Sentinel-1 acquisition are carried forward from the previous
  available month (flagged in the log).
- The WorldCover check is only valid at the **super-class** level because IGBP
  (training labels) and WorldCover use different taxonomies.
- Model and WorldCover differ in date (WorldCover v200 = 2021), so some
  disagreement is expected and is not necessarily model error.
