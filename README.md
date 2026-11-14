# Sentinel-1-Only LULC Semantic Segmentation

Land-use / land-cover (LULC) semantic segmentation from **Sentinel-1 radar only** (no
optical bands), plus a prototype that maps how land cover at any location changes
across twelve months. Course project for *Predictive Analytics and Deep Learning
(Data Analytics 3)*, SRH Hochschule Heidelberg.

## Results (test set)

| Metric | 15 classes | 8 super-classes |
| --- | ---: | ---: |
| Overall accuracy | 41.2% | 58.0% |
| mean IoU | 0.250 | 0.358 |

- Best classes: Urban/Built-up and Evergreen Broadleaf Forest (the classes radar
  separates best).
- U-Net (ResNet-34) outperformed a DeepLabV3+ baseline.
- Munich case study agrees with ESA WorldCover (independent) on **58%** of pixels,
  matching the super-class test accuracy.

## Repository layout

```
prototype/            end-to-end prototype (GEE fetch -> model -> maps -> change -> sanity check)
  run.py              command-line runner
  app_gradio.py       web UI (Gradio)
  gee_fetch.py        Earth Engine access + monthly compositing
  model_infer.py      checkpoint loading + tiled inference
  analyze.py          per-month class % and change tables
  visualize.py        map grid, class-share chart, comparison figure
  sanity_check.py     ESA WorldCover super-class agreement
  classes.py          15-class scheme, colours, cross-taxonomy mapping
  prototype_demo.ipynb  interactive demo notebook
data_preparation/     scripts that build the training subset from SEN12MS
  scan_lc.py, build_subset.py, pack_subset.py, class_mapping.csv
notebooks/            training notebook (with outputs)
webui/                static demo front-end (sample data)
report/               the report (PDF)
figures/              figures used in the report / README
```

## Data and weights (hosted externally — too large for GitHub)

- **Dataset (SEN12MS):** https://mediatum.ub.tum.de/1474000 (Sentinel-1 `_s1` + MODIS
  land cover `_lc`). The processed training subset is rebuilt by the scripts in
  `data_preparation/`.
- **Trained weights:** `best_unet_resnet34_v2.pth` — see the Google Drive link in the
  project submission (not committed here; model files exceed GitHub's size limit).

## Setup

```bash
pip install -r prototype/requirements.txt
```

A free **Google Earth Engine** account + Cloud project is required for the prototype
(the first run opens a browser for authentication).

## Run the prototype

Command line:

```bash
python prototype/run.py --lat 48.1372 --lon 11.5756 --start 2023-01 \
    --weights path/to/best_unet_resnet34_v2.pth --ee-project YOUR_GCP_PROJECT \
    --tiles 2 --name munich
```

Web UI:

```bash
EE_PROJECT=YOUR_GCP_PROJECT WEIGHTS=path/to/best_unet_resnet34_v2.pth python prototype/app_gradio.py
```

## Method in brief

- **Input:** Sentinel-1 VV/VH backscatter (dB), expanded to 3 channels
  `[VV, VH, VV-VH]` so an ImageNet-pretrained encoder can be used.
- **Model:** U-Net (ResNet-34), class-weighted cross-entropy + Dice, gradient
  clipping, early stopping on validation mIoU.
- **Prototype:** monthly Sentinel-1 median composites from Google Earth Engine, run
  tile-by-tile with per-scene normalization (to bridge the SEN12MS-to-live domain
  shift), then per-class area %, month-to-month and total change, and a sanity check
  against an independent LULC map.

## Limitations

SAR carries less class-discriminating signal than optical imagery, the MODIS training
labels are coarse (500 m), and seasonal backscatter change can mimic real land-use
change. See the report for the full discussion.
