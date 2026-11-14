"""Assemble the prototype demo notebook (good for the screen-recording)."""
import json, os

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md("""# LULC Prototype Demo - Sentinel-1 only, 12 monthly maps

Runs the full prototype interactively for one location (good for the required
screen recording). Needs: a Google Earth Engine account + Cloud project, and the
trained weights `best_*.pth` on Drive.

Put the `prototype/` folder on Drive (or upload it) and set `PROTO_DIR` below.""")

code("""
!pip -q install earthengine-api segmentation-models-pytorch==0.3.4 rasterio
from google.colab import drive; drive.mount('/content/drive')
import sys
PROTO_DIR = '/content/drive/MyDrive/sen12ms/prototype'   # <-- folder with run.py etc.
sys.path.append(PROTO_DIR)
""")

md("## Parameters (edit for your urban area)")
code("""
LAT, LON   = 48.1372, 11.5756          # Munich city centre (urban example)
START      = '2023-01'                 # YYYY-MM
WEIGHTS    = '/content/drive/MyDrive/sen12ms/checkpoints/best_unet_resnet34.pth'
EE_PROJECT = 'your-gcp-project-id'     # <-- your Earth Engine Cloud project
TILES      = 2                         # 2 -> 5.12 km window
NAME       = 'munich'
year, month = (int(x) for x in START.split('-'))
size_px = 256 * TILES
""")

md("## 1. Earth Engine auth + fetch 12 monthly Sentinel-1 composites")
code("""
import gee_fetch, numpy as np
gee_fetch.init_ee(EE_PROJECT)          # first run: follow the auth link
months, stack = gee_fetch.fetch_year(LAT, LON, year, month, size_px)
print('stack', stack.shape)
""")

md("## 2. Run the trained model -> 12 LULC maps")
code("""
from model_infer import LULCModel
model = LULCModel(WEIGHTS)
labels = model.predict_series(stack)
print('labels', labels.shape)
""")

md("## 3. Class percentages + change (percentage points)")
code("""
import analyze
pct = analyze.class_percentages(labels, months)
m2m, total = analyze.change_tables(pct)
display(pct.round(2))
print(analyze.summarize(total))
display(total.round(2))
""")

md("## 4. Visualize: monthly LULC maps + class share over time")
code("""
import visualize
from IPython.display import Image, display
visualize.maps_grid(labels, months, 'maps.png', title=f'Monthly LULC - {NAME}')
visualize.share_over_time(pct, 'share.png', title=f'Class share - {NAME}')
display(Image('maps.png')); display(Image('share.png'))
""")

md("## 5. Sanity check vs ESA WorldCover (independent, not used in training)")
code("""
import sanity_check
from classes import SUPER
esa = gee_fetch.fetch_esa_worldcover(LAT, LON, size_px)
model_super = sanity_check.model_to_super(labels[0])
esa_super   = sanity_check.esa_to_super(esa)
overall, cmp_df = sanity_check.agreement(model_super, esa_super)
print('Overall super-class agreement (month 1): %.3f' % overall)
display(cmp_df.round(2))
visualize.comparison_maps(labels[0], esa_super, model_super, SUPER, 'cmp.png')
display(Image('cmp.png'))
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(os.path.dirname(__file__), "prototype_demo.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
print("wrote", out, "cells:", len(cells))
