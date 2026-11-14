"""Gradio web UI for the Sentinel-1-only LULC prototype (cinematic theme).

Wired to the real pipeline: lat/lon + start month -> 12 monthly Sentinel-1
composites (Earth Engine) -> trained model -> monthly LULC maps, class
percentages, change, and an ESA WorldCover sanity check.

Run locally:
    EE_PROJECT=my-gcp-project WEIGHTS=/path/best_unet_resnet34.pth python app_gradio.py

In Colab (reuses an already-authenticated session):
    from app_gradio import build_ui
    build_ui(WEIGHTS).launch(share=True)
"""
import os
import tempfile

import numpy as np
import gradio as gr

import gee_fetch
import analyze
import visualize
import sanity_check
from model_infer import LULCModel
from classes import SUPER

# ---------------------------------------------------------------- look & feel
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
).set(
    body_background_fill="#060910",
    body_text_color="#e8eef8",
    block_background_fill="#0c1322",
    block_border_color="#1d2b45",
    block_label_text_color="#8aa0c2",
    block_title_text_color="#e8eef8",
    input_background_fill="#070d18",
    input_border_color="#1d2b45",
    button_primary_background_fill="linear-gradient(100deg, #2fe6c8, #4aa8ff)",
    button_primary_background_fill_hover="linear-gradient(100deg, #4aa8ff, #2fe6c8)",
    button_primary_text_color="#042019",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&display=swap');
.gradio-container{max-width:1120px !important;
  background-image:radial-gradient(1000px 500px at 85% -12%,rgba(47,230,200,.10),transparent 60%),
                   radial-gradient(820px 440px at -5% 4%,rgba(74,168,255,.08),transparent 55%);}
#hero{padding:26px 4px 6px;}
#hero .eyebrow{font-family:'IBM Plex Mono',monospace;letter-spacing:.22em;text-transform:uppercase;
  color:#2fe6c8;font-size:12px;}
#hero h1{font-family:'Chakra Petch','IBM Plex Sans',sans-serif;font-weight:700;
  font-size:clamp(30px,5.5vw,50px);line-height:1.03;margin:.15em 0;color:#e8eef8;
  animation:heroin .7s cubic-bezier(.2,.7,.2,1) both;}
#hero h1 .grad{background:linear-gradient(100deg,#2fe6c8,#4aa8ff);
  -webkit-background-clip:text;background-clip:text;color:transparent;}
#hero p{color:#8aa0c2;max-width:64ch;font-size:15px;}
@keyframes heroin{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
#runbtn{font-family:'Chakra Petch',sans-serif !important;font-weight:600 !important;
  letter-spacing:.02em;box-shadow:0 6px 24px rgba(47,230,200,.25);}
.block .label-wrap span,label span{font-family:'IBM Plex Mono',monospace;letter-spacing:.06em;}
footer{display:none !important;}
"""

HERO = """
<div id="hero">
  <div class="eyebrow">Predictive Analytics &amp; Deep Learning · SAR Remote Sensing</div>
  <h1>Sentinel-1 <span class="grad">LULC</span> Explorer</h1>
  <p>Land-use / land-cover from <b>Sentinel-1 radar only</b> — enter a location and a
  start month to map 12 monthly composites, track how each class shifts, and
  sanity-check against an independent reference.</p>
</div>
"""


def _run(model, lat, lon, start, tiles):
    year, month = (int(x) for x in str(start).split("-"))
    size_px = 256 * int(tiles)
    months, stack = gee_fetch.fetch_year(float(lat), float(lon), year, month, size_px)
    labels = model.predict_series(stack)              # per-scene norm built in

    pct = analyze.class_percentages(labels, months)
    _, total = analyze.change_tables(pct)

    d = tempfile.mkdtemp()
    maps_p = os.path.join(d, "maps.png")
    share_p = os.path.join(d, "share.png")
    cmp_p = os.path.join(d, "cmp.png")
    visualize.maps_grid(labels, months, maps_p, title=f"Monthly LULC ({lat:.3f}, {lon:.3f})")
    visualize.share_over_time(pct, share_p, title="Class share over 12 months")

    esa = gee_fetch.fetch_esa_worldcover(float(lat), float(lon), size_px)
    model_super = sanity_check.model_to_super(labels[0])
    esa_super = sanity_check.esa_to_super(esa)
    overall, _ = sanity_check.agreement(model_super, esa_super)
    visualize.comparison_maps(labels[0], esa_super, model_super, SUPER, cmp_p)

    agree = f"{overall * 100:.1f}%  super-class agreement vs ESA WorldCover (month 1)"
    return (maps_p, share_p,
            pct.round(1).reset_index(),
            total.round(2).reset_index(),
            agree, cmp_p)


def build_ui(weights):
    model = LULCModel(weights)
    with gr.Blocks(title="Sentinel-1 LULC Explorer", theme=THEME, css=CSS) as demo:
        gr.HTML(HERO)
        with gr.Row():
            lat = gr.Number(value=48.1372, label="Latitude")
            lon = gr.Number(value=11.5756, label="Longitude")
            start = gr.Textbox(value="2023-01", label="Start month (YYYY-MM)")
            tiles = gr.Dropdown(choices=["1", "2", "3"], value="2",
                                label="Window (x 2.56 km)")
        btn = gr.Button("Run 12-month analysis", variant="primary", elem_id="runbtn")
        with gr.Row():
            maps_out = gr.Image(label="Monthly LULC maps")
        with gr.Row():
            share_out = gr.Image(label="Class share over 12 months")
        agree_out = gr.Textbox(label="Sanity check", show_label=True)
        cmp_out = gr.Image(label="Model vs ESA WorldCover (super-classes)")
        with gr.Accordion("Data tables", open=False):
            pct_out = gr.Dataframe(label="Class % per month")
            total_out = gr.Dataframe(label="Total change (percentage points)")
        btn.click(lambda la, lo, s, t: _run(model, la, lo, s, t),
                  [lat, lon, start, tiles],
                  [maps_out, share_out, pct_out, total_out, agree_out, cmp_out])
    return demo


if __name__ == "__main__":
    weights = os.environ.get("WEIGHTS", "best_unet_resnet34.pth")
    project = os.environ.get("EE_PROJECT")
    if not project:
        raise SystemExit("Set EE_PROJECT (your Earth Engine Cloud project id).")
    gee_fetch.init_ee(project)
    build_ui(weights).launch(share=True)
