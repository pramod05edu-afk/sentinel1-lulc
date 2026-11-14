"""Visualizations: 12-month LULC map grid, legend, and class-share over time."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from classes import CLASS_NAMES, CLASS_COLORS, N_CLASSES, MONTH_NAMES

_CMAP = ListedColormap(CLASS_COLORS)
_NORM = BoundaryNorm(np.arange(-0.5, N_CLASSES + 0.5, 1), N_CLASSES)


def _legend_handles():
    return [Patch(facecolor=CLASS_COLORS[i], label=f"{i} {CLASS_NAMES[i]}")
            for i in range(N_CLASSES)]


def maps_grid(label_series, months, out_path, title="Monthly LULC"):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i, ax in enumerate(axes.ravel()):
        if i < label_series.shape[0]:
            ax.imshow(label_series[i], cmap=_CMAP, norm=_NORM, interpolation="nearest")
            y, m = months[i]
            ax.set_title(f"{y}-{m:02d} {MONTH_NAMES[m - 1]}", fontsize=11)
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    fig.legend(handles=_legend_handles(), loc="lower center",
               ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def share_over_time(pct, out_path, title="Class share over 12 months"):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(range(len(pct)), pct.values.T, labels=CLASS_NAMES, colors=CLASS_COLORS)
    ax.set_xticks(range(len(pct)))
    ax.set_xticklabels([ix.split()[-1] for ix in pct.index], rotation=45, ha="right")
    ax.set_ylabel("% of area"); ax.set_ylim(0, 100); ax.set_xlim(0, len(pct) - 1)
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def comparison_maps(pred_map, esa_super, model_super, super_names, out_path):
    """Side-by-side model vs ESA WorldCover, both reduced to super-classes."""
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(model_super, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    ax[0].set_title("Model (super-classes)"); ax[0].axis("off")
    ax[1].imshow(esa_super, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    ax[1].set_title("ESA WorldCover (super-classes)"); ax[1].axis("off")
    handles = [Patch(facecolor=cmap(i), label=f"{i} {n}") for i, n in enumerate(super_names)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
