"""Sanity-check a model LULC map against ESA WorldCover (not used in training).

Both maps are reduced to 8 coarse super-classes (the only fair comparison, since
IGBP and WorldCover taxonomies differ), then per-super-class agreement and an
overall pixel-agreement score are computed.
"""
import numpy as np
import pandas as pd

from classes import SUPER, LABEL2SUPER, ESA2SUPER


def model_to_super(label_map):
    return LABEL2SUPER[label_map]


def esa_to_super(esa_map):
    """Map ESA codes to super-class index; unmapped (e.g. snow) -> 255."""
    out = np.full(esa_map.shape, 255, np.int64)
    for code, sup in ESA2SUPER.items():
        if sup is not None:
            out[esa_map == code] = sup
    return out


def agreement(model_super, esa_super):
    """Overall agreement + per-super-class area shares (model vs ESA)."""
    valid = esa_super != 255
    overall = float((model_super[valid] == esa_super[valid]).mean())

    n = len(SUPER)
    rows = []
    for s in range(n):
        m_share = 100.0 * (model_super == s).mean()
        e_share = 100.0 * ((esa_super == s) & valid).mean()
        rows.append((SUPER[s], m_share, e_share, m_share - e_share))
    df = pd.DataFrame(rows, columns=["super_class", "model_%", "esa_%", "diff_pp"])
    return overall, df
