"""Per-month class area percentages and change metrics."""
import numpy as np
import pandas as pd

from classes import CLASS_NAMES, N_CLASSES, MONTH_NAMES


def _month_label(months, i):
    y, m = months[i]
    return f"{y}-{m:02d} {MONTH_NAMES[m - 1]}"


def class_percentages(label_series, months):
    """(12,S,S) -> DataFrame [month x class] of % area per class (rows sum ~100)."""
    rows = []
    for i in range(label_series.shape[0]):
        counts = np.bincount(label_series[i].ravel(), minlength=N_CLASSES)[:N_CLASSES]
        rows.append(100.0 * counts / counts.sum())
    df = pd.DataFrame(rows, columns=CLASS_NAMES,
                      index=[_month_label(months, i) for i in range(len(months))])
    df.index.name = "month"
    return df


def change_tables(pct):
    """Month-to-month change and total first->last change, in percentage points."""
    m2m = pct.diff().iloc[1:].copy()
    m2m.index = [f"{a} -> {b}" for a, b in zip(pct.index[:-1], pct.index[1:])]
    m2m.index.name = "transition"

    total = (pct.iloc[-1] - pct.iloc[0]).to_frame("total_change_pp")
    total["start_%"] = pct.iloc[0].values
    total["end_%"] = pct.iloc[-1].values
    total = total[["start_%", "end_%", "total_change_pp"]]
    total.index.name = "class"
    return m2m, total


def summarize(total, top=5):
    """Short text summary of the largest absolute total changes."""
    s = total["total_change_pp"].reindex(
        total["total_change_pp"].abs().sort_values(ascending=False).index)
    lines = ["Largest first->last changes (percentage points):"]
    for name, v in s.head(top).items():
        if abs(v) < 1e-6:
            continue
        lines.append(f"  {name:<32} {v:+6.2f} pp")
    return "\n".join(lines)
