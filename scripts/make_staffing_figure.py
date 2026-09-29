"""The controller-presence figure, from the cached staffing panel.

    python scripts/make_staffing_figure.py

Two panels, because the result is a claim about a gradient and a claim about it being real:

1. Close-pair rate against separation gate, staffed and unstaffed, within field. The two
   curves converge as the gate widens, which is the shape the mechanism predicts.
2. The pooled ratio per gate against the placebo band from shuffled staffing labels, so the
   effect and the null it has to beat are in the same picture.

Colours follow the project's validated categorical palette, and both series are directly
labelled so identity never rests on colour alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pcas.analysis.staffing import (
    CRITERIA,
    placebo_ratios,
    standardised_comparison,
    stratified_rates,
)

UNSTAFFED = "#eb6834"
STAFFED = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"

GATE_LABELS = {
    "nmac": "500 ft\n100 ft",
    "proximity": "0.5 nm\n500 ft",
    "near_1nm": "1 nm\n1000 ft",
    "loose_2nm": "2 nm\n1000 ft",
    "wide_3nm": "3 nm\n2000 ft",
}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="artifacts/staffing/panel.parquet")
    parser.add_argument("--out", default="docs/figures/controller_presence.png")
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)

    rows = []
    for criterion in CRITERIA:
        metric = f"close_{criterion.name}"
        pooled = standardised_comparison(
            stratified_rates(panel, arm="tower", metric=metric, by_field=True)
        )
        rows.append({"gate": criterion.name, **pooled})
    sweep = pd.DataFrame(rows)

    placebo = placebo_ratios(panel, arm="tower", metric="close_near_1nm")
    placebo = [r for r in placebo if r == r]

    plt = _style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    positions = range(len(sweep))
    labels = [GATE_LABELS[g] for g in sweep["gate"]]

    left.plot(
        positions,
        sweep["rate_off_per_1k"],
        color=UNSTAFFED,
        linewidth=2,
        marker="o",
        markersize=5,
        label="no tower",
    )
    left.plot(
        positions,
        sweep["rate_on_per_1k"],
        color=STAFFED,
        linewidth=2,
        marker="o",
        markersize=5,
        label="tower online",
    )
    left.set_yscale("log")
    left.set_xticks(list(positions))
    left.set_xticklabels(labels, fontsize=8)
    left.set_xlabel("separation gate (horizontal / vertical)")
    left.set_ylabel("pair samples inside the gate, per 1,000")
    left.set_title("Same field, same traffic, same hour", fontsize=10)
    left.grid(True, alpha=0.5, linewidth=0.6)
    left.legend(frameon=False, fontsize=8.5)

    right.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
    if placebo:
        right.axhspan(
            min(placebo), max(placebo), color=MUTED, alpha=0.16, label="placebo (labels shuffled)"
        )
    right.plot(
        positions,
        sweep["ratio_on_over_off"],
        color=STAFFED,
        linewidth=2,
        marker="o",
        markersize=5,
        label="measured",
    )
    right.set_ylim(0, 1.35)
    right.set_xticks(list(positions))
    right.set_xticklabels(labels, fontsize=8)
    right.set_xlabel("separation gate (horizontal / vertical)")
    right.set_ylabel("rate with a tower / rate without")
    right.set_title("The effect fades as the gate widens", fontsize=10)
    right.grid(True, alpha=0.5, linewidth=0.6)
    right.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
