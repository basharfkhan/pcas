"""Regenerate the README's figures from saved results.

    python scripts/make_figures.py --compute-horizons   # one inference pass, writes a CSV
    python scripts/make_figures.py                      # plots from the CSVs

Three figures, each answering a question the tables answer slowly:

1. `detection_vs_false_alarms.png` - the headline. Detection against false alarms, one panel
   per lead-time band, so where each method is worth using is visible at a glance.
2. `reliability.png` - the stated conflict probability against how often conflicts followed,
   before and after calibration, against the diagonal.
3. `error_vs_horizon.png` - trajectory error as a function of how far ahead, which is where
   the physics assumption visibly dies.

Colours come from a validated categorical palette (see the project notes); series are also
directly labelled, so identity never rests on colour alone.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("pcas.figures")

# Validated categorical slots: blue, orange, aqua, violet.
COLORS = {
    "closure rate": "#2a78d6",
    "Kalman": "#eb6834",
    "LSTM": "#1baf7a",
    "Transformer": "#4a3aa7",
}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"

BANDS = ["det_(0, 30]", "det_(30, 60]", "det_(60, 90]", "det_(90, 120]"]
BAND_TITLES = ["0 to 30 s ahead", "30 to 60 s ahead", "60 to 90 s ahead", "90 to 120 s ahead"]


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


def figure_detection(artifacts: Path, out: Path) -> None:
    """Detection against false alarms, per lead-time band."""
    plt = _style()

    mm = pd.read_csv(artifacts / "tradeoff_mm.csv")
    lstm = pd.read_csv(artifacts / "tradeoff_tartan.csv")

    series = [
        ("closure rate", mm[mm.method == "closure_rate"]),
        ("Kalman", mm[mm.method == "kalman_cv_horizon"]),
        ("LSTM", lstm[lstm.method == "lstm_horizon"]),
        ("Transformer", mm[mm.method == "transformer_probability"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), sharey=True)
    for ax, band, title in zip(axes, BANDS, BAND_TITLES, strict=True):
        for name, frame in series:
            data = (
                frame[["false_alarms_per_hour", band]].dropna().sort_values("false_alarms_per_hour")
            )
            if data.empty:
                continue
            ax.plot(
                data["false_alarms_per_hour"],
                data[band],
                color=COLORS[name],
                linewidth=2,
                marker="o",
                markersize=4,
                label=name,
            )
        ax.set_xscale("log")
        ax.set_xlim(0.15, 200)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.5, linewidth=0.6)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("false alarms per hour")

    axes[0].set_ylabel("conflicts detected")
    # A shaded band for the operating region a real system would live in.
    for ax in axes:
        ax.axvspan(0.15, 2.0, color="#2a78d6", alpha=0.06, linewidth=0)
    axes[0].text(0.22, 0.94, "usable\nbudget", fontsize=7.5, color=MUTED, va="top")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Detection against false alarms, by how far ahead the conflict was",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    log.info("wrote %s", out)


def figure_reliability(artifacts: Path, out: Path) -> None:
    """Stated conflict probability against observed frequency, before and after calibration."""
    plt = _style()
    pairs_path = artifacts / "calibration_pairs.csv"
    if not pairs_path.exists():
        log.warning("%s missing; run scripts/calibrate.py first", pairs_path)
        return

    pairs = pd.read_csv(pairs_path)
    bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    def curve(column: str):
        binned = pd.cut(pairs[column], bins=bins, include_lowest=True)
        grouped = pairs.groupby(binned, observed=True).agg(
            stated=(column, "mean"), observed=("event", "mean"), n=("event", "size")
        )
        return grouped.reset_index(drop=True)

    before, after = curve("stated"), curve("calibrated")

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), label="perfect")

    for frame, name, color in (
        (before, "raw", COLORS["Kalman"]),
        (after, "calibrated", COLORS["Transformer"]),
    ):
        ax.plot(
            frame["stated"],
            frame["observed"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=7,
            label=name,
        )

    # Direct labels, so identity does not rest on colour.
    if not before.empty:
        row = before.iloc[-1]
        ax.annotate(
            "raw: says 93%,\nhappens 70%",
            (row["stated"], row["observed"]),
            textcoords="offset points",
            xytext=(-118, -150),
            fontsize=8,
            color=COLORS["Kalman"],
        )
    if not after.empty:
        row = after.iloc[-1]
        ax.annotate(
            "calibrated",
            (row["stated"], row["observed"]),
            textcoords="offset points",
            xytext=(-66, 12),
            fontsize=8,
            color=COLORS["Transformer"],
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("stated probability of conflict")
    ax.set_ylabel("conflicts that actually followed")
    ax.set_title("Is a stated 50% chance really 50%?", fontsize=11)
    ax.grid(True, alpha=0.5, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    log.info("wrote %s", out)


def figure_horizon(artifacts: Path, out: Path) -> None:
    """Trajectory error against how far ahead the prediction reaches."""
    plt = _style()
    path = artifacts / "horizon_errors.csv"
    if not path.exists():
        log.warning("%s missing; run with --compute-horizons first", path)
        return

    table = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    for name, frame in table.groupby("model"):
        frame = frame.sort_values("horizon_s")
        ax.plot(
            frame["horizon_s"],
            frame["median_m"],
            color=COLORS.get(name, MUTED),
            linewidth=2,
            marker="o",
            markersize=4,
            label=name,
        )
        last = frame.iloc[-1]
        ax.annotate(
            f"{name}  {last['median_m']:.0f} m",
            (last["horizon_s"], last["median_m"]),
            textcoords="offset points",
            xytext=(6, -2),
            fontsize=8,
            color=COLORS.get(name, MUTED),
        )

    ax.set_xlabel("seconds ahead")
    ax.set_ylabel("median position error (m)")
    ax.set_title("Where dead reckoning stops working", fontsize=11)
    ax.set_xlim(0, 155)
    ax.grid(True, alpha=0.5, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    log.info("wrote %s", out)


def compute_horizons(args) -> None:
    """One inference pass over the test sessions, writing per-horizon errors."""
    from pcas.data.adsb import day_to_scenes
    from pcas.data.scenes import OBS_LEN, build_windows, future_offsets
    from pcas.data.sources import iter_days, open_source
    from pcas.eval.metrics import displacement
    from pcas.models.baselines import KalmanConstantVelocity
    from pcas.models.load import load_predictor

    horizons = np.array(future_offsets()) - (OBS_LEN - 1)
    models: list[tuple[str, object]] = [("Kalman", KalmanConstantVelocity())]
    for name, checkpoint in (("LSTM", args.lstm), ("Transformer", args.transformer)):
        if checkpoint and Path(checkpoint).exists():
            models.append((name, load_predictor(checkpoint)))
        else:
            log.warning("skipping %s: %s not found", name, checkpoint)

    source = open_source(args.source, args.root, args.airport)
    _, test_dates = source.split(args.test_days)
    errors: dict[str, list[np.ndarray]] = {name: [] for name, _ in models}

    for day in iter_days(source, test_dates):
        windows = []
        for scene in day_to_scenes(day, min_agents=1):
            windows += build_windows(scene, stride=args.stride, min_agents=1, include_dense=False)

        for window in windows:
            for name, model in models:
                if hasattr(model, "with_wind"):
                    model = model.with_wind(window.wind)
                pred = model.predict(window.obs, horizons)
                # Best hypothesis per aircraft, which is what minADE/minFDE report.
                errors[name].append(displacement(pred, window.future).min(axis=1))
        log.info("%s: %d windows", day.date, len(windows))

    rows = []
    for name, chunks in errors.items():
        if not chunks:
            continue
        stacked = np.concatenate(chunks)
        for i, horizon in enumerate(horizons):
            rows.append(
                {
                    "model": name,
                    "horizon_s": float(horizon),
                    "mean_m": float(stacked[:, i].mean()),
                    "median_m": float(np.median(stacked[:, i])),
                    "p95_m": float(np.percentile(stacked[:, i], 95)),
                    "n": int(len(stacked)),
                }
            )

    out = Path(args.artifacts) / "horizon_errors.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("wrote %s", out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="docs/figures")
    parser.add_argument("--compute-horizons", action="store_true")
    parser.add_argument("--source", choices=["trajair", "tartan"], default="tartan")
    parser.add_argument("--root", default="data/tartan")
    parser.add_argument("--airport", default="kbtp")
    parser.add_argument("--test-days", type=int, default=40)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--lstm", default="artifacts/lstm_tartan/lstm.pt")
    parser.add_argument("--transformer", default="artifacts/transformer_mm/model.pt")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.compute_horizons:
        compute_horizons(args)
        return 0

    artifacts = Path(args.artifacts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    figure_detection(artifacts, out / "detection_vs_false_alarms.png")
    figure_reliability(artifacts, out / "reliability.png")
    figure_horizon(artifacts, out / "error_vs_horizon.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
