"""Where the model is wrong, and whether its explanation holds up.

    python scripts/error_analysis.py

Four things, on the held-out sessions:

1. **Error by flight phase.** An average over every window hides the thing worth knowing: an
   aircraft transiting overhead is nearly free to predict, while one turning base is where
   conflicts happen.
2. **The mechanism test.** The ablation shows *that* seeing other aircraft helps. If the
   explanation offered for it is right, the advantage should *grow* with the number of
   neighbours. If it is flat, the explanation needs rewriting, so this is worth running even
   though it can only embarrass the story.
3. **Missed conflicts and false alarms, characterised** by phase and lead time.
4. **A failure gallery**: the worst predictions, drawn, so the failures can be described
   rather than summarised.

Writes CSVs to artifacts/ and figures to docs/figures/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pcas.data.adsb import day_to_scenes
from pcas.data.scenes import OBS_LEN, build_windows, future_offsets
from pcas.data.sources import iter_days, open_source
from pcas.eval.conflicts import PROXIMITY, first_violation, predicted_alerts
from pcas.eval.metrics import displacement
from pcas.eval.phases import classify
from pcas.models.baselines import KalmanConstantVelocity
from pcas.models.load import load_predictor

log = logging.getLogger("pcas.error_analysis")

COLORS = {
    "Kalman": "#eb6834",
    "no social": "#1baf7a",
    "social": "#4a3aa7",
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"
CRITICAL = "#d03b3b"


def collect(source, dates, models, multimodal, horizons, dense, stride):
    """Per-aircraft errors with phase labels, plus per-pair conflict outcomes."""
    per_aircraft, per_pair, worst = [], [], []

    for day in iter_days(source, dates):
        windows = []
        for scene in day_to_scenes(day, min_agents=1):
            windows += build_windows(scene, stride=stride, min_agents=1)

        for window in windows:
            phases = classify(window.obs, day.field_elev_m)
            n_neighbours = window.n_agents - 1

            errors = {}
            for name, model in models.items():
                predictor = model.with_wind(window.wind) if hasattr(model, "with_wind") else model
                pred = predictor.predict(window.obs, horizons)
                errors[name] = displacement(pred, window.future).min(axis=1)

            for i in range(window.n_agents):
                row = {
                    "scene_id": window.scene_id,
                    "date": window.date,
                    "start_frame": window.start_frame,
                    "agent": window.agent_ids[i],
                    "phase": phases[i],
                    "neighbours": n_neighbours,
                }
                for name in models:
                    row[f"fde_{name}"] = float(errors[name][i, -1])
                    row[f"ade_{name}"] = float(errors[name][i].mean())
                per_aircraft.append(row)

            # Keep the worst social-model predictions for the gallery.
            social_fde = errors["social"][:, -1]
            worst.append((float(social_fde.max()), int(social_fde.argmax()), window, phases))

            if window.n_agents < 2 or window.future_dense is None:
                continue

            predictor = multimodal.with_wind(window.wind)
            dense_pred = predictor.predict(window.obs, dense)
            weights = getattr(multimodal, "last_probabilities", None)
            alerts = predicted_alerts(
                dense_pred, dense, PROXIMITY, probability_threshold=0.0, mode_probabilities=weights
            )
            stated = {(int(r.i), int(r.j)): float(r.probability) for r in alerts.itertuples()}

            times = np.arange(1, window.future_dense.shape[1] + 1, dtype=float)
            truth = first_violation(window.future_dense, PROXIMITY, times)
            onsets = {
                (int(r.i), int(r.j)): float(r.onset_s)
                for r in truth.itertuples()
                if r.onset_s <= 120.0
            }

            for i in range(window.n_agents):
                for j in range(i + 1, window.n_agents):
                    per_pair.append(
                        {
                            "scene_id": window.scene_id,
                            "phase_i": phases[i],
                            "phase_j": phases[j],
                            "neighbours": n_neighbours,
                            "probability": stated.get((i, j), 0.0),
                            "event": (i, j) in onsets,
                            "onset_s": onsets.get((i, j)),
                        }
                    )

        log.info("%s: %d windows, %d aircraft rows", day.date, len(windows), len(per_aircraft))

    worst.sort(key=lambda item: -item[0])
    return pd.DataFrame(per_aircraft), pd.DataFrame(per_pair), worst[:8]


def _format(table: pd.DataFrame, precision: dict[str, int]) -> str:
    """Print metre columns whole and fraction columns at full precision."""
    shown = table.copy()
    for column in shown.columns:
        shown[column] = shown[column].round(precision.get(column, 0))
    return shown.to_string()


def report_phases(aircraft: pd.DataFrame, out: Path) -> pd.DataFrame:
    table = (
        aircraft.groupby("phase")
        .agg(
            n=("fde_social", "size"),
            kalman=("fde_Kalman", "median"),
            no_social=("fde_no social", "median"),
            social=("fde_social", "median"),
        )
        .sort_values("social")
    )
    table["social_vs_kalman"] = (1 - table["social"] / table["kalman"]).round(3)
    table.to_csv(out / "error_by_phase.csv")
    print("\nMEDIAN 120 s ERROR BY FLIGHT PHASE (metres)")
    print(_format(table, {"social_vs_kalman": 3}))
    return table


def report_mechanism(aircraft: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Does the social advantage grow with the number of neighbours?"""
    frame = aircraft.copy()
    frame["bucket"] = pd.cut(
        frame["neighbours"], [-1, 0, 1, 2, 4, 100], labels=["0", "1", "2", "3-4", "5+"]
    )
    table = frame.groupby("bucket", observed=True).agg(
        n=("fde_social", "size"),
        no_social=("fde_no social", "median"),
        social=("fde_social", "median"),
    )
    table["social_gain"] = (1 - table["social"] / table["no_social"]).round(3)
    table.to_csv(out / "error_by_neighbours.csv")
    print("\nSOCIAL GAIN BY NUMBER OF NEIGHBOURS (median 120 s error, metres)")
    print(_format(table, {"social_gain": 3}))
    return table


def report_conflicts(pairs: pd.DataFrame, threshold: float = 0.5) -> None:
    if pairs.empty:
        return
    pairs = pairs.copy()
    pairs["alerted"] = pairs["probability"] >= threshold

    missed = pairs[pairs["event"] & ~pairs["alerted"]]
    caught = pairs[pairs["event"] & pairs["alerted"]]
    false_alarms = pairs[~pairs["event"] & pairs["alerted"]]

    print(f"\nCONFLICT OUTCOMES at P >= {threshold}")
    print(f"  events {int(pairs['event'].sum())}, caught {len(caught)}, missed {len(missed)}")
    print(f"  false alarms {len(false_alarms)} of {int((~pairs['event']).sum())} quiet pairs")

    if not missed.empty:
        print("\n  missed conflicts, by phase pair (top 6):")
        combos = (
            missed.assign(pair=lambda d: d["phase_i"] + " / " + d["phase_j"])["pair"]
            .value_counts()
            .head(6)
        )
        print("   " + combos.to_string().replace("\n", "\n   "))
        print(f"\n  missed conflicts arrive at a median {missed['onset_s'].median():.0f} s;")
        print(f"  caught ones at a median {caught['onset_s'].median():.0f} s.")

    if not false_alarms.empty:
        print("\n  false alarms, by phase pair (top 6):")
        combos = (
            false_alarms.assign(pair=lambda d: d["phase_i"] + " / " + d["phase_j"])["pair"]
            .value_counts()
            .head(6)
        )
        print("   " + combos.to_string().replace("\n", "\n   "))


def figure_phases(table: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    phases = list(table.index)
    positions = np.arange(len(phases))
    width = 0.27

    for offset, (name, column) in zip(
        (-width, 0.0, width),
        (("Kalman", "kalman"), ("no social", "no_social"), ("social", "social")),
        strict=True,
    ):
        ax.bar(
            positions + offset,
            table[column],
            width * 0.92,
            label=name,
            color=COLORS[name],
            edgecolor=SURFACE,
            linewidth=1.2,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(phases, fontsize=9)
    ax.set_ylabel("median error 120 s ahead (m)", color=MUTED, fontsize=9)
    ax.set_title("The model earns its place in the circuit, not in cruise", fontsize=11, color=INK)
    ax.grid(True, axis="y", color=GRID, alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out / "error_by_phase.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out / "error_by_phase.png")


def figure_gallery(worst, multimodal, horizons, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
    fig.patch.set_facecolor(SURFACE)

    for ax, (error, index, window, phases) in zip(axes.ravel(), worst, strict=False):
        ax.set_facecolor(SURFACE)
        predictor = multimodal.with_wind(window.wind)
        pred = predictor.predict(window.obs, horizons)
        weights = getattr(multimodal, "last_probabilities", None)

        obs = window.obs[index] / 1000
        truth = window.future[index] / 1000
        ax.plot(obs[:, 0], obs[:, 1], color=INK, linewidth=2, label="observed")
        ax.plot(
            [obs[-1, 0], *truth[:, 0]],
            [obs[-1, 1], *truth[:, 1]],
            color=CRITICAL,
            linewidth=2,
            label="what happened",
        )

        for k in range(pred.shape[1]):
            p = float(weights[index][k]) if weights is not None else 1.0 / pred.shape[1]
            path = np.concatenate([window.obs[index, -1:, :2], pred[index, k, :, :2]]) / 1000
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=COLORS["social"],
                linewidth=0.8 + 2.4 * p,
                alpha=min(0.15 + 0.8 * p, 0.95),
            )

        ax.plot(obs[-1, 0], obs[-1, 1], marker="o", markersize=7, color=INK, zorder=5)
        ax.set_aspect("equal")
        ax.set_title(
            f"{phases[index]}  ·  {error:,.0f} m out  ·  {window.n_agents} aircraft",
            fontsize=9,
            color=INK,
        )
        ax.grid(True, color=GRID, alpha=0.4, linewidth=0.5)
        ax.tick_params(colors=MUTED, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    axes.ravel()[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "The eight worst predictions: observed track in black, what happened in red, "
        "the model's six futures in violet",
        fontsize=11,
        color=INK,
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out / "failure_gallery.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out / "failure_gallery.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--social",
        default="artifacts/transformer/model.pt",
        help="Single-mode social model: the ablation pair, differing in one respect only.",
    )
    parser.add_argument("--no-social", default="artifacts/transformer_nosocial/model.pt")
    parser.add_argument(
        "--multimodal",
        default="artifacts/transformer_mm/model.pt",
        help="Used for conflict outcomes and the gallery, where probabilities are wanted.",
    )
    parser.add_argument("--source", choices=["trajair", "tartan"], default="tartan")
    parser.add_argument("--root", default="data/tartan")
    parser.add_argument("--airport", default="kbtp")
    parser.add_argument("--test-days", type=int, default=40)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--figures", default="docs/figures")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Errors are compared across the ablation pair, which differ only in whether neighbours
    # are visible. The multimodal model is kept separately: its errors are best-of-6 and so
    # not comparable to single-path predictions.
    models = {
        "Kalman": KalmanConstantVelocity(),
        "no social": load_predictor(args.no_social),
        "social": load_predictor(args.social),
    }
    multimodal = load_predictor(args.multimodal)

    horizons = np.array(future_offsets()) - (OBS_LEN - 1)
    dense = np.arange(1.0, horizons.max() + 1.0)

    source = open_source(args.source, args.root, args.airport)
    _, test_dates = source.split(args.test_days)

    aircraft, pairs, worst = collect(
        source, test_dates, models, multimodal, horizons, dense, args.stride
    )

    artifacts, figures = Path(args.artifacts), Path(args.figures)
    artifacts.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    aircraft.to_csv(artifacts / "error_by_window.csv", index=False)

    phases = report_phases(aircraft, artifacts)
    report_mechanism(aircraft, artifacts)
    report_conflicts(pairs)

    figure_phases(phases, figures)
    figure_gallery(worst, multimodal, horizons, figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
