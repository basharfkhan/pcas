"""Replay real traffic with the model's predictions, as an animation.

    python scripts/make_replay.py --out docs/figures/replay.gif

Everything else in this repo is a number. This is the same model, on the same held-out
sessions, drawn: recorded aircraft moving, the six futures it thinks each one might fly, and
the calibrated probability that a pair is about to lose separation.

It searches the held-out sessions for a scene containing a real conflict, and replays the
couple of minutes around it. Nothing is staged: the aircraft are ADS-B recordings the model
never trained on, and the predictions are made from the 11 seconds before each frame.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pcas.data.adsb import day_to_scenes
from pcas.data.scenes import OBS_LEN, PRED_LEN, build_windows, future_offsets
from pcas.data.sources import iter_days, open_source
from pcas.eval.calibration import ProbabilityCalibrator
from pcas.eval.conflicts import PROXIMITY, first_violation, predicted_alerts
from pcas.models.load import load_predictor

log = logging.getLogger("pcas.replay")

# Validated categorical slots for aircraft identity, and the fixed status red for an alert.
AIRCRAFT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100", "#e87ba4"]
CRITICAL = "#d03b3b"
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d8d7d2"


def find_conflict_scene(source, dates, stride: int, criterion):
    """First scene in the held-out sessions that contains a real loss of separation."""
    for day in iter_days(source, dates):
        for scene in day_to_scenes(day, min_agents=2):
            windows = build_windows(scene, stride=stride, min_agents=2)
            for window in windows:
                times = np.arange(1, window.future_dense.shape[1] + 1, dtype=float)
                events = first_violation(window.future_dense, criterion, times)
                if not events.empty:
                    log.info(
                        "%s scene %s: conflict at frame %d, %d aircraft",
                        day.date,
                        scene.scene_id,
                        window.start_frame,
                        window.n_agents,
                    )
                    return day, scene, window
    return None, None, None


def observations_at(frames: pd.DataFrame, end_frame: int, obs_len: int = OBS_LEN):
    """Positions for every aircraft present through the `obs_len` frames ending here."""
    wanted = np.arange(end_frame - obs_len + 1, end_frame + 1)
    present = None
    for frame in wanted:
        here = set(frames.loc[frames["frame"] == frame, "agent_id"])
        present = here if present is None else present & here
        if not present:
            return (), np.zeros((0, obs_len, 3))

    agents = tuple(sorted(present))
    lookup = {
        (int(row.frame), row.agent_id): (row.x_m, row.y_m, row.z_m)
        for row in frames[frames["frame"].isin(wanted)].itertuples(index=False)
    }
    obs = np.array([[lookup[(f, a)] for f in wanted] for a in agents], dtype=float)
    return agents, obs


def draw_frame(plt, scene, frames, end_frame, model, calibrator, criterion, horizons, limits):
    """One animation frame: aircraft, their trails, the model's futures, and any alert."""
    agents, obs = observations_at(frames, end_frame)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Runway, drawn along the frame's x axis because the frame is runway-aligned.
    ax.plot([-0.5, 0.5], [0, 0], color=MUTED, linewidth=3, solid_capstyle="butt", zorder=2)
    ax.plot(0, 0, marker="o", markersize=4, color=MUTED, zorder=2)
    ax.text(0, -0.45, "KBTP", fontsize=8, color=MUTED, ha="center", va="top")

    alert_text = None
    if len(agents):
        pred = model.with_wind(scene.frames[["windx", "windy"]].iloc[0].to_numpy()).predict(
            obs, horizons
        )
        probabilities = getattr(model, "last_probabilities", None)

        for i, agent in enumerate(agents):
            color = AIRCRAFT_COLORS[i % len(AIRCRAFT_COLORS)]

            trail = frames[
                (frames["agent_id"] == agent)
                & (frames["frame"] <= end_frame)
                & (frames["frame"] > end_frame - 45)
            ]
            ax.plot(
                trail["x_m"] / 1000, trail["y_m"] / 1000, color=color, linewidth=1.2, alpha=0.55
            )

            x, y = obs[i, -1, 0] / 1000, obs[i, -1, 1] / 1000
            ax.plot(x, y, marker="o", markersize=9, color=color, zorder=5)
            ax.plot(
                x,
                y,
                marker="o",
                markersize=9,
                markerfacecolor="none",
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                zorder=6,
            )
            ax.text(
                x + 0.18, y + 0.18, f"{obs[i, -1, 2]:.0f} m", fontsize=7.5, color=color, zorder=6
            )

            # Each hypothesis, with opacity carrying its probability.
            weights = probabilities[i] if probabilities is not None else None
            for k in range(pred.shape[1]):
                p = float(weights[k]) if weights is not None else 1.0 / pred.shape[1]
                path = np.concatenate([obs[i, -1:, :2], pred[i, k, :, :2]]) / 1000
                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    color=color,
                    linewidth=1.0 + 2.2 * p,
                    alpha=min(0.12 + 0.85 * p, 0.95),
                    zorder=4,
                )

        if len(agents) >= 2:
            alerts = predicted_alerts(
                pred,
                horizons,
                criterion,
                probability_threshold=0.0,
                mode_probabilities=probabilities,
            )
            for row in alerts.itertuples():
                calibrated = float(calibrator.predict(np.array([row.probability]))[0])
                if calibrated < 0.2:
                    continue
                a, b = obs[int(row.i), -1, :2] / 1000, obs[int(row.j), -1, :2] / 1000
                ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    color=CRITICAL,
                    linewidth=1.6,
                    linestyle=(0, (5, 3)),
                    zorder=7,
                )
                separation = np.linalg.norm(a - b) * 1000
                alert_text = (
                    f"CONFLICT  {calibrated:.0%} within {row.alert_time_s:.0f} s"
                    f"   (now {separation:,.0f} m apart)"
                )

    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, alpha=0.45, linewidth=0.6)
    ax.set_xlabel("km along the runway", color=MUTED, fontsize=9)
    ax.set_ylabel("km across the runway", color=MUTED, fontsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)

    seconds = end_frame - frames["frame"].min()
    ax.set_title(
        f"{scene.date}   t+{seconds}s   {len(agents)} aircraft   predicting {PRED_LEN} s ahead",
        fontsize=10,
        color=INK,
    )
    if alert_text:
        # Icon plus words: the status colour never carries the meaning alone.
        ax.text(
            0.5,
            0.965,
            f"!  {alert_text}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
            color="#ffffff",
            zorder=9,
            bbox={
                "facecolor": CRITICAL,
                "edgecolor": "none",
                "pad": 5,
                "boxstyle": "round,pad=0.4",
            },
        )

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/transformer_mm/model.pt")
    parser.add_argument("--calibration", default="artifacts/calibration.json")
    parser.add_argument("--source", choices=["trajair", "tartan"], default="tartan")
    parser.add_argument("--root", default="data/tartan")
    parser.add_argument("--airport", default="kbtp")
    parser.add_argument("--test-days", type=int, default=40)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--seconds", type=int, default=150, help="Scene time to replay.")
    parser.add_argument("--step", type=int, default=3, help="Seconds between frames.")
    parser.add_argument("--out", default="docs/figures/replay.gif")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    model = load_predictor(args.checkpoint)
    calibrator = ProbabilityCalibrator.from_json(args.calibration)
    horizons = np.array(future_offsets()) - (OBS_LEN - 1)

    source = open_source(args.source, args.root, args.airport)
    _, test_dates = source.split(args.test_days)

    day, scene, window = find_conflict_scene(source, test_dates, args.stride, PROXIMITY)
    if scene is None:
        log.error("no conflict found in the held-out sessions")
        return 1

    frames = scene.frames
    # Start a little before the window whose future holds the conflict.
    start = max(int(frames["frame"].min()) + OBS_LEN, window.start_frame - 30)
    end = min(int(frames["frame"].max()) - PRED_LEN, start + args.seconds)

    # Frame on the pair that actually conflicts, over the replayed span. One transiting
    # aircraft 10 km away would otherwise stretch the view until the interesting part is a
    # few pixels across.
    times = np.arange(1, window.future_dense.shape[1] + 1, dtype=float)
    events = first_violation(window.future_dense, PROXIMITY, times)
    involved = {
        window.agent_ids[int(events.iloc[0].i)],
        window.agent_ids[int(events.iloc[0].j)],
    }
    log.info("framing on %s", sorted(involved))

    span = frames[
        frames["agent_id"].isin(involved)
        & (frames["frame"] >= start)
        & (frames["frame"] <= end + PRED_LEN)
    ]
    centre_x, centre_y = span["x_m"].mean() / 1000, span["y_m"].mean() / 1000
    half = max(
        2.5,
        (span["x_m"].max() - span["x_m"].min()) / 2000 + 1.0,
        (span["y_m"].max() - span["y_m"].min()) / 2000 + 1.0,
    )
    limits = (centre_x - half, centre_x + half, centre_y - half, centre_y + half)

    images = []
    for frame_no in range(start, end, args.step):
        buffer = draw_frame(
            plt, scene, frames, frame_no, model, calibrator, PROXIMITY, horizons, limits
        )
        images.append(Image.open(buffer).convert("P", palette=Image.ADAPTIVE))
        if len(images) % 10 == 0:
            log.info("  %d frames", len(images))

    if not images:
        log.error("no frames rendered")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out, save_all=True, append_images=images[1:], duration=int(args.step * 260), loop=0
    )
    log.info("wrote %s (%d frames)", out, len(images))
    return 0


if __name__ == "__main__":
    sys.exit(main())
