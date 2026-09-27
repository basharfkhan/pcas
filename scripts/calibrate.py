"""Fit and check calibration of the stated conflict probability.

    python scripts/calibrate.py --checkpoint artifacts/transformer_mm/model.pt

The map is fitted on the **validation** sessions and reported on the **test** sessions. Fitting
on the test days would turn the reliability table into a self-portrait, which is the whole
failure mode this is meant to expose.

Calibration is not expected to change detection: an isotonic map is monotone, so every
operating point survives, relabelled with a probability that means something. The script checks
that too, by reporting detection at a fixed calibrated threshold alongside the raw one.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from pcas.data.adsb import day_to_scenes
from pcas.data.scenes import OBS_LEN, build_windows, future_offsets
from pcas.data.sources import iter_days, open_source
from pcas.eval.calibration import (
    expected_calibration_error,
    fit_isotonic,
    reliability,
)
from pcas.eval.conflicts import NMAC, PROXIMITY, first_violation, predicted_alerts
from pcas.models.load import load_predictor

log = logging.getLogger("pcas.calibrate")


def collect(source, dates, model, criterion, dense, stride) -> pd.DataFrame:
    """Every aircraft pair's stated conflict probability and what actually happened.

    Pairs the model puts at zero probability are kept: they are most of the airspace, and a
    calibration fitted only on alerted pairs would be fitted on a biased sample.
    """
    rows = []
    for day in iter_days(source, dates):
        windows = []
        for scene in day_to_scenes(day, min_agents=2):
            windows += build_windows(scene, stride=stride, min_agents=2)

        for window in windows:
            pred = model.with_wind(window.wind).predict(window.obs, dense)
            weights = getattr(model, "last_probabilities", None)
            # threshold 0 keeps every pair that has any violating combination at all.
            alerts = predicted_alerts(
                pred, dense, criterion, probability_threshold=0.0, mode_probabilities=weights
            )
            stated = {(int(r.i), int(r.j)): float(r.probability) for r in alerts.itertuples()}

            times = np.arange(1, window.future_dense.shape[1] + 1, dtype=float)
            truth = first_violation(window.future_dense, criterion, times)
            events = {(int(r.i), int(r.j)) for r in truth.itertuples() if r.onset_s <= 120.0}

            for i in range(window.n_agents):
                for j in range(i + 1, window.n_agents):
                    rows.append(
                        {
                            "stated": stated.get((i, j), 0.0),
                            "event": float((i, j) in events),
                        }
                    )
        log.info("%s: %d windows, %d pairs so far", day.date, len(windows), len(rows))

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/transformer_mm/model.pt")
    parser.add_argument("--source", choices=["trajair", "tartan"], default="tartan")
    parser.add_argument("--root", default="data/tartan")
    parser.add_argument("--airport", default="kbtp")
    parser.add_argument("--test-days", type=int, default=40)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--criterion", choices=["proximity", "nmac"], default="proximity")
    parser.add_argument("--out", default="artifacts/calibration.json")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    criterion = PROXIMITY if args.criterion == "proximity" else NMAC

    model = load_predictor(args.checkpoint)
    if getattr(model.config, "n_modes", 1) <= 1:
        log.error("%s is single-mode: there is no probability to calibrate", args.checkpoint)
        return 1

    source = open_source(args.source, args.root, args.airport)
    dates = source.dates()
    test_dates = dates[-args.test_days :]
    val_dates = dates[-(args.test_days + args.val_days) : -args.test_days]
    log.info(
        "fitting on %d validation sessions, reporting on %d test sessions",
        len(val_dates),
        len(test_dates),
    )

    horizons = np.array(future_offsets()) - (OBS_LEN - 1)
    dense = np.arange(1.0, horizons.max() + 1.0)

    log.info("collecting validation pairs...")
    val = collect(source, val_dates, model, criterion, dense, args.stride)
    log.info("collecting test pairs...")
    test = collect(source, test_dates, model, criterion, dense, args.stride)

    calibrator = fit_isotonic(val["stated"].to_numpy(), val["event"].to_numpy())
    calibrator.to_json(args.out)

    calibrated = calibrator.predict(test["stated"].to_numpy())
    outcome = test["event"].to_numpy()

    print(f"\nvalidation pairs: {len(val)}, test pairs: {len(test)}")
    print(f"conflict base rate in test pairs: {outcome.mean():.4f}")

    print("\nreliability BEFORE calibration (test sessions):")
    print(reliability(test["stated"].to_numpy(), outcome).round(3).to_string(index=False))
    print("\nreliability AFTER calibration (test sessions):")
    print(reliability(calibrated, outcome).round(3).to_string(index=False))

    before = expected_calibration_error(test["stated"].to_numpy(), outcome)
    after = expected_calibration_error(calibrated, outcome)
    print(f"\nexpected calibration error: {before:.4f} -> {after:.4f}")

    # Monotone maps cannot reorder pairs, so the ranking is untouched. Show that the same
    # operating points still exist, just under different threshold labels.
    print("\nsame curve, relabelled: detection and alert rate at matched thresholds")
    rows = []
    for threshold in (0.05, 0.1, 0.2, 0.35, 0.5):
        raw_mask = test["stated"].to_numpy() >= threshold
        cal_mask = calibrated >= threshold
        rows.append(
            {
                "threshold": threshold,
                "raw_alerts": int(raw_mask.sum()),
                "raw_detection": float(outcome[raw_mask].mean()) if raw_mask.any() else np.nan,
                "cal_alerts": int(cal_mask.sum()),
                "cal_detection": float(outcome[cal_mask].mean()) if cal_mask.any() else np.nan,
            }
        )
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"\ncalibration map written to {args.out} ({calibrator.n_samples} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
