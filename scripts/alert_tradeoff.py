"""Trade-off curves: detection against false alarms, at matched operating points.

    python scripts/alert_tradeoff.py --checkpoint artifacts/lstm_big/lstm.pt

Comparing detection rates measured at different false alarm rates says nothing: any method
can detect more by alerting more. Each method is therefore swept over its own sensitivity
knob so the curves can be read at a common false alarm rate.

- closure rate: sweep tau, its alerting horizon.
- prediction based: sweep how tightly the predicted separation must break the threshold.
  Truth always uses the full criterion; only the alerting test tightens, so a method can
  be made quiet without redefining what counts as a conflict.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from pcas.data.adsb import day_to_scenes, read_raw_day
from pcas.data.scenes import OBS_LEN, build_windows, future_offsets
from pcas.data.subsets import chronological_days, raw_day_files
from pcas.eval.conflicts import (
    NMAC,
    PROXIMITY,
    AlertScorer,
    ConflictCriterion,
    closure_rate_alerts,
    predicted_alerts,
)
from pcas.models.baselines import KalmanConstantVelocity

log = logging.getLogger("pcas.tradeoff")

TAUS = (10.0, 20.0, 30.0, 40.0, 60.0, 90.0, 120.0)
SCALES = (0.15, 0.25, 0.35, 0.5, 0.7, 1.0)
BUCKETS = (0, 30, 60, 90, 120)


def tighten(criterion: ConflictCriterion, scale: float) -> ConflictCriterion:
    return ConflictCriterion(
        horizontal_m=criterion.horizontal_m * scale,
        vertical_m=criterion.vertical_m * scale,
        name=f"{criterion.name}x{scale}",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/lstm_big/lstm.pt")
    parser.add_argument("--subset", default="data/trajair/111_days/111_days")
    parser.add_argument("--test-days", type=int, default=22)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--criterion", choices=["proximity", "nmac"], default="proximity")
    parser.add_argument("--out", default="artifacts/tradeoff.csv")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    truth_criterion = PROXIMITY if args.criterion == "proximity" else NMAC

    from pcas.models.lstm import load_predictor

    lstm = load_predictor(args.checkpoint)
    kalman = KalmanConstantVelocity()

    horizons = np.array(future_offsets()) - (OBS_LEN - 1)
    dense = np.arange(1.0, horizons.max() + 1.0)

    _, test_dates = chronological_days(args.subset, args.test_days)
    files = raw_day_files(args.subset)

    # One scorer per (method, setting).
    scorers: dict[tuple[str, float], AlertScorer] = {}
    for tau in TAUS:
        scorers[("closure_rate", tau)] = AlertScorer(truth_criterion, args.stride)
    for scale in SCALES:
        scorers[("kalman_cv", scale)] = AlertScorer(truth_criterion, args.stride)
        scorers[("lstm", scale)] = AlertScorer(truth_criterion, args.stride)

    for date in test_dates:
        day = read_raw_day(files[date])
        windows = []
        for scene in day_to_scenes(day, min_agents=2):
            windows += build_windows(scene, stride=args.stride, min_agents=2)

        for w in windows:
            # Predict once per window, then reuse for every sensitivity setting.
            kal = kalman.predict(w.obs, dense)
            net = lstm.with_wind(w.wind).predict(w.obs, dense)

            for tau in TAUS:
                scorers[("closure_rate", tau)].update(
                    w, closure_rate_alerts(w.obs, truth_criterion, tau_s=tau), horizon_s=tau
                )
            for scale in SCALES:
                test = tighten(truth_criterion, scale)
                scorers[("kalman_cv", scale)].update(
                    w, predicted_alerts(kal, dense, test), horizon_s=120.0
                )
                scorers[("lstm", scale)].update(
                    w, predicted_alerts(net, dense, test), horizon_s=120.0
                )
        log.info("%s: %d windows", date, len(windows))

    rows = []
    for (method, setting), scorer in scorers.items():
        summary = scorer.summary()
        buckets = scorer.detection_by_lead_time(BUCKETS).set_index("lead_bucket_s")
        row = {
            "method": method,
            "setting": setting,
            "false_alarms_per_hour": summary["false_alarms_per_hour"],
            "detection_rate": summary["detection_rate"],
            "median_lead_time_s": summary["median_lead_time_s"],
            "events": summary["events"],
        }
        for label, values in buckets.iterrows():
            row[f"det_{label}"] = values["detection_rate"]
        rows.append(row)

    table = pd.DataFrame(rows).sort_values(["method", "false_alarms_per_hour"])
    table.to_csv(args.out, index=False)
    print("\n" + table.round(3).to_string(index=False))
    print(f"\nwritten to {args.out}")

    # Read each method at the quietest setting that stays under a common budget.
    print("\nmatched comparison, at most 2 false alarms per hour:")
    budget = table[table["false_alarms_per_hour"] <= 2.0]
    if budget.empty:
        print("  no setting for any method is that quiet")
    else:
        best = budget.sort_values("detection_rate").groupby("method").tail(1)
        print(best.round(3).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
