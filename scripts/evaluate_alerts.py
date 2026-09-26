"""The headline comparison: lead time vs false alarms, against closure-rate alerting.

    python scripts/evaluate_alerts.py --subset data/trajair/7days1/7days1

Closure-rate alerting is swept over its horizon (tau), and each trajectory predictor is
swept over the horizon it is allowed to look ahead. Both are scored by identical code:
alert at the last observed instant, then check what the aircraft actually did.
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pcas.data.adsb import day_to_scenes, read_raw_day
from pcas.data.scenes import OBS_LEN, build_windows, future_offsets
from pcas.eval.conflicts import NMAC, PROXIMITY, AlertScorer, closure_rate_alerts, predicted_alerts
from pcas.models.baselines import ConstantTurnRate, ConstantVelocity, KalmanConstantVelocity

log = logging.getLogger("pcas.alerts")

TAUS = (20.0, 40.0, 60.0, 90.0, 120.0)


def load_test_windows(subset: Path, test_days: int, stride: int) -> list:
    windows = []
    for path in sorted(glob.glob(str(subset / "raw_data" / "*" / "*.csv"))):
        day = read_raw_day(path)
        for scene in day_to_scenes(day, min_agents=2):
            windows += build_windows(scene, stride=stride, min_agents=2)
    days = sorted({w.date for w in windows if w.date})
    held_out = set(days[-test_days:])
    log.info("days %s, holding out %s", days, sorted(held_out))
    return [w for w in windows if w.date in held_out]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="data/trajair/7days1/7days1")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--criterion", choices=["proximity", "nmac"], default="proximity")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    criterion = PROXIMITY if args.criterion == "proximity" else NMAC

    test = load_test_windows(Path(args.subset), args.test_days, args.stride)
    multi = [w for w in test if w.n_agents >= 2]
    print(
        f"\ncriterion: {criterion.name} "
        f"({criterion.horizontal_m:.0f} m horizontal, {criterion.vertical_m:.0f} m vertical)"
    )
    print(f"test windows with 2+ aircraft: {len(multi)}")

    horizons = np.array(future_offsets()) - (OBS_LEN - 1)
    rows = []

    for tau in TAUS:
        scorer = AlertScorer(criterion, window_stride_s=args.stride)
        for w in multi:
            scorer.update(w, closure_rate_alerts(w.obs, criterion, tau_s=tau), horizon_s=tau)
        rows.append({"method": f"closure_rate (tau={tau:.0f}s)", **scorer.summary()})

    predictors = (ConstantVelocity(), ConstantTurnRate(), KalmanConstantVelocity())
    for model in predictors:
        for horizon in (40.0, 120.0):
            mask = horizons <= horizon
            scorer = AlertScorer(criterion, window_stride_s=args.stride)
            for w in multi:
                pred = model.predict(w.obs, horizons[mask])
                alerts = predicted_alerts(pred, horizons[mask], criterion)
                scorer.update(w, alerts, horizon_s=horizon)
            rows.append({"method": f"{model.name} (<={horizon:.0f}s)", **scorer.summary()})

    table = pd.DataFrame(rows).set_index("method")
    show = [
        "events",
        "detected",
        "detection_rate",
        "false_alarms",
        "false_alarms_per_hour",
        "median_lead_time_s",
    ]
    print("\n" + table[show].round(3).to_string())

    # Where the horizons differ: detection split by how far ahead the event really was.
    print("\ndetection by actual lead time, closure_rate(tau=40s) vs kalman_cv(<=120s):")
    for label, make in (
        ("closure_rate(40s)", lambda w: closure_rate_alerts(w.obs, criterion, tau_s=40.0)),
        (
            "kalman_cv(120s)",
            lambda w: predicted_alerts(
                KalmanConstantVelocity().predict(w.obs, horizons), horizons, criterion
            ),
        ),
    ):
        scorer = AlertScorer(criterion, window_stride_s=args.stride)
        for w in multi:
            scorer.update(w, make(w), horizon_s=120.0)
        buckets = scorer.detection_by_lead_time()
        print(f"\n  {label}")
        print("  " + buckets.round(3).to_string(index=False).replace("\n", "\n  "))

    return 0


if __name__ == "__main__":
    sys.exit(main())
