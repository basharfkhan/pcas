"""The headline comparison: lead time vs false alarms, against closure-rate alerting.

    python scripts/evaluate_alerts.py --subset data/trajair/7days1/7days1

Closure-rate alerting is swept over its horizon (tau), and each trajectory predictor is
swept over the horizon it is allowed to look ahead. Both are scored by identical code:
alert at the last observed instant, then check what the aircraft actually did.
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
from pcas.eval.conflicts import NMAC, PROXIMITY, AlertScorer, closure_rate_alerts, predicted_alerts
from pcas.models.baselines import ConstantTurnRate, ConstantVelocity, KalmanConstantVelocity

log = logging.getLogger("pcas.alerts")

TAUS = (20.0, 40.0, 60.0, 90.0, 120.0)


def load_test_windows(source, test_days: int, stride: int) -> list:
    """Build windows for the held-out days only; the rest is never read."""
    train, test = source.split(test_days)
    log.info(
        "%d days total, holding out %d: %s to %s", len(source.days), len(test), test[0], test[-1]
    )
    log.info("train days span %s to %s (not read here)", train[0], train[-1])

    windows = []
    for day in iter_days(source, test):
        for scene in day_to_scenes(day, min_agents=2):
            windows += build_windows(scene, stride=stride, min_agents=2)
        log.info("  %s: %d windows", day.date, len(windows))
    return windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["trajair", "tartan"], default="tartan")
    parser.add_argument("--root", default="data/tartan")
    parser.add_argument("--airport", default="kbtp")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--test-days", type=int, default=2)
    parser.add_argument("--criterion", choices=["proximity", "nmac"], default="proximity")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    criterion = PROXIMITY if args.criterion == "proximity" else NMAC

    source = open_source(args.source, args.root, args.airport)
    test = load_test_windows(source, args.test_days, args.stride)
    multi = [w for w in test if w.n_agents >= 2]
    print(
        f"\ncriterion: {criterion.name} "
        f"({criterion.horizontal_m:.0f} m horizontal, {criterion.vertical_m:.0f} m vertical)"
    )
    print(f"test windows with 2+ aircraft: {len(multi)}")

    # Waypoint horizons the models are trained to output (10 s spacing).
    horizons = np.array(future_offsets()) - (OBS_LEN - 1)
    # Alerting checks EVERY second instead. Closure-rate alerting solves for the closest
    # approach analytically, so scoring predictions only at 10 s waypoints would let them
    # step over a brief violation and lose on a technicality rather than on substance.
    dense_horizons = np.arange(1.0, horizons.max() + 1.0)
    rows = []

    for tau in TAUS:
        scorer = AlertScorer(criterion, window_stride_s=args.stride)
        for w in multi:
            scorer.update(w, closure_rate_alerts(w.obs, criterion, tau_s=tau), horizon_s=tau)
        rows.append({"method": f"closure_rate (tau={tau:.0f}s)", **scorer.summary()})

    predictors = (ConstantVelocity(), ConstantTurnRate(), KalmanConstantVelocity())
    for model in predictors:
        for horizon in (40.0, 120.0):
            mask = dense_horizons <= horizon
            scorer = AlertScorer(criterion, window_stride_s=args.stride)
            for w in multi:
                pred = model.predict(w.obs, dense_horizons[mask])
                alerts = predicted_alerts(pred, dense_horizons[mask], criterion)
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
                KalmanConstantVelocity().predict(w.obs, dense_horizons),
                dense_horizons,
                criterion,
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
