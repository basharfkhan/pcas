"""Score a trained LSTM against the physics baselines and closure-rate alerting.

    python scripts/evaluate_lstm.py --checkpoint artifacts/lstm/lstm.pt

Uses the same held-out days, the same metrics harness and the same alerting code as
scripts/evaluate_baselines.py and scripts/evaluate_alerts.py, so rows are comparable.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from pcas.data.adsb import day_to_scenes
from pcas.data.scenes import OBS_LEN, build_windows, future_offsets
from pcas.data.sources import open_source
from pcas.eval.conflicts import NMAC, PROXIMITY, AlertScorer, closure_rate_alerts, predicted_alerts
from pcas.eval.metrics import MetricAccumulator
from pcas.models.baselines import KalmanConstantVelocity
from pcas.models.lstm import load_predictor

log = logging.getLogger("pcas.evaluate_lstm")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/lstm/lstm.pt")
    parser.add_argument("--source", choices=["trajair", "tartan"], default="tartan")
    parser.add_argument("--root", default="data/tartan")
    parser.add_argument("--airport", default="kbtp")
    parser.add_argument("--test-days", type=int, default=22)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--criterion", choices=["proximity", "nmac"], default="proximity")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    criterion = PROXIMITY if args.criterion == "proximity" else NMAC

    lstm = load_predictor(args.checkpoint)
    kalman = KalmanConstantVelocity()
    log.info("loaded %s on %s", args.checkpoint, lstm.device)

    source = open_source(args.source, args.root, args.airport)
    _, test_dates = source.split(args.test_days)

    horizons = np.array(future_offsets()) - (OBS_LEN - 1)
    dense = np.arange(1.0, horizons.max() + 1.0)

    accs = {name: MetricAccumulator(horizons_s=horizons) for name in ("lstm", "kalman_cv")}
    scorers = {
        "lstm (<=120s)": AlertScorer(criterion, window_stride_s=args.stride),
        "kalman_cv (<=120s)": AlertScorer(criterion, window_stride_s=args.stride),
        "closure_rate (tau=40s)": AlertScorer(criterion, window_stride_s=args.stride),
    }

    for date in test_dates:
        day = source.read(date)
        windows = []
        for scene in day_to_scenes(day, min_agents=1):
            windows += build_windows(scene, stride=args.stride, min_agents=1)

        for w in windows:
            accs["lstm"].update(lstm.with_wind(w.wind).predict(w.obs, horizons), w.future)
            accs["kalman_cv"].update(kalman.predict(w.obs, horizons), w.future)

            if w.n_agents < 2:
                continue
            lstm_dense = lstm.with_wind(w.wind).predict(w.obs, dense)
            scorers["lstm (<=120s)"].update(
                w, predicted_alerts(lstm_dense, dense, criterion), 120.0
            )
            kal_dense = kalman.predict(w.obs, dense)
            scorers["kalman_cv (<=120s)"].update(
                w, predicted_alerts(kal_dense, dense, criterion), 120.0
            )
            scorers["closure_rate (tau=40s)"].update(
                w, closure_rate_alerts(w.obs, criterion, tau_s=40.0), 40.0
            )
        log.info("%s: %d windows", date, len(windows))

    print("\ntrajectory error (metres):")
    print(pd.DataFrame({k: v.summary() for k, v in accs.items()}).T.round(1).to_string())

    print("\nalerting:")
    print(pd.DataFrame({k: v.summary() for k, v in scorers.items()}).T.round(3).to_string())

    print("\ndetection by actual lead time:")
    for name, scorer in scorers.items():
        print(f"\n  {name}")
        table = scorer.detection_by_lead_time()
        print("  " + table.round(3).to_string(index=False).replace("\n", "\n  "))

    return 0


if __name__ == "__main__":
    sys.exit(main())
