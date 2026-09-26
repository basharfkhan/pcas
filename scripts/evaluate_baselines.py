"""Score the physics baselines on held-out days.

    python scripts/evaluate_baselines.py --subset data/trajair/7days1/7days1

Days are split chronologically: the model never sees the test days, and no day appears on
both sides. That is stricter than TrajAir's own random split over scene files, in which
every day of 7days1 appears on both sides.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pcas.data.adsb import day_to_scenes, read_raw_day
from pcas.data.scenes import OBS_LEN, PRED_STEP, build_windows, future_offsets
from pcas.data.subsets import chronological_days, raw_day_files
from pcas.eval.metrics import MetricAccumulator
from pcas.models.baselines import DEFAULT_BASELINES

log = logging.getLogger("pcas.evaluate")


def load_windows(subset: Path, stride: int, min_agents: int, dates: list[str]) -> list:
    """Build windows for the given dates only, so the full subset need not fit in RAM."""
    files = raw_day_files(subset)
    windows = []
    for date in dates:
        day = read_raw_day(files[date])
        for scene in day_to_scenes(day, min_agents=min_agents):
            windows += build_windows(scene, stride=stride, min_agents=min_agents)
        log.info("%s: %d windows so far", day.date, len(windows))
    return windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="data/trajair/7days1/7days1")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--min-agents", type=int, default=1)
    parser.add_argument("--test-days", type=int, default=2, help="Held-out final days.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _, test_dates = chronological_days(args.subset, args.test_days)
    test = load_windows(Path(args.subset), args.stride, args.min_agents, test_dates)
    if not test:
        log.error("no windows found under %s", args.subset)
        return 1

    # Horizons are seconds past the LAST OBSERVED sample, not past the window start.
    horizons = np.array(future_offsets()) - (OBS_LEN - 1)
    print(f"\nheld-out test days: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
    print(f"test windows: {len(test)}")
    print(f"agents in test windows: {sum(w.n_agents for w in test)}")
    print(f"horizons (s past last observation): {horizons.tolist()}\n")

    rows, horizon_tables = [], {}
    for model in DEFAULT_BASELINES:
        acc = MetricAccumulator(horizons_s=horizons)
        for window in test:
            acc.update(model.predict(window.obs, horizons), window.future)
        summary = acc.summary()
        rows.append({"model": model.name, **summary})
        horizon_tables[model.name] = acc.horizon_table()

    table = pd.DataFrame(rows).set_index("model")
    show = [
        "minADE_m",
        "minFDE_m",
        "minFDE_horizontal_m",
        "minFDE_vertical_m",
        "median_FDE_m",
        "p95_FDE_m",
        "miss_rate_152m",
    ]
    print(table[show].round(1).to_string())

    best = table["minFDE_m"].idxmin()
    print(f"\nbest baseline by minFDE: {best} ({table.loc[best, 'minFDE_m']:.0f} m at 120 s)")

    print(f"\nerror vs horizon for {best} (metres):")
    print(horizon_tables[best].round(1).to_string(index=False))

    print(f"\nprediction step: {PRED_STEP} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
