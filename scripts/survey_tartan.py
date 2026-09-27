"""Survey TartanAviation session quality, per airport.

    python scripts/survey_tartan.py --sample 12

Reception varies session to session, and the difference decides how much of this dataset is
usable: a session whose reports arrive every 5 s cannot support 1 Hz tracks, and a session
whose tracks are fragmented into 10 s pieces cannot fill a 131 s window however good its
sampling rate is. Rather than generalise from one or two days, this samples sessions evenly
across each airport's range and reports what survives each stage.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from pcas.data.adsb import day_to_scenes
from pcas.data.scenes import build_windows
from pcas.data.tartan import find_sessions, read_session

log = logging.getLogger("pcas.survey")


def survey(root: str, airport: str, sample: int, stride: int) -> pd.DataFrame:
    sessions = find_sessions(root, airport)
    dates = sorted(sessions)
    if not dates:
        return pd.DataFrame()

    picked = [dates[i] for i in np.linspace(0, len(dates) - 1, min(sample, len(dates))).astype(int)]
    rows = []

    for date in picked:
        try:
            day = read_session(sessions[date], date=date, weather_root=root)
        except (ValueError, OSError) as exc:
            log.warning("%s %s: %s", airport, date, exc)
            rows.append({"airport": airport, "date": date, "error": str(exc)[:40]})
            continue

        windows = []
        for scene in day_to_scenes(day, min_agents=1):
            windows += build_windows(scene, stride=stride, min_agents=1)

        rows.append(
            {
                "airport": airport,
                "date": date,
                "files": len(sessions[date]),
                "median_gap_s": round(day.median_gap_s, 2),
                "track_rows": len(day.tracks),
                "aircraft": int(day.tracks["agent_id"].nunique()) if len(day.tracks) else 0,
                "windows": len(windows),
                "multi_agent": sum(1 for w in windows if w.n_agents >= 2),
                "field_elev_m": round(day.field_elev_m),
            }
        )
        log.info("%s %s: %s", airport, date, rows[-1])

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/tartan")
    parser.add_argument("--sample", type=int, default=12, help="Sessions per airport.")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--out", default="artifacts/tartan_survey.csv")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tables = [survey(args.root, airport, args.sample, args.stride) for airport in ("kbtp", "kagc")]
    table = pd.concat([t for t in tables if not t.empty], ignore_index=True)
    table.to_csv(args.out, index=False)

    print("\n" + table.to_string(index=False))

    usable = table[table.get("windows", pd.Series(dtype=float)).fillna(0) > 0]
    print("\nper airport:")
    for airport, group in table.groupby("airport"):
        ok = group[group["windows"].fillna(0) > 0] if "windows" in group else group
        print(
            f"  {airport}: {len(ok)}/{len(group)} sessions yield windows | "
            f"median gap {group['median_gap_s'].median():.2f} s | "
            f"windows/session median {ok['windows'].median() if len(ok) else 0:.0f} | "
            f"multi-agent median {ok['multi_agent'].median() if len(ok) else 0:.0f}"
        )
    print(f"\nwritten to {args.out} ({len(usable)} usable of {len(table)} sampled)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
