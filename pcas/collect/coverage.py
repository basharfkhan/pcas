"""What the VATSIM collector actually captured.

The collector runs on a desktop that sleeps, so coverage has holes. That matters more than
it sounds: controller staffing on VATSIM peaks on evenings and weekends, so if the holes
land at the same hours every night, the controlled-vs-uncontrolled comparison is measuring
the collector's uptime rather than anything about air traffic.

This module reports coverage per hour of the day so the gaps can be stated instead of
hidden, and so a later analysis can weight or exclude thin hours.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

POLL_SECONDS = 15
EXPECTED_PER_HOUR = 3600 // POLL_SECONDS


def _read(data_dir: str | Path, dataset: str, columns: list[str]) -> pd.DataFrame:
    """Concatenate a dataset's parquet files, tolerating schema drift.

    Files written before a field existed simply do not have it: the collector gained
    `is_observer` partway through its first run. Requesting a missing column from pyarrow
    is an error, so each file is read for the columns it actually has and the rest are
    filled with NA.
    """
    files = sorted(Path(data_dir).glob(f"{dataset}/date=*/*.parquet"))
    if not files:
        return pd.DataFrame(columns=columns)

    frames = []
    for path in files:
        available = set(pq.ParquetFile(path).schema.names)
        frame = pd.read_parquet(path, columns=[c for c in columns if c in available])
        frames.append(frame.reindex(columns=columns))
    return pd.concat(frames, ignore_index=True)


def load_snapshots(data_dir: str | Path) -> pd.Series:
    """Distinct feed snapshot timestamps, sorted."""
    pilots = _read(data_dir, "pilots", ["snapshot_ts"])
    if pilots.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    ts = pd.to_datetime(pilots["snapshot_ts"], utc=True, format="ISO8601")
    return pd.Series(sorted(ts.unique())).reset_index(drop=True)


def gaps(snapshots: pd.Series, min_gap_s: float = 60.0) -> pd.DataFrame:
    """Stretches with no snapshots, longer than `min_gap_s`."""
    if len(snapshots) < 2:
        return pd.DataFrame(columns=["start", "end", "minutes"])

    deltas = snapshots.diff().dt.total_seconds()
    hit = deltas > min_gap_s
    return pd.DataFrame(
        {
            "start": snapshots[hit.shift(-1, fill_value=False)].to_numpy(),
            "end": snapshots[hit].to_numpy(),
            "minutes": (deltas[hit] / 60).round(1).to_numpy(),
        }
    )


def coverage_by_hour(snapshots: pd.Series) -> pd.DataFrame:
    """Observed vs expected snapshots for each hour of the day, pooled over dates.

    Coverage is measured against full collection on every calendar day of the span, so
    0.5 at hour 03 means that hour was captured on half the days, not that half of each
    hour was captured. An hour near 0 is one where any conditional statistic is thin.
    """
    if snapshots.empty:
        return pd.DataFrame(columns=["hour_utc", "snapshots", "days_seen", "coverage"])

    frame = pd.DataFrame({"ts": snapshots})
    frame["hour_utc"] = frame["ts"].dt.hour
    frame["date"] = frame["ts"].dt.date

    # Every calendar day between the first and last snapshot counts, including days that
    # were missed entirely: leaving them out would flatter the coverage figure.
    span_days = max((snapshots.iloc[-1].date() - snapshots.iloc[0].date()).days + 1, 1)
    grouped = frame.groupby("hour_utc").agg(snapshots=("ts", "size"), days_seen=("date", "nunique"))
    grouped = grouped.reindex(range(24), fill_value=0)
    grouped["coverage"] = (grouped["snapshots"] / (EXPECTED_PER_HOUR * span_days)).round(3)
    return grouped.reset_index()


def controller_presence(data_dir: str | Path) -> pd.DataFrame:
    """Working controllers per snapshot, excluding observers.

    Observers connect on the placeholder frequency and are flagged at collection time; an
    observer online does not make a field controlled.

    Rows written before the collector had that flag are not assumed to be controllers:
    the flag is rederived from the frequency, exactly as the collector does it.
    """
    columns = ["snapshot_ts", "callsign", "frequency", "is_observer"]
    controllers = _read(data_dir, "controllers", columns)
    if controllers.empty:
        return pd.DataFrame(columns=["snapshot_ts", "controllers", "towers"])

    from pcas.collect.vatsim import OBSERVER_FREQUENCY

    flagged = controllers["is_observer"]
    backfill = controllers["frequency"] == OBSERVER_FREQUENCY
    is_observer = flagged.where(flagged.notna(), backfill).fillna(False).astype(bool)

    working = controllers[~is_observer]
    grouped = working.groupby("snapshot_ts").agg(
        controllers=("callsign", "nunique"),
        towers=("callsign", lambda s: s.str.endswith("_TWR").sum()),
    )
    return grouped.reset_index()


def summarise(data_dir: str | Path) -> dict[str, float | str]:
    snapshots = load_snapshots(data_dir)
    if snapshots.empty:
        return {"snapshots": 0.0}

    span_hours = (snapshots.iloc[-1] - snapshots.iloc[0]).total_seconds() / 3600.0
    holes = gaps(snapshots)
    return {
        "snapshots": float(len(snapshots)),
        "first": str(snapshots.iloc[0]),
        "last": str(snapshots.iloc[-1]),
        "span_hours": round(span_hours, 1),
        "collected_hours": round(len(snapshots) * POLL_SECONDS / 3600.0, 1),
        "duty_cycle": round(len(snapshots) * POLL_SECONDS / 3600.0 / span_hours, 3)
        if span_hours
        else float("nan"),
        "gaps_over_1min": float(len(holes)),
        "longest_gap_minutes": float(holes["minutes"].max()) if len(holes) else 0.0,
    }
