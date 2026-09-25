"""Build scenes from TrajAir's raw ADS-B CSVs.

Why not just use TrajAir's `processed_data`? Because its scene files are numbered
(`1.txt`, `2.txt`, ...) with no date, and its train/test split is random over those
scenes. We measured the consequence: all 7 days of `7days1` appear on BOTH sides of the
official split, so a model can be tested on the same day's traffic it trained on.

Reconstructing the date of each processed scene turned out to be guesswork (aircraft
recur across days, and range/altitude matching is ambiguous). The raw CSVs, by contrast,
are already one file per day with absolute timestamps. Rebuilding from them gives:

- real dates, so a split can hold out whole days
- absolute time, which conflict labelling needs to pair aircraft anyway
- control over filtering and interpolation, rather than inheriting choices we cannot see

TrajAir's processed data stays supported (see `trajair.py`) for one job only: reproducing
published numbers on the published split.

Raw columns: ID, Time (UTC), Date, Altitude (ft), Speed, Heading, Lat, Lon, Age, Range
(km from the airport), Bearing, Tail, Metar.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pcas.data.geo import FEET_TO_METERS, KBTP, LocalFrame
from pcas.data.trajair import Scene

# METAR wind group: 27012KT, 27012G20KT, VRB03KT, 00000KT.
_METAR_WIND = re.compile(r"\b(\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?KT\b")
KNOTS_TO_MS = 0.514444

# A track is cut here rather than interpolated across the hole.
MAX_GAP_S = 5
# Shorter fragments cannot fill even one window, so they are dropped early.
MIN_TRACK_S = 140
# Everything within this of the estimated field elevation counts as on the ground.
GROUND_MARGIN_M = 30.0
# The receiver hears traffic out to ~110 km, but enroute aircraft passing overhead are not
# terminal-area traffic: they would pad scenes with "interacting" pairs tens of km apart.
MAX_RANGE_M = 15_000.0
# Some transponders keep reporting a frozen position. Interpolating those gives a perfectly
# motionless "aircraft" that any predictor nails, flattering every metric.
MIN_PATH_M = 200.0


@dataclass(frozen=True)
class RawDay:
    """One day of raw ADS-B, cleaned and resampled onto a 1 Hz grid."""

    date: str
    tracks: pd.DataFrame  # ts, agent_id, track_id, x_m, y_m, z_m
    wind: pd.DataFrame  # ts, windx, windy (m/s, frame-aligned)
    field_elev_m: float


def parse_metar_wind(metar: str, runway_heading_deg: float) -> tuple[float, float]:
    """METAR wind to (windx, windy) m/s in the runway frame.

    Meteorological convention: the direction is where the wind blows FROM, so the vector
    it pushes an aircraft along is the reverse. Variable ("VRB") and calm winds have no
    usable direction, so they come back as zero - the same thing TrajAir does, which is
    why a fifth of its scenes carry a zero wind vector.
    """
    match = _METAR_WIND.search(str(metar))
    if not match:
        return 0.0, 0.0

    direction, speed_kt = match.group(1), int(match.group(2))
    if direction == "VRB" or speed_kt == 0:
        return 0.0, 0.0

    speed = speed_kt * KNOTS_TO_MS
    # Unit vector the air is moving towards, in east/north.
    blowing_to = math.radians(float(direction) + 180.0)
    east, north = speed * math.sin(blowing_to), speed * math.cos(blowing_to)

    theta = math.radians(runway_heading_deg)
    along = east * math.sin(theta) + north * math.cos(theta)
    left = -east * math.cos(theta) + north * math.sin(theta)
    return along, left


def read_raw_day(
    path: str | Path,
    frame: LocalFrame = KBTP,
    max_range_m: float = MAX_RANGE_M,
    field_elev_m: float | None = None,
) -> RawDay:
    """Read one raw day CSV into cleaned, 1 Hz, runway-frame tracks."""
    path = Path(path)
    raw = pd.read_csv(
        path,
        low_memory=False,
        usecols=["ID", "Time", "Date", "Altitude", "Lat", "Lon", "Metar"],
    )
    raw = raw.dropna(subset=["Lat", "Lon", "Altitude", "Time", "Date"])

    ts = pd.to_datetime(
        raw["Date"].astype(str) + " " + raw["Time"].astype(str),
        format="%m/%d/%Y %H:%M:%S.%f",
        utc=True,
        errors="coerce",
    )
    raw = raw.assign(ts=ts).dropna(subset=["ts"])
    if raw.empty:
        raise ValueError(f"{path}: no parsable rows")

    date = raw["ts"].dt.strftime("%Y-%m-%d").mode().iloc[0]
    alt_m = raw["Altitude"].astype(float) * FEET_TO_METERS

    # The raw altitudes are not reliably MSL (values below KBTP's published field
    # elevation appear for aircraft in the pattern), so the ground level is estimated
    # from the data instead of assumed: the low percentile of altitude among samples
    # close to the airport.
    x, y = _local_xy(raw, frame)
    if field_elev_m is not None:
        field_elev = field_elev_m
    else:
        near = np.hypot(x, y) < 1000.0
        field_elev = float(np.percentile(alt_m[near], 2)) if near.any() else float(alt_m.min())

    points = pd.DataFrame(
        {
            "ts": raw["ts"].to_numpy(),
            "agent_id": raw["ID"].astype("string").to_numpy(),
            "x_m": x,
            "y_m": y,
            "z_m": alt_m.to_numpy(),
        }
    )
    points = points[points["z_m"] > field_elev + GROUND_MARGIN_M]
    points = points[np.hypot(points["x_m"], points["y_m"]) <= max_range_m]
    points = points.sort_values(["agent_id", "ts"]).drop_duplicates(["agent_id", "ts"])

    tracks = _resample_tracks(points)
    wind = _wind_series(raw, frame.runway_heading_deg)

    return RawDay(date=date, tracks=tracks, wind=wind, field_elev_m=field_elev)


_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _epoch_seconds(ts: pd.Series) -> np.ndarray:
    """Seconds since the epoch, independent of the column's time resolution.

    `ts.astype("int64")` is NOT safe here: pandas may parse these timestamps at
    microsecond rather than nanosecond resolution, in which case that cast returns
    microseconds and every gap looks 1000x smaller than it is.
    """
    return (ts - _EPOCH).dt.total_seconds().to_numpy()


def _local_xy(raw: pd.DataFrame, frame: LocalFrame) -> tuple[np.ndarray, np.ndarray]:
    x, y, _ = frame.to_local(
        raw["Lat"].to_numpy(dtype=float),
        raw["Lon"].to_numpy(dtype=float),
        np.zeros(len(raw)),
    )
    return x, y


def _resample_tracks(points: pd.DataFrame) -> pd.DataFrame:
    """Split each aircraft's samples at gaps, then interpolate onto whole seconds."""
    out: list[pd.DataFrame] = []

    for agent_id, group in points.groupby("agent_id", sort=True):
        seconds = _epoch_seconds(group["ts"])
        breaks = np.flatnonzero(np.diff(seconds) > MAX_GAP_S) + 1

        for n, piece in enumerate(np.split(np.arange(len(group)), breaks)):
            if len(piece) < 2:
                continue
            t = seconds[piece]
            if t[-1] - t[0] < MIN_TRACK_S:
                continue

            px, py = group["x_m"].to_numpy()[piece], group["y_m"].to_numpy()[piece]
            if np.hypot(np.diff(px), np.diff(py)).sum() < MIN_PATH_M:
                continue

            grid = np.arange(math.ceil(t[0]), math.floor(t[-1]) + 1)
            frame = pd.DataFrame(
                {
                    "ts": pd.to_datetime(grid, unit="s", utc=True),
                    "agent_id": agent_id,
                    "track_id": f"{agent_id}_{n}",
                    "x_m": np.interp(grid, t, group["x_m"].to_numpy()[piece]),
                    "y_m": np.interp(grid, t, group["y_m"].to_numpy()[piece]),
                    "z_m": np.interp(grid, t, group["z_m"].to_numpy()[piece]),
                }
            )
            out.append(frame)

    if not out:
        return pd.DataFrame(columns=["ts", "agent_id", "track_id", "x_m", "y_m", "z_m"])
    return pd.concat(out, ignore_index=True).sort_values(["ts", "agent_id"], ignore_index=True)


def _wind_series(raw: pd.DataFrame, runway_heading_deg: float) -> pd.DataFrame:
    """One wind vector per distinct METAR, timestamped by when it first appears."""
    metars = raw.dropna(subset=["Metar"]).drop_duplicates("Metar")[["ts", "Metar"]]
    winds = [parse_metar_wind(m, runway_heading_deg) for m in metars["Metar"]]
    return pd.DataFrame(
        {
            "ts": metars["ts"].to_numpy(),
            "windx": [w[0] for w in winds],
            "windy": [w[1] for w in winds],
        }
    ).sort_values("ts", ignore_index=True)


def day_to_scenes(day: RawDay, min_agents: int = 1) -> list[Scene]:
    """Split a day into scenes: stretches where at least `min_agents` aircraft are up.

    Scene ids carry the date (`2020-09-18_0731`), so the day-based splits in `splits.py`
    can read it straight off the name.
    """
    tracks = day.tracks
    if tracks.empty:
        return []

    per_second = tracks.groupby("ts")["agent_id"].nunique()
    busy = per_second[per_second >= min_agents].index.to_series().sort_values()
    if busy.empty:
        return []

    # Consecutive busy seconds form one scene.
    gaps = busy.diff().dt.total_seconds().fillna(1.0) > 1.0
    blocks = gaps.cumsum()

    scenes: list[Scene] = []
    for _, block in busy.groupby(blocks):
        start, end = block.iloc[0], block.iloc[-1]
        window = tracks[(tracks["ts"] >= start) & (tracks["ts"] <= end)].copy()
        if window.empty:
            continue

        window["frame"] = (window["ts"] - start).dt.total_seconds().round().astype("int64")
        wind = _wind_at(day.wind, start)
        window["windx"], window["windy"] = wind

        frames = window[["frame", "agent_id", "windx", "windy", "x_m", "y_m", "z_m"]].sort_values(
            ["frame", "agent_id"], ignore_index=True
        )
        scenes.append(
            Scene(
                scene_id=f"{day.date}_{start.strftime('%H%M')}",
                path=Path(f"{day.date}_{start.strftime('%H%M')}"),
                date=day.date,
                frames=frames,
            )
        )

    return scenes


def _wind_at(wind: pd.DataFrame, ts: pd.Timestamp) -> tuple[float, float]:
    """Most recent wind observation at or before `ts`; zero if none precedes it."""
    if wind.empty:
        return 0.0, 0.0
    earlier = wind[wind["ts"] <= ts]
    row = (earlier if not earlier.empty else wind).iloc[-1 if not earlier.empty else 0]
    return float(row["windx"]), float(row["windy"])
