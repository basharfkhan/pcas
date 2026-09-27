"""Read TartanAviation ADS-B sessions.

TartanAviation is TrajAir's successor from the same lab: 660 recording days across two
airports instead of 111 at one, and the second airport is **towered**. That buys two things
this project could not otherwise get from real aircraft:

- a towered (KAGC) versus non-towered (KBTP) comparison, where the VATSIM feed can only
  offer simulated traffic;
- a second field, so "the model learned how traffic behaves" can be told apart from "the
  model memorised KBTP's geometry" by training on one and testing on the other.

Same measurements as TrajAir, different packaging, and the differences all need handling:

| | TrajAir | TartanAviation |
|---|---|---|
| Time | `06:56:37.639` | `[u'13', u'00', u'03.465']` (a Python 2 list repr) |
| Date | `09/18/2020` | `[u'2020', u'08', u'01']` |
| Weather | a `Metar` string on every row | separate Iowa Mesonet tables per airport |
| Files per day | one | one full-day file, or many per-encounter segments |
| Columns | fixed | `AltisGNSS` appears from 2021 |

Two quirks in the files themselves: a session folder's rows can cross midnight UTC (so the
folder date names the *recording session*, not every row in it), and at least one full-day
file is truncated mid-line because the recorder was killed.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pcas.data.adsb import MAX_RANGE_M, RawDay, build_raw_day
from pcas.data.geo import KBTP, LocalFrame

log = logging.getLogger(__name__)

# Receiver positions from the Iowa Mesonet tables shipped with the dataset, which also
# confirm the field elevations (KBTP 380 m, KAGC 382 m).
KAGC = LocalFrame(
    lat_deg=40.3547,
    lon_deg=-79.9217,
    alt_m=382.0,
    # Allegheny County's longer runway. The exact value matters less than consistency: it
    # only fixes which way the local x axis points, and every scene at an airport uses the
    # same one.
    runway_heading_deg=100.0,
)

AIRPORTS: dict[str, LocalFrame] = {"kbtp": KBTP, "kagc": KAGC}
WEATHER_STATIONS: dict[str, str] = {"kbtp": "BTP", "kagc": "AGC"}

_SESSION_DATE = re.compile(r"^(?P<m>\d{2})-(?P<d>\d{2})-(?P<y>\d{2})$")
# Pulls the three quoted parts out of "[u'13', u'00', u'03.465']".
_LIST_PARTS = re.compile(r"u?'([^']*)'")

KNOTS_TO_MS = 0.514444

# Reception varies by session: most are close to 1 Hz, but some KAGC days report only every
# 4 to 6 s, which cannot support 1 Hz tracks. Such sessions are read but flagged.
MAX_USABLE_GAP_S = 2.0


def find_sessions(root: str | Path, airport: str | None = None) -> dict[str, list[Path]]:
    """ISO session date -> the CSV files recorded in it.

    Extracted archives are laid out as `<root>/<airport>_raw_<year>/<MM-DD-YY>/*.csv`.
    Sessions from different years cannot collide, so one flat mapping is enough.
    """
    root = Path(root)
    sessions: dict[str, list[Path]] = {}

    for archive in sorted(root.glob("*_raw_*")):
        if not archive.is_dir():
            continue
        if airport and not archive.name.startswith(f"{airport}_"):
            continue

        for folder in sorted(archive.iterdir()):
            match = _SESSION_DATE.match(folder.name) if folder.is_dir() else None
            if not match:
                continue
            csvs = [p for p in sorted(folder.glob("*.csv")) if p.stat().st_size > 0]
            if csvs:
                sessions.setdefault(f"20{match['y']}-{match['m']}-{match['d']}", []).extend(csvs)

    return dict(sorted(sessions.items()))


def airport_of(path: Path) -> str:
    """Which airport a session path belongs to, from the archive folder name."""
    for part in path.parts:
        for code in AIRPORTS:
            if part.startswith(f"{code}_raw_"):
                return code
    raise ValueError(f"cannot tell which airport {path} belongs to")


def read_csv_tolerantly(path: str | Path) -> pd.DataFrame:
    """Read one session CSV, dropping lines the recorder left incomplete.

    At least one full-day file ends mid-field because the recorder was killed, which makes
    pandas raise "EOF inside string". Rather than lose the whole day, lines with unbalanced
    quotes are dropped first. That is cheap and only ever discards a truncated tail.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    kept = [line for line in lines if line.count('"') % 2 == 0]
    dropped = len(lines) - len(kept)
    if dropped:
        log.debug("%s: dropped %d incomplete line(s)", path.name, dropped)

    frame = pd.read_csv(io.StringIO("\n".join(kept)), low_memory=False)
    return frame


def parse_list_field(values: pd.Series, n_parts: int) -> pd.DataFrame:
    """Turn `"[u'2020', u'08', u'01']"` into numeric columns."""
    parts = values.astype("string").str.extractall(_LIST_PARTS)[0].unstack()
    parts = parts.reindex(columns=range(n_parts))
    return parts.apply(pd.to_numeric, errors="coerce")


def load_weather(root: str | Path, airport: str, runway_heading_deg: float) -> pd.DataFrame:
    """Wind for one airport, as runway-frame components in m/s.

    The dataset ships Iowa Mesonet observations rather than raw METAR strings, so direction
    (`drct`) and speed (`sknt`) arrive as numbers. Missing values are the string "M", and
    calm or variable winds have no usable direction, so both become zero, matching what the
    TrajAir path does with `VRB` and `00000KT`.
    """
    station = WEATHER_STATIONS[airport]
    path = Path(root) / "weather" / f"{station}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; fetch weather/{station}.csv from the TartanAviation repo"
        )

    table = pd.read_csv(path, usecols=["valid", "drct", "sknt"], low_memory=False)
    ts = pd.to_datetime(table["valid"], utc=True, errors="coerce")
    direction = pd.to_numeric(table["drct"], errors="coerce")
    speed_kt = pd.to_numeric(table["sknt"], errors="coerce")

    usable = ts.notna()
    speed = (speed_kt.fillna(0.0) * KNOTS_TO_MS).to_numpy()
    # Meteorological convention: the direction is where the wind comes FROM.
    blowing_to = np.radians(direction.fillna(0.0).to_numpy() + 180.0)
    east, north = speed * np.sin(blowing_to), speed * np.cos(blowing_to)

    theta = np.radians(runway_heading_deg)
    along = east * np.sin(theta) + north * np.cos(theta)
    left = -east * np.cos(theta) + north * np.sin(theta)

    calm = direction.isna().to_numpy() | (speed <= 0)
    along = np.where(calm, 0.0, along)
    left = np.where(calm, 0.0, left)

    return (
        pd.DataFrame({"ts": ts[usable], "windx": along[usable], "windy": left[usable]})
        .sort_values("ts", ignore_index=True)
        .reset_index(drop=True)
    )


def read_session(
    paths: list[Path] | Path,
    date: str | None = None,
    airport: str | None = None,
    weather_root: str | Path | None = None,
    max_range_m: float = MAX_RANGE_M,
    field_elev_m: float | None = None,
) -> RawDay:
    """Read one recording session (all its CSVs) into cleaned, 1 Hz, runway-frame tracks."""
    paths = [Path(paths)] if isinstance(paths, Path | str) else [Path(p) for p in paths]
    if not paths:
        raise ValueError("no session files given")

    airport = airport or airport_of(paths[0])
    frame = AIRPORTS[airport]

    frames = [read_csv_tolerantly(p) for p in paths]
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.dropna(subset=["Lat", "Lon", "Altitude", "Time", "Date"])
    if raw.empty:
        raise ValueError(f"{paths[0]}: no usable rows")

    hms = parse_list_field(raw["Time"], 3)
    ymd = parse_list_field(raw["Date"], 3)
    ts = pd.to_datetime(
        {
            "year": ymd[0],
            "month": ymd[1],
            "day": ymd[2],
            "hour": hms[0],
            "minute": hms[1],
            "second": hms[2].astype(float).astype("int64", errors="ignore"),
        },
        errors="coerce",
        utc=True,
    )
    # Keep sub-second precision, which the integer-second construction above discards.
    fractional = hms[2].astype(float) % 1.0
    ts = ts + pd.to_timedelta(fractional.fillna(0.0), unit="s")

    raw = raw.assign(ts=ts).dropna(subset=["ts"])
    if raw.empty:
        raise ValueError(f"{paths[0]}: no parsable timestamps")

    # A session can cross midnight UTC, so the folder date names the recording session
    # rather than every row in it. Splits group by session, which keeps them leak-free.
    session_date = date or raw["ts"].dt.strftime("%Y-%m-%d").mode().iloc[0]

    if weather_root is not None:
        wind = load_weather(weather_root, airport, frame.runway_heading_deg)
    else:
        wind = pd.DataFrame(columns=["ts", "windx", "windy"])

    day = build_raw_day(
        ts=raw["ts"],
        agent_id=raw["ID"],
        lat=raw["Lat"],
        lon=raw["Lon"],
        altitude_ft=raw["Altitude"],
        wind=wind,
        date=session_date,
        frame=frame,
        max_range_m=max_range_m,
        field_elev_m=field_elev_m,
    )

    if day.median_gap_s > MAX_USABLE_GAP_S:
        log.warning(
            "%s %s: reports every %.1f s, too sparse for 1 Hz tracks (%d rows kept)",
            airport,
            session_date,
            day.median_gap_s,
            len(day.tracks),
        )
    return day
