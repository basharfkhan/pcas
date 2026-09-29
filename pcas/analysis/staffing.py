"""Does a controller's presence change how close aircraft get at a field?

This is the one question in PCAS that the ADS-B datasets cannot answer. TartanAviation
records a single towered field, so controller presence is a constant there. The VATSIM
feed reports every staffed position alongside every aircraft position, which turns
staffing into a variable and gives a natural experiment: the same field, the same kind of
traffic, sometimes with a tower online and sometimes without.

**The confound is the whole problem.** Controllers do not appear at random. They log on to
busy fields at busy times, and more aircraft in a volume mechanically means more pairs and
more close passes. A raw comparison of staffed against unstaffed minutes therefore
measures traffic, not control, and it will do so with a confident-looking number. Every
rate here is computed inside strata of (hour of day, aircraft in the volume) and only then
pooled. Strata seen in only one arm contribute nothing.

**What is measured** is the share of aircraft pairs near a field that are closer than a
separation threshold at the same instant. Not a collision risk, and not an outcome a
controller is graded on: it is the density of close convergences a pilot would want to
hear about, which is the quantity PCAS predicts.

**What this is not.** VATSIM is a simulation network, so the result describes the
behaviour of pilots flying online, not the US airspace system. That limit is real and is
stated rather than argued away. What survives it is the direction of the effect, measured
on data nobody has curated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pcas.eval.conflicts import FEET, NM, NMAC, PROXIMITY, ConflictCriterion

EARTH_R = 6_371_000.0

# A field's traffic pattern sits inside a few miles of the runway and below ~1,500 ft AGL.
# The volume is larger than that so the arrival and departure funnel is included, which is
# where a controller's sequencing does most of its work, while en-route traffic cruising
# overhead is not.
FIELD_RADIUS_M = 10 * NM
MAX_AGL_FT = 5_000.0

# Airborne only, and this is not a detail. Aircraft parked at adjacent gates are a few
# hundred metres apart at identical altitude, so a volume that includes the ramp reports
# every airport apron as one continuous close encounter: the first run of this analysis
# came back with 399 close pairs per 1,000, four hundred times what the airborne data
# gives, because most "encounters" were aeroplanes standing still next to each other.
# Two gates are required, since either one alone lets something through: a stationary
# aircraft holding short is still on the ground, and an aircraft rolling on a long runway
# can exceed a rotation speed while its altitude is still field elevation.
MIN_AGL_FT = 200.0
AIRBORNE_KT = 40.0

# Separation gates, reported together rather than one at a time.
#
# The feed samples every 15 seconds. Two aircraft closing at 200 kt cover 0.8 nm between
# consecutive samples, so asking whether they were ever inside 0.5 nm *at a sample instant*
# throws away most of the encounters that happened: on 40 files of real data it found 452
# pairs and zero violations, which is not a small effect but an unmeasurable one. Wider
# gates are not a weaker question, they are the question this sampling rate can answer, and
# reporting the whole curve is what shows the answer does not depend on where the line is
# drawn.
NEAR = ConflictCriterion(horizontal_m=1 * NM, vertical_m=1_000 * FEET, name="near_1nm")
LOOSE = ConflictCriterion(horizontal_m=2 * NM, vertical_m=1_000 * FEET, name="loose_2nm")
WIDE = ConflictCriterion(horizontal_m=3 * NM, vertical_m=2_000 * FEET, name="wide_3nm")
CRITERIA = (NMAC, PROXIMITY, NEAR, LOOSE, WIDE)

# Controller callsign suffixes that actually separate traffic at or around a field.
TOWER_POSITIONS = frozenset({"TWR"})
APPROACH_POSITIONS = frozenset({"APP", "DEP"})

# A field position needs this many distinct parked flights before its coordinates are
# trusted, and their spread has to be small enough to be one airport rather than two.
MIN_FIELD_SAMPLES = 20
MAX_FIELD_SPREAD_M = 4_000.0

STATIONARY_KT = 5.0
MAX_PARKED_ALT_FT = 15_000.0

FLIGHT_KEY = ["cid", "callsign", "logon_time"]
PARKED_COLUMNS = [
    "cid",
    "callsign",
    "logon_time",
    "departure",
    "latitude",
    "longitude",
    "altitude",
    "snapshot_ts",
]


def _offsets_m(
    lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float
) -> tuple[np.ndarray, np.ndarray]:
    """East/north metres from a reference point, flat earth. Fine over a few miles."""
    north = np.radians(np.asarray(lat, dtype=float) - lat0) * EARTH_R
    east = np.radians(np.asarray(lon, dtype=float) - lon0) * EARTH_R * np.cos(np.radians(lat0))
    return east, north


def first_stationary_samples(pilots: pd.DataFrame) -> pd.DataFrame:
    """One row per flight: its earliest sample while stopped on the ground.

    The flight key is (cid, callsign, logon_time), so one pilot flying twice in the
    collection window contributes two rows rather than one.
    """
    if pilots.empty:
        return pd.DataFrame(columns=PARKED_COLUMNS)

    departure = pilots["departure"].astype("string").str.upper()
    stopped = pilots[
        (pilots["groundspeed"] <= STATIONARY_KT)
        & (pilots["altitude"] < MAX_PARKED_ALT_FT)
        & departure.notna()
        & (departure.str.len() == 4)
    ]
    if stopped.empty:
        return pd.DataFrame(columns=PARKED_COLUMNS)

    stopped = stopped.assign(departure=departure[stopped.index])[PARKED_COLUMNS]
    stopped = stopped.sort_values("snapshot_ts")
    return stopped.drop_duplicates(subset=FLIGHT_KEY, keep="first")


def estimate_field_positions(
    parked: pd.DataFrame,
    min_samples: int = MIN_FIELD_SAMPLES,
    max_spread_m: float = MAX_FIELD_SPREAD_M,
) -> pd.DataFrame:
    """Airport coordinates and elevations derived from the feed itself.

    There is no airport database in this project and adding one would mean trusting a
    third file to agree with the feed's identifiers. Instead: an aircraft's first
    stationary sample of a flight is sitting on its departure field, so the median of
    those positions is that field's position and the median of their altitudes is its
    elevation.

    Taking the *first* stationary sample matters. An aircraft parked at the end of a flight
    is also stationary, but it is at the arrival field while its `departure` still names
    the origin, so pooling all stationary samples would place every airport somewhere
    between the two.

    Fields whose samples are too few or too scattered are dropped rather than repaired: a
    scattered cluster means the identifier covers more than one place, and every downstream
    number would inherit the error silently.
    """
    columns = ["icao", "lat", "lon", "elevation_ft", "flights", "spread_m"]
    if parked.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for icao, group in parked.groupby("departure", sort=True):
        if len(group) < min_samples:
            continue
        lat0 = float(group["latitude"].median())
        lon0 = float(group["longitude"].median())
        east, north = _offsets_m(
            group["latitude"].to_numpy(), group["longitude"].to_numpy(), lat0, lon0
        )
        spread = float(np.median(np.hypot(east, north)))
        if spread > max_spread_m:
            continue
        rows.append(
            {
                "icao": str(icao),
                "lat": lat0,
                "lon": lon0,
                "elevation_ft": float(group["altitude"].median()),
                "flights": int(len(group)),
                "spread_m": round(spread, 1),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def parse_controller_positions(controllers: pd.DataFrame) -> pd.DataFrame:
    """Which field each staffed position covers, and what kind of position it is.

    VATSIM callsigns are `IDENT_POSITION`, sometimes with a sector letter in between
    (`SFO_N_TWR`). US idents usually drop the leading K, so `BTP_TWR` and `KBTP_TWR` are
    the same tower; both spellings are emitted and the caller keeps whichever matches a
    field it knows about.

    Observers are excluded by the caller, not here, because a snapshot with only an
    observer online is an unstaffed snapshot and has to stay in the comparison as one.
    """
    columns = ["snapshot_ts", "ident", "position"]
    if controllers.empty:
        return pd.DataFrame(columns=columns)

    parts = controllers["callsign"].astype(str).str.upper().str.split("_")
    frame = pd.DataFrame(
        {
            "snapshot_ts": controllers["snapshot_ts"].to_numpy(),
            "ident": parts.str[0].to_numpy(),
            "position": parts.str[-1].to_numpy(),
        }
    )[(parts.str.len() >= 2).to_numpy()]

    # Emit the K-prefixed spelling alongside the bare one so a 3-letter ident can match a
    # 4-letter field identifier without a lookup table.
    short = frame[frame["ident"].str.len() == 3].copy()
    short["ident"] = "K" + short["ident"]
    return pd.concat([frame, short], ignore_index=True).drop_duplicates()


def staffing_by_snapshot(positions: pd.DataFrame, fields: pd.Series) -> pd.DataFrame:
    """Per (field, snapshot): is a tower online, is approach online.

    Only fields present in `fields` are considered, so idents that match nothing in the
    traffic data are discarded rather than counted as staffing something.
    """
    columns = ["icao", "snapshot_ts", "tower", "approach"]
    if positions.empty:
        return pd.DataFrame(columns=columns)

    known = positions[positions["ident"].isin(set(fields))]
    if known.empty:
        return pd.DataFrame(columns=columns)

    known = known.assign(
        tower=known["position"].isin(TOWER_POSITIONS),
        approach=known["position"].isin(APPROACH_POSITIONS),
    )
    grouped = known.groupby(["ident", "snapshot_ts"], sort=False)[["tower", "approach"]].any()
    return grouped.reset_index().rename(columns={"ident": "icao"})[columns]


def near_field_traffic(
    pilots: pd.DataFrame,
    fields: pd.DataFrame,
    radius_m: float = FIELD_RADIUS_M,
    max_agl_ft: float = MAX_AGL_FT,
    min_agl_ft: float = MIN_AGL_FT,
    airborne_kt: float = AIRBORNE_KT,
) -> pd.DataFrame:
    """Airborne aircraft inside each field's volume: one row per aircraft per snapshot per field.

    An aircraft between two close airports belongs to both volumes. That is intended: the
    unit of analysis is the field, and traffic near a field is traffic near that field
    whoever else can also see it.

    Ground traffic is excluded on both altitude and speed. See `MIN_AGL_FT` for what
    happens when it is not.
    """
    columns = ["icao", "snapshot_ts", "cid", "east_m", "north_m", "altitude", "agl_ft"]
    if pilots.empty or fields.empty:
        return pd.DataFrame(columns=columns)

    flying = pilots[pilots["groundspeed"] >= airborne_kt]
    if flying.empty:
        return pd.DataFrame(columns=columns)

    chunks = []
    for field in fields.itertuples(index=False):
        # Bounding box first: a degree of latitude is ~111 km, so this cheap filter removes
        # almost everything before any trigonometry runs.
        dlat = np.degrees(radius_m / EARTH_R)
        dlon = dlat / max(np.cos(np.radians(field.lat)), 1e-6)
        box = flying[
            flying["latitude"].between(field.lat - dlat, field.lat + dlat)
            & flying["longitude"].between(field.lon - dlon, field.lon + dlon)
            & (flying["altitude"] < field.elevation_ft + max_agl_ft)
            & (flying["altitude"] >= field.elevation_ft + min_agl_ft)
        ]
        if box.empty:
            continue

        east, north = _offsets_m(
            box["latitude"].to_numpy(), box["longitude"].to_numpy(), field.lat, field.lon
        )
        inside = np.hypot(east, north) <= radius_m
        if not inside.any():
            continue

        altitude = box["altitude"].to_numpy()[inside]
        chunks.append(
            pd.DataFrame(
                {
                    "icao": field.icao,
                    "snapshot_ts": box["snapshot_ts"].to_numpy()[inside],
                    "cid": box["cid"].to_numpy()[inside],
                    "east_m": east[inside],
                    "north_m": north[inside],
                    "altitude": altitude,
                    "agl_ft": altitude - field.elevation_ft,
                }
            )
        )
    if not chunks:
        return pd.DataFrame(columns=columns)
    return pd.concat(chunks, ignore_index=True)[columns]


def encounter_panel(
    traffic: pd.DataFrame, criteria: tuple[ConflictCriterion, ...] = CRITERIA
) -> pd.DataFrame:
    """Per (field, snapshot): aircraft present, pairs formed, and pairs inside each gate.

    Every gate is counted in the same pass, plus the closest horizontal separation actually
    observed among pairs that were also within 1,000 ft vertically. Reporting the curve
    rather than one threshold is what keeps the conclusion from depending on a choice made
    before the data was seen.

    Snapshots with one aircraft form no pairs and cannot produce an encounter, but they are
    kept with `pairs = 0`: dropping them would remove exactly the quiet minutes that make
    the busy ones interpretable.
    """
    count_columns = [f"close_{c.name}" for c in criteria]
    columns = ["icao", "snapshot_ts", "aircraft", "pairs", *count_columns, "min_sep_m"]
    if traffic.empty:
        return pd.DataFrame(columns=columns)

    counts = (
        traffic.groupby(["icao", "snapshot_ts"], sort=False)
        .agg(aircraft=("cid", "nunique"))
        .reset_index()
    )
    counts["pairs"] = counts["aircraft"] * (counts["aircraft"] - 1) // 2

    multi = counts.loc[counts["aircraft"] > 1, ["icao", "snapshot_ts"]]
    if multi.empty:
        for column in count_columns:
            counts[column] = 0
        return counts.assign(min_sep_m=np.nan)[columns]

    busy = traffic.merge(multi, on=["icao", "snapshot_ts"], how="inner")
    busy = busy.drop_duplicates(subset=["icao", "snapshot_ts", "cid"])
    busy = busy.sort_values(["icao", "snapshot_ts", "cid"])
    busy["slot"] = busy.groupby(["icao", "snapshot_ts"], sort=False).cumcount()

    joined = busy.merge(busy, on=["icao", "snapshot_ts"], suffixes=("_a", "_b"))
    joined = joined[joined["slot_a"] < joined["slot_b"]]

    horizontal = np.hypot(
        joined["east_m_a"].to_numpy() - joined["east_m_b"].to_numpy(),
        joined["north_m_a"].to_numpy() - joined["north_m_b"].to_numpy(),
    )
    vertical = np.abs(joined["altitude_a"].to_numpy() - joined["altitude_b"].to_numpy()) * FEET

    for criterion in criteria:
        joined[f"close_{criterion.name}"] = criterion.violated(horizontal, vertical)
    # A horizontal distance only means anything for aircraft at comparable altitudes; two
    # aeroplanes a mile apart laterally and 4,000 ft apart vertically are not close.
    joined["sep_m"] = np.where(vertical < 1_000 * FEET, horizontal, np.nan)

    aggregated = joined.groupby(["icao", "snapshot_ts"], sort=False).agg(
        **{column: (column, "sum") for column in count_columns},
        min_sep_m=("sep_m", "min"),
    )
    panel = counts.merge(aggregated.reset_index(), on=["icao", "snapshot_ts"], how="left")
    for column in count_columns:
        panel[column] = panel[column].fillna(0).astype(int)
    return panel[columns]


DEFAULT_METRIC = f"close_{NEAR.name}"


def build_panel(
    traffic: pd.DataFrame,
    staffing: pd.DataFrame,
    criteria: tuple[ConflictCriterion, ...] = CRITERIA,
) -> pd.DataFrame:
    """The analysis table: one row per field per snapshot, with staffing attached.

    A (field, snapshot) with no controller row is unstaffed, not missing. That is the
    correct reading of the feed, which lists only positions that are online.
    """
    panel = encounter_panel(traffic, criteria)
    if panel.empty:
        return panel.assign(tower=False, approach=False, hour=0)

    if staffing.empty:
        panel = panel.assign(tower=np.nan, approach=np.nan)
    else:
        panel = panel.merge(staffing, on=["icao", "snapshot_ts"], how="left")
    panel["tower"] = panel["tower"].fillna(False).astype(bool)
    panel["approach"] = panel["approach"].fillna(False).astype(bool)
    ts = pd.to_datetime(panel["snapshot_ts"], utc=True, format="ISO8601")
    panel["hour"] = ts.dt.hour
    return panel


def crude_rates(
    panel: pd.DataFrame, arm: str = "tower", metric: str = DEFAULT_METRIC
) -> pd.DataFrame:
    """The comparison anyone would run first, kept so the confound can be shown."""
    out = panel.groupby(panel[arm]).agg(
        snapshots=("pairs", "size"),
        aircraft_mean=("aircraft", "mean"),
        pairs=("pairs", "sum"),
        close_pairs=(metric, "sum"),
        median_sep_m=("min_sep_m", "median"),
    )
    out["rate_per_1k_pairs"] = 1000 * out["close_pairs"] / out["pairs"].replace(0, np.nan)
    return out.reset_index().rename(columns={arm: "staffed"})


def stratified_rates(
    panel: pd.DataFrame,
    arm: str = "tower",
    metric: str = DEFAULT_METRIC,
    traffic_bins: tuple[int, ...] = (2, 3, 4, 6, 9, 10_000),
    hour_bins: int = 4,
    by_field: bool = False,
) -> pd.DataFrame:
    """Close-pair rate by stratum, for the two arms side by side.

    Strata are (aircraft in the volume, hour-of-day block). Both are needed: aircraft count
    is the mechanical driver of close pairs, and hour of day carries everything about when
    people fly and when controllers log on that the count does not.

    With `by_field`, the field joins the stratum and the comparison becomes strictly
    within-field: the same airport, the same traffic level, the same time of day, staffed
    against unstaffed. That is the actual natural experiment, and without it a field that is
    always staffed is being compared against a different field that never is, which is a
    comparison between airports wearing the clothes of a comparison between staffing.
    """
    keys = ["aircraft_bin", "hour_bin"]
    if by_field:
        keys = ["icao", *keys]
    columns = [*keys, "pairs_off", "pairs_on", "rate_off", "rate_on"]
    busy = panel[panel["pairs"] > 0].copy()
    if busy.empty:
        return pd.DataFrame(columns=columns)

    busy["aircraft_bin"] = pd.cut(busy["aircraft"], bins=(1, *traffic_bins), right=True)
    block = max(24 // hour_bins, 1)
    busy["hour_bin"] = (busy["hour"] // block) * block

    grouped = busy.groupby([*keys, busy[arm]], observed=True).agg(
        pairs=("pairs", "sum"), close_pairs=(metric, "sum")
    )
    wide = grouped.unstack(arm)
    wide.columns = [f"{stat}_{'on' if staffed else 'off'}" for stat, staffed in wide.columns]
    for column in ("pairs_off", "pairs_on", "close_pairs_off", "close_pairs_on"):
        if column not in wide:
            wide[column] = 0.0
    wide = wide.fillna(0.0)
    wide["rate_off"] = 1000 * wide["close_pairs_off"] / wide["pairs_off"].replace(0, np.nan)
    wide["rate_on"] = 1000 * wide["close_pairs_on"] / wide["pairs_on"].replace(0, np.nan)
    return wide.reset_index()


def placebo_ratios(
    panel: pd.DataFrame,
    arm: str = "tower",
    metric: str = DEFAULT_METRIC,
    seeds: int = 5,
    hour_block: int = 6,
) -> list[float]:
    """Re-run the comparison with the staffing label shuffled inside each stratum.

    If the pooling machinery can produce a ratio far from 1 from labels that carry no
    information, then the real ratio says nothing either. This is the cheapest available
    check on the whole chain, and it tests the code as well as the design: a bug in the
    weighting would show up here as a placebo effect.
    """
    shuffled = panel.copy()
    shuffled["hour_block"] = (shuffled["hour"] // max(hour_block, 1)) * max(hour_block, 1)
    ratios = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)

        def shuffle(column: pd.Series, rng: np.random.Generator = rng) -> np.ndarray:
            return rng.permutation(column.to_numpy())

        shuffled[arm] = shuffled.groupby(["icao", "aircraft", "hour_block"])[arm].transform(shuffle)
        pooled = standardised_comparison(
            stratified_rates(shuffled, arm=arm, metric=metric, by_field=True)
        )
        ratios.append(pooled.get("ratio_on_over_off", float("nan")))
    return ratios


def standardised_comparison(strata: pd.DataFrame) -> dict[str, float]:
    """Pool the strata onto one number, weighting by the evidence both arms bring.

    The weight is the harmonic mean of the two arms' pair counts, as in Mantel-Haenszel
    pooling. It is the natural choice here: a stratum contributes in proportion to how much
    *both* arms know about it, so a stratum that is almost entirely staffed cannot dominate
    the answer with an unstaffed rate estimated from a handful of pairs.
    """
    if strata.empty:
        return {"strata": 0.0}

    usable = strata.dropna(subset=["rate_off", "rate_on"])
    usable = usable[(usable["pairs_off"] > 0) & (usable["pairs_on"] > 0)]
    if usable.empty:
        return {"strata": 0.0}

    weight = (usable["pairs_off"] * usable["pairs_on"]) / (usable["pairs_off"] + usable["pairs_on"])
    rate_off = float((usable["rate_off"] * weight).sum() / weight.sum())
    rate_on = float((usable["rate_on"] * weight).sum() / weight.sum())
    return {
        "strata": float(len(usable)),
        "pairs_off": float(usable["pairs_off"].sum()),
        "pairs_on": float(usable["pairs_on"].sum()),
        "rate_off_per_1k": round(rate_off, 3),
        "rate_on_per_1k": round(rate_on, 3),
        "ratio_on_over_off": round(rate_on / rate_off, 3) if rate_off else float("nan"),
        "strata_with_lower_rate_staffed": float((usable["rate_on"] < usable["rate_off"]).sum()),
    }
