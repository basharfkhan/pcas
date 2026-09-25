"""Raw ADS-B pipeline: cleaning, resampling, wind, and scene construction."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pcas.data.adsb import (
    MIN_TRACK_S,
    day_to_scenes,
    parse_metar_wind,
    read_raw_day,
)
from pcas.data.geo import FEET_TO_METERS, KBTP
from pcas.data.scenes import build_windows
from pcas.data.trajair import parse_scene_date

HEADER = "ID,Time,Date,Altitude,Speed,Heading,Lat,Lon,Age,Range,Bearing,Tail,Metar"
CALM_METAR = "KBTP 181000Z AUTO 00000KT 10SM CLR 13/12 A3002"


def write_raw(
    path: Path,
    aircraft=((1001, 0, 300),),
    seconds=200,
    start="10:00:00",
    date="09/18/2020",
    metar=CALM_METAR,
    skip=(),
    alt_ft=2000,
    ground_ft=1100,
) -> Path:
    """Write a synthetic raw day CSV: each aircraft tracks north-east at ~1 Hz.

    `aircraft` entries are (id, lat_offset_index, ignored); positions stay within a few km
    of KBTP so nothing is dropped by the range filter.

    A stationary aircraft at `ground_ft` sits on the field, as in every real file. Without
    it the data-driven field-elevation estimate lands on the airborne traffic itself and
    filters everything out.
    """
    h, m, s = (int(v) for v in start.split(":"))
    base = h * 3600 + m * 60 + s
    lines = [HEADER]

    if ground_ft is not None:
        for step in range(0, 60):
            t = base + step
            stamp = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}.000"
            lines.append(
                f"9999,{stamp},{date},{ground_ft},,,{KBTP.lat_deg:.6f},"
                f"{KBTP.lon_deg:.6f},1.0,0.0,0.0,NPARKED,{metar}"
            )

    for i, (aid, *_rest) in enumerate(aircraft):
        for step in range(seconds):
            if step in skip:
                continue
            t = base + step
            # milliseconds present, as in the real files
            stamp = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}.{(step * 37) % 1000:03d}"
            lat = KBTP.lat_deg + 0.0001 * step + 0.01 * i
            lon = KBTP.lon_deg + 0.0001 * step
            lines.append(
                f"{aid},{stamp},{date},{alt_ft},,,{lat:.6f},{lon:.6f},1.0,1.0,0.0,N{aid},{metar}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- METAR wind ---------------------------------------------------------------


def test_calm_and_variable_winds_are_zero():
    assert parse_metar_wind("KBTP 181000Z AUTO 00000KT", 80.0) == (0.0, 0.0)
    assert parse_metar_wind("KBTP 181000Z AUTO VRB03KT", 80.0) == (0.0, 0.0)
    assert parse_metar_wind("no wind group here", 80.0) == (0.0, 0.0)


def test_wind_speed_is_converted_to_metres_per_second():
    wx, wy = parse_metar_wind("KBTP 181000Z 27010KT", 0.0)
    assert math.hypot(wx, wy) == pytest.approx(10 * 0.514444, rel=1e-6)


def test_wind_direction_is_where_it_blows_from():
    # Runway heading 090, wind FROM 270 means it pushes an aircraft along the runway:
    # a tailwind on that heading, so +x and no crosswind.
    wx, wy = parse_metar_wind("KBTP 181000Z 27010KT", 90.0)
    assert wx > 5.0
    assert abs(wy) < 1e-6

    # Wind from 090 on the same runway is a headwind: -x.
    wx, _ = parse_metar_wind("KBTP 181000Z 09010KT", 90.0)
    assert wx < -5.0


def test_gusts_use_the_sustained_speed():
    sustained = parse_metar_wind("KBTP 181000Z 27010KT", 0.0)
    gusting = parse_metar_wind("KBTP 181000Z 27010G25KT", 0.0)
    assert sustained == gusting


# --- reading a day ------------------------------------------------------------


def test_millisecond_timestamps_still_resolve_to_seconds(tmp_path):
    # Regression: pandas may parse these timestamps at microsecond resolution, so an
    # astype("int64") cast returns microseconds and every 1 s gap looks like 1 ms. That
    # made the segmenter treat a whole day as one too-short blob and drop everything.
    day = read_raw_day(write_raw(tmp_path / "1.csv", seconds=400))

    assert not day.tracks.empty
    spacing = np.diff(day.tracks["ts"].to_numpy()).astype("timedelta64[ms]").astype(int)
    assert set(spacing.tolist()) == {1000}


def test_tracks_are_resampled_to_whole_seconds(tmp_path):
    day = read_raw_day(write_raw(tmp_path / "1.csv", seconds=300))
    ts = day.tracks["ts"]

    assert (ts.dt.microsecond == 0).all()
    assert day.date == "2020-09-18"
    assert day.tracks["track_id"].nunique() == 1


def test_gap_splits_a_track_instead_of_interpolating_across_it(tmp_path):
    # A 30 s hole in the middle: two segments, and no sample inside the hole.
    path = write_raw(tmp_path / "1.csv", seconds=400, skip=set(range(180, 210)))
    day = read_raw_day(path)

    assert day.tracks["track_id"].nunique() == 2
    per_track = day.tracks.groupby("track_id")["ts"].agg(["min", "max"])
    first_end, second_start = per_track.iloc[0]["max"], per_track.iloc[1]["min"]
    assert (second_start - first_end).total_seconds() > 25


def test_short_fragments_are_dropped(tmp_path):
    day = read_raw_day(write_raw(tmp_path / "1.csv", seconds=MIN_TRACK_S - 20))
    assert day.tracks.empty


def test_ground_samples_are_dropped(tmp_path):
    path = write_raw(tmp_path / "1.csv", seconds=300)
    # The parked aircraft is never emitted as a track.
    assert "9999" not in set(read_raw_day(path).tracks["agent_id"])

    # Pin the field at the airborne traffic's own altitude: now nothing counts as flying.
    pinned = read_raw_day(path, field_elev_m=2000 * FEET_TO_METERS)
    assert pinned.tracks.empty


def test_distant_traffic_is_outside_the_terminal_area(tmp_path):
    path = write_raw(tmp_path / "1.csv", seconds=300)
    near = read_raw_day(path, max_range_m=15_000)
    far = read_raw_day(path, max_range_m=100.0)

    assert not near.tracks.empty
    assert far.tracks.empty


# --- scenes -------------------------------------------------------------------


def test_scene_ids_carry_the_date_so_splits_can_read_it(tmp_path):
    day = read_raw_day(write_raw(tmp_path / "1.csv", seconds=400))
    scenes = day_to_scenes(day)

    assert scenes
    assert parse_scene_date(scenes[0].scene_id) == "2020-09-18"
    assert scenes[0].date == "2020-09-18"


def test_scenes_feed_the_window_builder(tmp_path):
    path = write_raw(tmp_path / "1.csv", aircraft=((1001, 0, 0), (1002, 1, 0)), seconds=400)
    scenes = day_to_scenes(read_raw_day(path), min_agents=2)
    windows = [w for s in scenes for w in build_windows(s, stride=10)]

    assert windows
    two_up = [w for w in windows if w.n_agents == 2]
    assert two_up
    assert two_up[0].obs.shape == (2, 11, 3)
    assert two_up[0].future.shape == (2, 12, 3)


def test_min_agents_requires_simultaneous_traffic(tmp_path):
    solo = write_raw(tmp_path / "1.csv", seconds=400)
    assert day_to_scenes(read_raw_day(solo), min_agents=1)
    assert day_to_scenes(read_raw_day(solo), min_agents=2) == []


def test_scene_wind_comes_from_the_metar(tmp_path):
    path = write_raw(tmp_path / "1.csv", seconds=400, metar="KBTP 181000Z 09010KT 10SM CLR")
    scene = day_to_scenes(read_raw_day(path))[0]

    # Runway heading 80, wind from 090: nearly a straight headwind.
    assert scene.frames["windx"].iloc[0] < -4.0


def test_frozen_position_tracks_are_dropped(tmp_path):
    # A transponder reporting the same position for ten minutes is not an aircraft in
    # flight; interpolating it would give a motionless target any predictor gets right.
    path = tmp_path / "1.csv"
    lines = [HEADER]
    for step in range(600):
        t = 36000 + step
        stamp = f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}.000"
        alt = 1100 if step < 60 else 2000  # parked aircraft first, to anchor the field
        lat, lon = KBTP.lat_deg + 0.02, KBTP.lon_deg
        aid = 9999 if step < 60 else 1001
        lines.append(
            f"{aid},{stamp},09/18/2020,{alt},,,{lat:.6f},{lon:.6f},1.0,1.0,0.0,N{aid},{CALM_METAR}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert read_raw_day(path).tracks.empty
