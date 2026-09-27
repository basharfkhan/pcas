"""TartanAviation reader: its format quirks, and the defects in the distributed files."""

from __future__ import annotations

import pandas as pd
import pytest

from pcas.data.tartan import (
    KAGC,
    find_sessions,
    load_weather,
    parse_list_field,
    read_csv_tolerantly,
    read_session,
)

HEADER = "ID,Time,Date,Altitude,Speed,Heading,Lat,Lon,Age,Range,Bearing,Tail"


def row(aid, hours, minutes, seconds, lat, lon, alt=2000):
    """One raw row, with the Python 2 list reprs TartanAviation ships."""
    time_field = f"\"[u'{hours:02d}', u'{minutes:02d}', u'{seconds:06.3f}']\""
    date_field = "\"[u'2022', u'03', u'15']\""
    return f"{aid},{time_field},{date_field},{alt},,,{lat:.6f},{lon:.6f},1.0,1.0,0.0,N{aid}"


def write_session(folder, n_seconds=400, aircraft=(1001,), alt=2000):
    folder.mkdir(parents=True, exist_ok=True)
    lines = [HEADER]

    for i, aid in enumerate(aircraft):
        for step in range(n_seconds):
            t = 36000 + step
            lines.append(
                row(
                    aid,
                    t // 3600,
                    (t % 3600) // 60,
                    float(t % 60),
                    KAGC.lat_deg + 0.0001 * step + 0.005 * i,
                    KAGC.lon_deg + 0.0001 * step,
                    alt,
                )
            )

    # An aircraft parked at the field, so the field-elevation estimate has a floor.
    for step in range(60):
        t = 36000 + step
        lines.append(
            row(9999, t // 3600, (t % 3600) // 60, float(t % 60), KAGC.lat_deg, KAGC.lon_deg, 1250)
        )

    path = folder / "1.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_parse_list_field_handles_python2_reprs():
    values = pd.Series(["[u'2020', u'08', u'01']", "[u'2021', u'12', u'31']"])
    parts = parse_list_field(values, 3)
    assert parts[0].tolist() == [2020, 2021]
    assert parts[2].tolist() == [1, 31]


def test_binary_file_is_skipped_not_fatal(tmp_path):
    # One file in the distributed dataset is binary garbage, and it used to abort a run
    # hours in.
    junk = tmp_path / "23.csv"
    junk.write_bytes(bytes(range(256)) * 20)
    assert read_csv_tolerantly(junk) is None


def test_truncated_line_is_dropped(tmp_path):
    path = tmp_path / "1.csv"
    good = row(1001, 10, 0, 1.5, 40.35, -79.92)
    truncated = "1001,\"[u'10', u'00',"
    path.write_text(f"{HEADER}\n{good}\n{truncated}\n", encoding="utf-8")

    frame = read_csv_tolerantly(path)
    assert frame is not None
    assert len(frame) == 1


def test_session_with_only_bad_files_raises(tmp_path):
    folder = tmp_path / "kagc_raw_2022" / "03-15-22"
    folder.mkdir(parents=True)
    (folder / "1.csv").write_bytes(b"\x00\x01\x02not a csv")

    with pytest.raises(ValueError, match="no usable CSV"):
        read_session(list(folder.glob("*.csv")), date="2022-03-15", airport="kagc")


def test_session_reads_into_tracks(tmp_path):
    folder = tmp_path / "kagc_raw_2022" / "03-15-22"
    path = write_session(folder, aircraft=(1001, 1002))
    day = read_session([path], date="2022-03-15", airport="kagc")

    assert day.date == "2022-03-15"
    assert not day.tracks.empty
    assert set(day.tracks["agent_id"]) == {"1001", "1002"}
    # 1 Hz reports, so the quality gate should say so.
    assert day.median_gap_s == pytest.approx(1.0, abs=0.01)
    assert day.tracks["ts"].dt.tz is not None
    # The parked aircraft is filtered as ground traffic.
    assert "9999" not in set(day.tracks["agent_id"])


def test_find_sessions_groups_by_date_across_years(tmp_path):
    write_session(tmp_path / "kbtp_raw_2021" / "03-15-21")
    write_session(tmp_path / "kbtp_raw_2022" / "03-15-22")
    write_session(tmp_path / "kagc_raw_2022" / "04-16-22")

    assert sorted(find_sessions(tmp_path, "kbtp")) == ["2021-03-15", "2022-03-15"]
    assert sorted(find_sessions(tmp_path, "kagc")) == ["2022-04-16"]
    assert len(find_sessions(tmp_path)) == 3


def test_weather_is_converted_to_runway_components(tmp_path):
    weather = tmp_path / "weather"
    weather.mkdir()
    (weather / "AGC.csv").write_text(
        "station,valid,drct,sknt\n"
        "AGC,2022-03-15 10:00,100,10\n"  # straight down a 100 degree runway
        "AGC,2022-03-15 11:00,M,M\n"  # missing
        "AGC,2022-03-15 12:00,0,0\n",  # calm
        encoding="utf-8",
    )
    wind = load_weather(tmp_path, "kagc", 100.0)

    assert len(wind) == 3
    # Wind FROM 100 degrees on a 100 degree runway is a headwind: negative along-runway.
    assert wind["windx"].iloc[0] < -4.0
    assert abs(wind["windy"].iloc[0]) < 1e-6
    # Missing and calm both come back as zero, as the TrajAir path does for VRB and calm.
    assert wind.loc[1, ["windx", "windy"]].tolist() == [0.0, 0.0]
    assert wind.loc[2, ["windx", "windy"]].tolist() == [0.0, 0.0]


def test_missing_weather_file_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="AGC.csv"):
        load_weather(tmp_path, "kagc", 100.0)
