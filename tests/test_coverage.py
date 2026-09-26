from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from pcas.collect.coverage import (
    EXPECTED_PER_HOUR,
    controller_presence,
    coverage_by_hour,
    gaps,
    load_snapshots,
    summarise,
)

START = datetime(2026, 9, 25, 0, 0, tzinfo=UTC)


def write_collected(root, snapshot_times, controllers=None):
    """Write parquet in the collector's own layout."""
    pilots = pd.DataFrame(
        {
            "snapshot_ts": [t.isoformat().replace("+00:00", "Z") for t in snapshot_times],
            "callsign": ["N172SP"] * len(snapshot_times),
        }
    )
    part = root / "pilots" / "date=2026-09-25"
    part.mkdir(parents=True, exist_ok=True)
    pilots.to_parquet(part / "pilots-1.parquet", index=False)

    if controllers is not None:
        cpart = root / "controllers" / "date=2026-09-25"
        cpart.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(controllers).to_parquet(cpart / "controllers-1.parquet", index=False)
    return root


def test_no_data_summarises_to_zero(tmp_path):
    assert summarise(tmp_path) == {"snapshots": 0.0}
    assert load_snapshots(tmp_path).empty


def test_snapshots_are_deduplicated_and_sorted(tmp_path):
    times = [START, START + timedelta(seconds=15), START]  # duplicate
    write_collected(tmp_path, times)
    snapshots = load_snapshots(tmp_path)
    assert len(snapshots) == 2
    assert snapshots.is_monotonic_increasing


def test_gaps_are_found_with_start_end_and_length(tmp_path):
    times = [START, START + timedelta(seconds=15), START + timedelta(minutes=40)]
    write_collected(tmp_path, times)

    holes = gaps(load_snapshots(tmp_path))
    assert len(holes) == 1
    assert holes.iloc[0]["minutes"] == pytest.approx(39.75, abs=0.1)


def test_continuous_collection_has_no_gaps(tmp_path):
    times = [START + timedelta(seconds=15 * i) for i in range(40)]
    write_collected(tmp_path, times)
    assert gaps(load_snapshots(tmp_path)).empty


def test_coverage_by_hour_flags_thin_hours(tmp_path):
    # Hour 0 fully collected, hour 1 only a quarter, the rest missing.
    times = [START + timedelta(seconds=15 * i) for i in range(EXPECTED_PER_HOUR)]
    times += [START + timedelta(hours=1, seconds=15 * i) for i in range(EXPECTED_PER_HOUR // 4)]
    write_collected(tmp_path, times)

    table = coverage_by_hour(load_snapshots(tmp_path)).set_index("hour_utc")
    assert len(table) == 24
    assert table.loc[0, "coverage"] == pytest.approx(1.0)
    assert table.loc[1, "coverage"] == pytest.approx(0.25)
    assert table.loc[5, "coverage"] == pytest.approx(0.0)


def test_duty_cycle_reflects_how_much_was_missed(tmp_path):
    # Two hours of wall clock, one hour of snapshots.
    times = [START + timedelta(seconds=15 * i) for i in range(EXPECTED_PER_HOUR)]
    times.append(START + timedelta(hours=2))
    write_collected(tmp_path, times)

    summary = summarise(tmp_path)
    assert summary["span_hours"] == pytest.approx(2.0, abs=0.01)
    assert summary["duty_cycle"] == pytest.approx(0.5, abs=0.01)
    assert summary["longest_gap_minutes"] > 30


def test_observers_do_not_count_as_controllers(tmp_path):
    stamp = START.isoformat().replace("+00:00", "Z")
    write_collected(
        tmp_path,
        [START],
        controllers=[
            {"snapshot_ts": stamp, "callsign": "BTP_TWR", "is_observer": False},
            {"snapshot_ts": stamp, "callsign": "PIT_APP", "is_observer": False},
            {"snapshot_ts": stamp, "callsign": "N238CT", "is_observer": True},
        ],
    )

    presence = controller_presence(tmp_path)
    assert presence.iloc[0]["controllers"] == 2
    assert presence.iloc[0]["towers"] == 1


def test_missing_is_observer_column_is_backfilled_from_frequency(tmp_path):
    # Files written before the collector had the flag still carry the frequency, so an
    # observer must not be silently counted as a working controller.
    stamp = START.isoformat().replace("+00:00", "Z")
    cpart = tmp_path / "controllers" / "date=2026-09-25"
    cpart.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"snapshot_ts": stamp, "callsign": "BTP_TWR", "frequency": "128.325"},
            {"snapshot_ts": stamp, "callsign": "N238CT", "frequency": "199.998"},
        ]
    ).to_parquet(cpart / "controllers-old.parquet", index=False)
    write_collected(tmp_path, [START])

    presence = controller_presence(tmp_path)
    assert presence.iloc[0]["controllers"] == 1
    assert presence.iloc[0]["towers"] == 1


def test_schema_drift_across_files_is_tolerated(tmp_path):
    stamp = START.isoformat().replace("+00:00", "Z")
    later = (START + timedelta(seconds=15)).isoformat().replace("+00:00", "Z")
    cpart = tmp_path / "controllers" / "date=2026-09-25"
    cpart.mkdir(parents=True, exist_ok=True)
    # Old file without the flag, new file with it.
    pd.DataFrame(
        [{"snapshot_ts": stamp, "callsign": "BTP_TWR", "frequency": "128.325"}]
    ).to_parquet(cpart / "controllers-1.parquet", index=False)
    pd.DataFrame(
        [
            {
                "snapshot_ts": later,
                "callsign": "PIT_APP",
                "frequency": "124.150",
                "is_observer": False,
            }
        ]
    ).to_parquet(cpart / "controllers-2.parquet", index=False)
    write_collected(tmp_path, [START])

    presence = controller_presence(tmp_path)
    assert len(presence) == 2
    assert presence["controllers"].tolist() == [1, 1]
