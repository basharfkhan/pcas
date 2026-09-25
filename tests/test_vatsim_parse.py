import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from pcas.collect.vatsim import parse_controllers, parse_pilots, snapshot_timestamp
from pcas.collect.writer import ParquetBuffer
from pcas.config import CollectorConfig

FIXTURE = Path(__file__).parent / "fixtures" / "sample_datafeed.json"


@pytest.fixture
def feed() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_pilot_rows_carry_snapshot_and_flight_plan(feed):
    rows = parse_pilots(feed)

    assert len(rows) == 3
    assert {row["snapshot_ts"] for row in rows} == {snapshot_timestamp(feed)}

    cessna = next(row for row in rows if row["callsign"] == "N172SP")
    assert cessna["aircraft"] == "C172"
    assert cessna["departure"] == cessna["arrival"] == "KBTP"
    assert cessna["altitude"] == 1800


def test_pilot_rows_drop_personal_name(feed):
    # CID is enough to stitch a track together; the free-text name is personal data.
    assert all("name" not in row for row in parse_pilots(feed))


def test_missing_flight_plan_yields_nulls_not_errors(feed):
    row = next(row for row in parse_pilots(feed) if row["callsign"] == "N999XX")
    assert row["aircraft"] is None
    assert row["flight_rules"] is None


def test_bbox_filters_by_position(feed):
    # A box around Pittsburgh-Butler should keep the Cessna and drop the airliner over
    # Atlanta, along with the pilot reporting no position at all.
    rows = parse_pilots(feed, bbox=[40.0, -81.0, 41.5, -79.0])
    assert [row["callsign"] for row in rows] == ["N172SP"]


def test_controllers_include_atis_connections(feed):
    rows = parse_controllers(feed)

    assert {row["callsign"] for row in rows} == {"BTP_TWR", "KATL_ATIS", "N238CT"}
    assert {row["kind"] for row in rows} == {"controllers", "atis"}
    assert next(r for r in rows if r["callsign"] == "BTP_TWR")["frequency"] == "128.325"


def test_observers_are_flagged_not_counted_as_controllers(feed):
    # The live feed lists observers among `controllers` on frequency 199.998. Treating
    # one as a working controller would poison the controlled-vs-uncontrolled split.
    rows = parse_controllers(feed)
    by_callsign = {row["callsign"]: row for row in rows}

    assert by_callsign["N238CT"]["is_observer"] is True
    assert by_callsign["BTP_TWR"]["is_observer"] is False
    assert by_callsign["KATL_ATIS"]["is_observer"] is False


def test_buffer_flushes_to_date_partitioned_parquet(tmp_path, feed):
    buffer = ParquetBuffer(tmp_path, "pilots")
    buffer.extend(parse_pilots(feed))
    assert len(buffer) == 3

    stamp = datetime(2026, 9, 25, 14, 5, tzinfo=UTC)
    path = buffer.flush(stamp)

    assert path is not None
    assert path.parent.name == "date=2026-09-25"
    assert len(pd.read_parquet(path)) == 3
    # Flushing empties the buffer, so the next flush is a no-op.
    assert len(buffer) == 0
    assert buffer.flush(stamp) is None


def test_poll_interval_floor_is_enforced(tmp_path):
    with pytest.raises(ValueError, match="poll_seconds"):
        CollectorConfig(data_dir=tmp_path, poll_seconds=5)
