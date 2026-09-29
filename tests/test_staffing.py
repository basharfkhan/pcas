"""Tests for the controlled-vs-uncontrolled analysis.

The important ones are at the bottom: a synthetic panel where staffing has no effect but is
correlated with traffic must come out as no effect, and a panel where it does have an effect
must come out with the effect intact. Those two together are what distinguishes this from a
raw comparison that would report the confound as a finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pcas.analysis.staffing import (
    DEFAULT_METRIC,
    LOOSE,
    NEAR,
    NM,
    NMAC,
    PROXIMITY,
    WIDE,
    build_panel,
    crude_rates,
    estimate_field_positions,
    first_stationary_samples,
    near_field_traffic,
    parse_controller_positions,
    placebo_ratios,
    staffing_by_snapshot,
    standardised_comparison,
    stratified_rates,
)

KBTP = (40.7769, -79.9498)


def _pilot_rows(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "snapshot_ts": "2026-09-27T12:00:00Z",
        "cid": 1,
        "callsign": "N123",
        "latitude": KBTP[0],
        "longitude": KBTP[1],
        "altitude": 1200,
        "groundspeed": 0,
        "logon_time": "2026-09-27T11:00:00Z",
        "departure": "KBTP",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_first_stationary_sample_is_the_departure_field_not_the_arrival() -> None:
    """A flight is parked twice: at its origin, and again after landing somewhere else.

    Only the first belongs to `departure`. Pooling both would place KBTP halfway to KAGC,
    which is the kind of error that produces a plausible-looking number and no warning.
    """
    pilots = _pilot_rows(
        [
            {"snapshot_ts": "2026-09-27T12:00:00Z", "altitude": 1200, "latitude": 40.7769},
            {"snapshot_ts": "2026-09-27T12:30:00Z", "altitude": 1200, "latitude": 40.3547},
        ]
    )
    parked = first_stationary_samples(pilots)
    assert len(parked) == 1
    assert parked["latitude"].iloc[0] == 40.7769


def test_moving_aircraft_are_not_parked() -> None:
    pilots = _pilot_rows([{"groundspeed": 90}])
    assert first_stationary_samples(pilots).empty


def test_field_position_is_the_median_of_parked_aircraft() -> None:
    rows = [
        {"cid": i, "logon_time": f"2026-09-27T1{i}:00:00Z", "latitude": 40.7769 + i * 1e-4}
        for i in range(25)
    ]
    fields = estimate_field_positions(first_stationary_samples(_pilot_rows(rows)))
    assert list(fields["icao"]) == ["KBTP"]
    assert abs(fields["lat"].iloc[0] - (40.7769 + 12 * 1e-4)) < 1e-6
    assert fields["flights"].iloc[0] == 25


def test_scattered_identifier_is_rejected_rather_than_averaged() -> None:
    """Half the samples an hour's flying away: that is not one airport, so drop it."""
    rows = [
        {
            "cid": i,
            "logon_time": f"2026-09-27T{i:02d}:00:00Z",
            "latitude": 40.7769 + (0.0 if i % 2 else 1.0),
        }
        for i in range(30)
    ]
    assert estimate_field_positions(first_stationary_samples(_pilot_rows(rows))).empty


def test_too_few_flights_is_not_a_field() -> None:
    rows = [{"cid": i, "logon_time": f"2026-09-27T{i:02d}:00:00Z"} for i in range(5)]
    assert estimate_field_positions(first_stationary_samples(_pilot_rows(rows))).empty


def test_controller_callsigns_yield_both_ident_spellings() -> None:
    controllers = pd.DataFrame(
        {"snapshot_ts": ["t", "t", "t"], "callsign": ["BTP_TWR", "SFO_N_TWR", "PIT_APP"]}
    )
    positions = parse_controller_positions(controllers)
    pairs = set(zip(positions["ident"], positions["position"], strict=True))
    assert ("BTP", "TWR") in pairs and ("KBTP", "TWR") in pairs
    assert ("SFO", "TWR") in pairs and ("KSFO", "TWR") in pairs
    assert ("PIT", "APP") in pairs and ("KPIT", "APP") in pairs


def test_ground_control_does_not_staff_the_pattern() -> None:
    """Delivery and ground do not separate airborne traffic, so neither arm turns on."""
    controllers = pd.DataFrame({"snapshot_ts": ["t", "t"], "callsign": ["KBTP_GND", "KBTP_DEL"]})
    staffing = staffing_by_snapshot(parse_controller_positions(controllers), pd.Series(["KBTP"]))
    assert not staffing["tower"].any()
    assert not staffing["approach"].any()


def test_unknown_idents_do_not_staff_anything() -> None:
    controllers = pd.DataFrame({"snapshot_ts": ["t"], "callsign": ["EGLL_TWR"]})
    staffing = staffing_by_snapshot(parse_controller_positions(controllers), pd.Series(["KBTP"]))
    assert staffing.empty


def test_traffic_outside_the_volume_is_excluded() -> None:
    fields = pd.DataFrame([{"icao": "KBTP", "lat": KBTP[0], "lon": KBTP[1], "elevation_ft": 380.0}])
    pilots = _pilot_rows(
        [
            {"cid": 1, "altitude": 1200, "groundspeed": 90},  # in the pattern
            {"cid": 2, "altitude": 1200, "groundspeed": 90, "latitude": KBTP[0] + 1.0},
            {"cid": 3, "altitude": 9000, "groundspeed": 90},  # far above the pattern
            {"cid": 4, "altitude": 420, "groundspeed": 12},  # taxiing, still on the field
        ]
    )
    traffic = near_field_traffic(pilots, fields)
    assert list(traffic["cid"]) == [1]


def test_close_pair_is_counted_and_a_separated_pair_is_not() -> None:
    fields = pd.DataFrame([{"icao": "KBTP", "lat": KBTP[0], "lon": KBTP[1], "elevation_ft": 380.0}])
    # Two aircraft 0.1 nm apart at the same altitude, and a third 8 nm away: inside the
    # volume, outside every separation gate.
    offset = np.degrees(0.1 * NM / 6_371_000.0)
    pilots = _pilot_rows(
        [
            {"cid": 1, "groundspeed": 90},
            {"cid": 2, "groundspeed": 90, "latitude": KBTP[0] + offset},
            {"cid": 3, "groundspeed": 90, "latitude": KBTP[0] + offset * 80},
        ]
    )
    traffic = near_field_traffic(pilots, fields)
    panel = build_panel(traffic, pd.DataFrame(columns=["icao", "snapshot_ts", "tower", "approach"]))
    assert panel["aircraft"].iloc[0] == 3
    assert panel["pairs"].iloc[0] == 3
    # 185 m apart co-altitude: inside every gate except NMAC, whose horizontal bound is
    # 500 ft. Exactly one of the three pairs qualifies in each case.
    assert panel[f"close_{NMAC.name}"].iloc[0] == 0
    for criterion in (PROXIMITY, NEAR, LOOSE, WIDE):
        assert panel[f"close_{criterion.name}"].iloc[0] == 1, criterion.name
    assert panel["min_sep_m"].iloc[0] < 0.2 * NM


def test_aircraft_parked_at_adjacent_gates_are_not_an_encounter() -> None:
    """The defect that made the first run of this analysis meaningless.

    Two airliners on a ramp sit a few hundred metres apart at the same altitude, which
    satisfies a 0.5 nm / 500 ft proximity criterion perfectly. Counting them turned every
    apron into a permanent close encounter and put the close-pair rate at 399 per 1,000.
    """
    fields = pd.DataFrame([{"icao": "KBTP", "lat": KBTP[0], "lon": KBTP[1], "elevation_ft": 380.0}])
    offset = np.degrees(300.0 / 6_371_000.0)
    parked = _pilot_rows(
        [
            {"cid": 1, "altitude": 380, "groundspeed": 0},
            {"cid": 2, "altitude": 380, "groundspeed": 0, "latitude": KBTP[0] + offset},
        ]
    )
    assert near_field_traffic(parked, fields).empty

    # A departure rolling fast but not yet climbing is also still on the ground, and the
    # speed gate alone would let it through.
    rolling = _pilot_rows(
        [
            {"cid": 1, "altitude": 380, "groundspeed": 120},
            {"cid": 2, "altitude": 380, "groundspeed": 130, "latitude": KBTP[0] + offset},
        ]
    )
    assert near_field_traffic(rolling, fields).empty


def test_a_snapshot_without_a_controller_row_is_unstaffed_not_missing() -> None:
    fields = pd.DataFrame([{"icao": "KBTP", "lat": KBTP[0], "lon": KBTP[1], "elevation_ft": 380.0}])
    traffic = near_field_traffic(_pilot_rows([{"cid": 1, "groundspeed": 90}]), fields)
    panel = build_panel(traffic, pd.DataFrame(columns=["icao", "snapshot_ts", "tower", "approach"]))
    assert panel["tower"].tolist() == [False]
    assert panel["hour"].tolist() == [12]


def _panel(records: list[tuple[int, bool, int, int]]) -> pd.DataFrame:
    """(aircraft, tower, pairs, close_pairs) rows spread across the clock.

    Hours advance within each (aircraft, tower) group rather than over the whole list, so
    the two arms cover the same hours instead of one arm landing in the morning and the
    other in the evening. Otherwise no stratum would ever hold both arms and the pooled
    comparison would have nothing to work with, which is a property of the fixture rather
    than of the code under test.
    """
    clock: dict[tuple[int, bool], int] = {}
    rows = []
    for aircraft, tower, pairs, close in records:
        key = (aircraft, tower)
        hour = clock.get(key, 0) % 24
        clock[key] = hour + 1
        rows.append(
            {
                "icao": "KBTP",
                "snapshot_ts": f"2026-09-27T{hour:02d}:00:00Z",
                "aircraft": aircraft,
                "pairs": pairs,
                DEFAULT_METRIC: close,
                "min_sep_m": np.nan,
                "tower": tower,
                "hour": hour,
            }
        )
    return pd.DataFrame(rows)


def test_stratification_removes_a_confound_the_crude_rate_reports_as_an_effect() -> None:
    """Staffing does nothing here, but controllers only show up when it is busy.

    Within every traffic level the rate is identical, at 20 close pairs per 1,000. Busy
    snapshots simply have a higher rate than quiet ones, and staffed snapshots are the busy
    ones. The crude comparison therefore claims staffing raises the rate; the standardised
    one has to find no difference.
    """
    quiet = [(2, False, 1000, 10) for _ in range(12)]
    quiet_staffed = [(2, True, 1000, 10) for _ in range(12)]
    busy = [(8, False, 1000, 50) for _ in range(12)]
    busy_staffed = [(8, True, 1000, 50) for _ in range(12)]

    # The confound: quiet snapshots are mostly unstaffed, busy ones mostly staffed.
    panel = _panel(quiet * 4 + quiet_staffed + busy + busy_staffed * 4)

    crude = crude_rates(panel).set_index("staffed")["rate_per_1k_pairs"]
    assert crude[True] > 1.5 * crude[False]

    pooled = standardised_comparison(stratified_rates(panel))
    assert abs(pooled["ratio_on_over_off"] - 1.0) < 0.01


def test_a_real_effect_survives_stratification() -> None:
    """Same confounded design, but staffing genuinely halves the rate within each stratum."""
    quiet = [(2, False, 1000, 20) for _ in range(12)]
    quiet_staffed = [(2, True, 1000, 10) for _ in range(12)]
    busy = [(8, False, 1000, 60) for _ in range(12)]
    busy_staffed = [(8, True, 1000, 30) for _ in range(12)]

    panel = _panel(quiet * 4 + quiet_staffed + busy + busy_staffed * 4)
    pooled = standardised_comparison(stratified_rates(panel))
    assert abs(pooled["ratio_on_over_off"] - 0.5) < 0.01
    assert pooled["strata_with_lower_rate_staffed"] == pooled["strata"]


def test_strata_present_in_only_one_arm_are_dropped() -> None:
    """An unstaffed-only traffic level cannot contribute: there is nothing to compare it to."""
    panel = _panel(
        [(2, False, 1000, 20) for _ in range(12)]
        + [(2, True, 1000, 10) for _ in range(12)]
        + [(20, False, 1000, 900) for _ in range(12)]  # never staffed
    )
    strata = stratified_rates(panel)
    pooled = standardised_comparison(strata)
    assert pooled["pairs_off"] < strata["pairs_off"].sum()
    assert abs(pooled["ratio_on_over_off"] - 0.5) < 0.01


def test_placebo_finds_nothing_when_staffing_is_noise() -> None:
    """Shuffled labels must pool to about 1, or the machinery is inventing effects."""
    panel = _panel(
        [(2, False, 1000, 20) for _ in range(12)]
        + [(2, True, 1000, 10) for _ in range(12)]
        + [(8, False, 1000, 60) for _ in range(12)]
        + [(8, True, 1000, 30) for _ in range(12)]
    )
    real = standardised_comparison(stratified_rates(panel, by_field=True))
    assert abs(real["ratio_on_over_off"] - 0.5) < 0.01

    ratios = [r for r in placebo_ratios(panel, seeds=5) if r == r]
    assert ratios
    assert all(0.7 < r < 1.4 for r in ratios), ratios


def test_by_field_keeps_the_comparison_inside_one_airport() -> None:
    """Two airports, opposite staffing, no within-field contrast: nothing to report.

    Pooled across fields this looks like a large effect, because the always-staffed field
    happens to be the quiet one. Within field there is no comparison to make at all, and
    saying so is the correct answer.
    """
    quiet_staffed = _panel([(2, True, 1000, 5) for _ in range(12)]).assign(icao="KQUI")
    busy_unstaffed = _panel([(2, False, 1000, 50) for _ in range(12)]).assign(icao="KBUS")
    panel = pd.concat([quiet_staffed, busy_unstaffed], ignore_index=True)

    pooled_across = standardised_comparison(stratified_rates(panel))
    assert pooled_across["ratio_on_over_off"] == 0.1

    within = standardised_comparison(stratified_rates(panel, by_field=True))
    assert within["strata"] == 0.0
