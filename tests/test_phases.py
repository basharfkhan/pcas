"""Flight-phase labelling, on synthetic tracks whose phase is unambiguous."""

from __future__ import annotations

import numpy as np
import pytest

from pcas.eval.phases import classify, motion_features


def track(speed=45.0, heading_deg=0.0, climb=0.0, start=(0.0, 0.0, 400.0), turn_deg_s=0.0, n=11):
    """A synthetic aircraft: constant speed, optional climb and turn."""
    x, y, z = start
    xs, ys, zs = [x], [y], [z]
    heading = np.radians(heading_deg)
    for _ in range(n - 1):
        heading += np.radians(turn_deg_s)
        x += speed * np.cos(heading)
        y += speed * np.sin(heading)
        z += climb
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=-1)[None]


def test_motion_features_recover_speed_climb_and_turn():
    f = motion_features(track(speed=50.0, climb=2.0, turn_deg_s=3.0))
    assert f["speed_ms"][0] == pytest.approx(50.0, rel=0.02)
    assert f["climb_ms"][0] == pytest.approx(2.0, rel=0.02)
    assert f["turn_deg_s"][0] == pytest.approx(3.0, abs=0.2)


def test_transit_is_the_default():
    # High and far from the field, flying straight and level.
    assert classify(track(start=(9000.0, 9000.0, 1500.0)))[0] == "transit"


def test_climb_and_descent():
    assert classify(track(start=(8000.0, 0.0, 900.0), climb=3.0))[0] == "climb"
    assert classify(track(start=(8000.0, 0.0, 900.0), climb=-3.0))[0] == "descent"


def test_pattern_and_pattern_turn():
    level_in_circuit = track(start=(2000.0, 1000.0, 400.0))
    assert classify(level_in_circuit)[0] == "pattern"

    turning_in_circuit = track(start=(2000.0, 1000.0, 400.0), turn_deg_s=3.0)
    assert classify(turning_in_circuit)[0] == "pattern turn"


def test_final_approach_beats_the_other_labels():
    # Low, close, descending and turning: still final approach.
    descending_close = track(start=(1500.0, 200.0, 150.0), climb=-2.0, turn_deg_s=3.0)
    assert classify(descending_close)[0] == "final approach"


def test_field_elevation_is_subtracted():
    # The same aircraft is "in the circuit" only once the field elevation is accounted for.
    aloft = track(start=(2000.0, 500.0, 780.0))
    assert classify(aloft, field_elev_m=0.0)[0] == "transit"
    assert classify(aloft, field_elev_m=380.0)[0] == "pattern"


def test_labels_are_per_aircraft():
    obs = np.concatenate(
        [track(start=(9000.0, 9000.0, 1500.0)), track(start=(2000.0, 1000.0, 400.0))]
    )
    assert classify(obs).tolist() == ["transit", "pattern"]
