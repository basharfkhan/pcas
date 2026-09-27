"""Probability calibration: the isotonic fit, and what it may and may not change."""

from __future__ import annotations

import numpy as np
import pytest

from pcas.eval.calibration import (
    ProbabilityCalibrator,
    expected_calibration_error,
    fit_isotonic,
    reliability,
)

RNG = np.random.default_rng(0)


def overconfident_sample(n=4000, factor=3.0):
    """Stated probabilities that are `factor` times too high."""
    stated = RNG.uniform(0.05, 0.95, size=n)
    true = np.clip(stated / factor, 0, 1)
    outcome = (RNG.uniform(size=n) < true).astype(float)
    return stated, outcome


def test_calibration_fixes_overconfidence():
    stated, outcome = overconfident_sample()
    before = expected_calibration_error(stated, outcome)

    calibrator = fit_isotonic(stated, outcome)
    after = expected_calibration_error(calibrator.predict(stated), outcome)

    # Stated three times too high, so the gap before is large and after is small.
    assert before > 0.2
    assert after < 0.05


def test_calibration_preserves_ordering():
    # Monotonicity is the whole assumption: an isotonic map may not reorder anything, which is
    # why detection versus false alarm behaviour cannot change.
    stated, outcome = overconfident_sample()
    calibrated = fit_isotonic(stated, outcome).predict(stated)

    order_before = np.argsort(stated, kind="stable")
    assert np.all(np.diff(calibrated[order_before]) >= -1e-12)


def test_calibration_is_monotone_on_new_inputs():
    stated, outcome = overconfident_sample()
    calibrator = fit_isotonic(stated, outcome)
    probe = np.linspace(0.0, 1.0, 50)
    assert np.all(np.diff(calibrator.predict(probe)) >= -1e-12)


def test_predictions_stay_in_range():
    stated, outcome = overconfident_sample()
    calibrated = fit_isotonic(stated, outcome).predict(np.array([-0.5, 0.0, 0.5, 1.0, 2.0]))
    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0


def test_underconfident_probabilities_are_raised():
    stated = RNG.uniform(0.02, 0.3, size=3000)
    outcome = (RNG.uniform(size=3000) < np.clip(stated * 3, 0, 1)).astype(float)
    calibrator = fit_isotonic(stated, outcome)
    # A model that says 10% when conflicts follow 30% of the time should be corrected upwards.
    assert calibrator.predict(np.array([0.1]))[0] > 0.15


def test_perfectly_calibrated_input_is_left_alone():
    stated = RNG.uniform(0.05, 0.95, size=6000)
    outcome = (RNG.uniform(size=6000) < stated).astype(float)
    calibrator = fit_isotonic(stated, outcome)

    probe = np.array([0.2, 0.5, 0.8])
    assert calibrator.predict(probe) == pytest.approx(probe, abs=0.08)


def test_empty_input_gives_an_identity_map():
    calibrator = fit_isotonic(np.array([]), np.array([]))
    probe = np.array([0.3, 0.7])
    assert calibrator.predict(probe) == pytest.approx(probe)


def test_single_class_does_not_explode():
    # Nothing ever conflicted, so everything should map towards zero.
    stated = RNG.uniform(0.1, 0.9, size=500)
    calibrator = fit_isotonic(stated, np.zeros(500))
    assert calibrator.predict(np.array([0.9]))[0] == pytest.approx(0.0, abs=1e-6)


def test_round_trip_through_json(tmp_path):
    stated, outcome = overconfident_sample(n=800)
    calibrator = fit_isotonic(stated, outcome)

    path = tmp_path / "calibration.json"
    calibrator.to_json(path)
    restored = ProbabilityCalibrator.from_json(path)

    probe = np.linspace(0.05, 0.95, 20)
    assert restored.predict(probe) == pytest.approx(calibrator.predict(probe))
    assert restored.n_samples == calibrator.n_samples


def test_reliability_table_reports_bins():
    stated = np.array([0.1, 0.15, 0.9, 0.95])
    outcome = np.array([0.0, 0.0, 1.0, 0.0])
    table = reliability(stated, outcome)

    assert table["n"].sum() == 4
    low = table.iloc[0]
    assert low["observed"] == pytest.approx(0.0)
    high = table.iloc[-1]
    assert high["observed"] == pytest.approx(0.5)


def test_expected_calibration_error_is_zero_when_perfect():
    stated = np.array([0.0, 0.0, 1.0, 1.0])
    outcome = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(stated, outcome) == pytest.approx(0.0, abs=1e-9)
