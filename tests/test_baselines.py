import numpy as np
import pytest

from pcas.models.baselines import (
    ConstantTurnRate,
    ConstantVelocity,
    KalmanConstantVelocity,
)

HORIZONS = np.array([10.0, 60.0, 120.0])
# 3 degrees per second: the standard rate a traffic pattern is flown at.
STANDARD_RATE = np.radians(3.0)


def straight_track(speed=50.0, climb=2.0, obs_len=11):
    """Aircraft flying east at `speed` m/s while climbing."""
    t = np.arange(obs_len, dtype=float)
    return np.stack([speed * t, np.zeros_like(t), 300.0 + climb * t], axis=-1)[None]


def circling_track(speed=50.0, turn_rate=STANDARD_RATE, obs_len=11):
    """Aircraft in a standard-rate turn."""
    t = np.arange(obs_len, dtype=float)
    radius = speed / turn_rate
    heading = turn_rate * t
    x = radius * np.sin(heading)
    y = radius * (1 - np.cos(heading))
    return np.stack([x, y, np.full_like(t, 300.0)], axis=-1)[None]


@pytest.mark.parametrize(
    "model", [ConstantVelocity(), ConstantVelocity(fit_len=4), KalmanConstantVelocity()]
)
def test_straight_line_is_predicted_exactly(model):
    obs = straight_track()
    pred = model.predict(obs, HORIZONS)

    assert pred.shape == (1, 1, 3, 3)
    last = obs[0, -1]
    for i, h in enumerate(HORIZONS):
        expected = last + np.array([50.0 * h, 0.0, 2.0 * h])
        assert pred[0, 0, i] == pytest.approx(expected, abs=1.0)


def test_constant_velocity_cannot_follow_a_turn():
    obs = circling_track()
    pred = ConstantVelocity().predict(obs, HORIZONS)
    # After 120 s of standard-rate turning the aircraft has come right around; a straight
    # line is kilometres out.
    assert np.linalg.norm(pred[0, 0, -1, :2]) > 3000.0


def test_constant_turn_rate_follows_a_standard_rate_turn():
    speed, turn_rate = 50.0, STANDARD_RATE
    obs = circling_track(speed=speed, turn_rate=turn_rate, obs_len=11)
    pred = ConstantTurnRate().predict(obs, HORIZONS)

    radius = speed / turn_rate
    for i, h in enumerate(HORIZONS):
        heading = turn_rate * (10.0 + h)  # last observed sample is at t = 10 s
        expected = np.array([radius * np.sin(heading), radius * (1 - np.cos(heading)), 300.0])
        assert pred[0, 0, i, :2] == pytest.approx(expected[:2], abs=60.0)


def test_constant_turn_rate_degenerates_to_a_straight_line():
    obs = straight_track()
    pred = ConstantTurnRate().predict(obs, HORIZONS)
    last = obs[0, -1]
    for i, h in enumerate(HORIZONS):
        assert pred[0, 0, i, :2] == pytest.approx(last[:2] + np.array([50.0 * h, 0.0]), abs=1.0)


def test_turn_estimate_survives_a_heading_wrap():
    # Flying west, so heading sits on the +/-pi branch cut. Without unwrapping, the
    # estimated turn rate would be enormous and the prediction would fly off sideways.
    t = np.arange(11, dtype=float)
    obs = np.stack([-50.0 * t, 0.1 * np.sin(t / 5), np.full_like(t, 300.0)], axis=-1)[None]
    pred = ConstantTurnRate().predict(obs, HORIZONS)

    assert pred[0, 0, -1, 0] == pytest.approx(obs[0, -1, 0] - 50.0 * 120.0, abs=200.0)
    assert abs(pred[0, 0, -1, 1]) < 200.0


def test_multiple_agents_are_predicted_independently():
    obs = np.concatenate([straight_track(speed=50.0), straight_track(speed=25.0)], axis=0)
    pred = ConstantVelocity().predict(obs, HORIZONS)

    assert pred.shape == (2, 1, 3, 3)
    assert pred[0, 0, -1, 0] == pytest.approx(obs[0, -1, 0] + 50.0 * 120.0, abs=1.0)
    assert pred[1, 0, -1, 0] == pytest.approx(obs[1, -1, 0] + 25.0 * 120.0, abs=1.0)


def test_kalman_reacts_faster_to_a_recent_speed_change():
    # Slow for 8 s, then accelerate. A plain fit over the whole window is dragged down by
    # the early samples; the filter weights recent evidence more.
    slow = np.arange(8, dtype=float) * 20.0
    fast = slow[-1] + np.arange(1, 4, dtype=float) * 60.0
    x = np.concatenate([slow, fast])
    obs = np.stack([x, np.zeros_like(x), np.full_like(x, 300.0)], axis=-1)[None]

    cv = ConstantVelocity().predict(obs, np.array([60.0]))[0, 0, 0, 0]
    kf = KalmanConstantVelocity().predict(obs, np.array([60.0]))[0, 0, 0, 0]
    assert kf > cv


def test_too_few_observations_is_rejected():
    with pytest.raises(ValueError, match="at least two"):
        ConstantVelocity().predict(np.zeros((1, 1, 3)), HORIZONS)
