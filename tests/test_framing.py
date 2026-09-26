import numpy as np
import pytest

from pcas.models.framing import features, heading_from, to_agent_frame, to_world


def track(heading_deg=0.0, speed=50.0, obs_len=11, start=(1000.0, 2000.0, 300.0)):
    t = np.arange(obs_len, dtype=float)
    h = np.radians(heading_deg)
    x = start[0] + speed * np.cos(h) * t
    y = start[1] + speed * np.sin(h) * t
    z = np.full_like(t, start[2])
    return np.stack([x, y, z], -1)[None]


@pytest.mark.parametrize("heading_deg", [0.0, 45.0, 90.0, 180.0, -120.0])
def test_heading_is_recovered(heading_deg):
    got = np.degrees(heading_from(track(heading_deg))[0])
    assert np.isclose(got, heading_deg, atol=0.5) or np.isclose(got, heading_deg + 360, atol=0.5)


def test_stationary_aircraft_gets_zero_heading_not_nan():
    still = np.tile(np.array([100.0, 200.0, 300.0]), (1, 11, 1))
    assert heading_from(still)[0] == 0.0


def test_last_observed_position_becomes_the_origin():
    obs_local, _, _ = to_agent_frame(track(heading_deg=30.0))
    assert obs_local[0, -1] == pytest.approx([0.0, 0.0, 0.0])


def test_aircraft_heads_along_positive_x_in_its_own_frame():
    # Whatever the true heading, past positions lie behind the aircraft on -x with ~no y.
    for heading in (0.0, 37.0, 200.0):
        obs_local, _, _ = to_agent_frame(track(heading_deg=heading))
        assert obs_local[0, 0, 0] < -400.0
        assert abs(obs_local[0, 0, 1]) < 1.0


def test_round_trip_returns_world_coordinates():
    obs = track(heading_deg=57.0)
    target = obs[:, -3:, :] + np.array([100.0, -50.0, 20.0])
    _, target_local, frame = to_agent_frame(obs, target)
    assert to_world(target_local, frame) == pytest.approx(target, abs=1e-6)


def test_round_trip_handles_hypothesis_dimension():
    obs = track(heading_deg=10.0)
    _, _, frame = to_agent_frame(obs)
    local = np.zeros((1, 3, 4, 3))  # 3 hypotheses, 4 waypoints
    world = to_world(local, frame)
    assert world.shape == (1, 3, 4, 3)
    # Every hypothesis at the local origin maps to the last observed position.
    assert world[0, 0, 0] == pytest.approx(obs[0, -1], abs=1e-6)


def test_two_aircraft_are_framed_independently():
    obs = np.concatenate([track(heading_deg=0.0), track(heading_deg=90.0)], axis=0)
    obs_local, _, frame = to_agent_frame(obs)

    assert obs_local[:, -1] == pytest.approx(np.zeros((2, 3)))
    assert np.degrees(frame["heading"]) == pytest.approx([0.0, 90.0], abs=0.5)
    assert to_world(obs_local, frame) == pytest.approx(obs, abs=1e-6)


def test_features_add_velocity_and_wind():
    obs_local, _, _ = to_agent_frame(track(speed=50.0))
    plain = features(obs_local)
    assert plain.shape == (1, 11, 6)
    # Velocity along the aircraft's own +x is its speed; the first step is padded to zero.
    assert plain[0, 5, 3] == pytest.approx(50.0, abs=0.01)
    assert plain[0, 0, 3:6] == pytest.approx([0.0, 0.0, 0.0])

    with_wind = features(obs_local, np.array([3.0, -1.0]))
    assert with_wind.shape == (1, 11, 8)
    assert with_wind[0, :, 6] == pytest.approx(3.0)
