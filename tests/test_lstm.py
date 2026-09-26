"""LSTM baseline. Skipped wholesale when torch is absent, as it is in CI."""

from __future__ import annotations

import numpy as np
import pytest

from pcas.data.scenes import Window
from pcas.models.framing import features, to_agent_frame
from pcas.models.lstm import LSTMConfig, LSTMPredictor, build_module, make_samples, normalise

torch = pytest.importorskip("torch")


def make_window(n_agents=2, obs_len=11, n_waypoints=12):
    t = np.arange(obs_len, dtype=float)
    obs, future = [], []
    for i in range(n_agents):
        speed = 40.0 + 10 * i
        obs.append(np.stack([speed * t, np.full_like(t, 500.0 * i), np.full_like(t, 300.0)], -1))
        h = np.arange(1, n_waypoints + 1, dtype=float) * 10.0
        future.append(
            np.stack(
                [
                    speed * (obs_len - 1) + speed * h,
                    np.full_like(h, 500.0 * i),
                    np.full_like(h, 300.0),
                ],
                -1,
            )
        )
    return Window(
        scene_id="2020-09-18_1200",
        date="2020-09-18",
        start_frame=0,
        agent_ids=tuple(f"A{i}" for i in range(n_agents)),
        obs=np.stack(obs),
        future=np.stack(future),
        wind=np.array([3.0, -1.0]),
        future_dense=None,
    )


def test_samples_are_flattened_per_aircraft():
    config = LSTMConfig()
    x, y = make_samples([make_window(n_agents=2), make_window(n_agents=3)], config)

    assert x.shape == (5, 11, 8)  # 5 aircraft, 11 steps, 6 motion + 2 wind
    assert y.shape == (5, 12, 3)
    assert x.dtype == np.float32


def test_targets_are_in_the_aircraft_own_frame():
    window = make_window(n_agents=1)
    _, y = make_samples([window], LSTMConfig())
    # Flying along +x at 40 m/s, so 120 s ahead is ~4800 m straight ahead and no lateral
    # offset, whatever the airport coordinates were.
    assert y[0, -1, 0] == pytest.approx(4800.0, rel=0.01)
    assert abs(y[0, -1, 1]) < 1.0


def test_no_wind_config_drops_those_features():
    x, _ = make_samples([make_window()], LSTMConfig(use_wind=False))
    assert x.shape[-1] == 6


def test_normalisation_scales_positions_and_velocities():
    config = LSTMConfig()
    obs_local, _, _ = to_agent_frame(make_window(n_agents=1).obs)
    raw = features(obs_local, np.array([3.0, -1.0]))
    scaled = normalise(raw, config)

    assert scaled[..., 0] == pytest.approx(raw[..., 0] / config.position_scale)
    assert scaled[..., 3] == pytest.approx(raw[..., 3] / config.velocity_scale)
    assert scaled[..., 6] == pytest.approx(raw[..., 6])  # wind left alone


def test_module_output_shape_and_predictor_interface():
    config = LSTMConfig()
    module = build_module(config)
    window = make_window(n_agents=2)

    predictor = LSTMPredictor(module=module, config=config)
    pred = predictor.with_wind(window.wind).predict(window.obs, np.arange(10.0, 121.0, 10.0))

    assert pred.shape == (2, 1, 12, 3)
    assert np.isfinite(pred).all()


def test_predictions_are_interpolated_onto_dense_horizons():
    config = LSTMConfig()
    predictor = LSTMPredictor(module=build_module(config), config=config)
    window = make_window(n_agents=1)

    dense = np.arange(1.0, 121.0)
    pred = predictor.with_wind(window.wind).predict(window.obs, dense)
    assert pred.shape == (1, 1, 120, 3)

    # Interpolation must agree with the native waypoints where they coincide.
    native = predictor.with_wind(window.wind).predict(window.obs, np.arange(10.0, 121.0, 10.0))
    assert pred[0, 0, 9] == pytest.approx(native[0, 0, 0], abs=1e-4)


def test_untrained_model_predicts_near_the_current_position():
    # Outputs are offsets from the last observed position, so an untrained net starts near
    # "stay put" rather than somewhere arbitrary in airport coordinates.
    config = LSTMConfig()
    predictor = LSTMPredictor(module=build_module(config), config=config)
    window = make_window(n_agents=1)
    pred = predictor.with_wind(window.wind).predict(window.obs, np.array([120.0]))

    assert np.linalg.norm(pred[0, 0, 0] - window.obs[0, -1]) < 2000.0
