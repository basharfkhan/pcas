"""Multi-agent Transformer: scene tensors, masking, and the social ablation."""

from __future__ import annotations

import numpy as np
import pytest

from pcas.data.scenes import Window
from pcas.models.transformer import (
    TransformerConfig,
    TransformerPredictor,
    build_module,
    make_samples,
    scene_tensors,
)

torch = pytest.importorskip("torch")

HORIZONS = np.arange(10.0, 121.0, 10.0)


def straight(speed=50.0, offset=(0.0, 0.0), obs_len=11):
    t = np.arange(obs_len, dtype=float)
    return np.stack([speed * t + offset[0], np.full_like(t, offset[1]), np.full_like(t, 300.0)], -1)


def make_window(n_agents=3, obs_len=11, n_waypoints=12):
    obs = np.stack([straight(40.0 + 5 * i, (0.0, 400.0 * i), obs_len) for i in range(n_agents)])
    h = np.arange(1, n_waypoints + 1, dtype=float) * 10.0
    future = np.stack(
        [
            np.stack(
                [
                    obs[i, -1, 0] + (40.0 + 5 * i) * h,
                    np.full_like(h, 400.0 * i),
                    np.full_like(h, 300.0),
                ],
                -1,
            )
            for i in range(n_agents)
        ]
    )
    return Window(
        scene_id="2022-03-15_1200",
        date="2022-03-15",
        start_frame=0,
        agent_ids=tuple(f"A{i}" for i in range(n_agents)),
        obs=obs,
        future=future,
        wind=np.array([2.0, -1.0]),
        future_dense=None,
    )


# --- scene tensors ------------------------------------------------------------


def test_scene_tensors_shapes_and_mask():
    config = TransformerConfig(max_neighbours=4)
    own, others, mask, _ = scene_tensors(make_window(n_agents=3).obs, config, np.zeros(2))

    assert own.shape == (3, 11, config.n_features)
    assert others.shape == (3, 4, 11, config.n_features)
    # Three aircraft, so each has two neighbours and two empty slots.
    assert mask.sum(axis=1).tolist() == [2, 2, 2]
    assert not mask[:, 2:].any()


def test_single_aircraft_has_no_neighbours():
    config = TransformerConfig()
    own, others, mask, _ = scene_tensors(make_window(n_agents=1).obs, config, np.zeros(2))

    assert own.shape[0] == 1
    assert not mask.any()
    assert (others == 0).all()


def test_neighbours_are_nearest_first_and_capped():
    # Five aircraft in a line; with two slots, the target keeps its two closest.
    config = TransformerConfig(max_neighbours=2)
    obs = np.stack([straight(40.0, (0.0, 300.0 * i)) for i in range(5)])
    _, others, mask, _ = scene_tensors(obs, config, np.zeros(2))

    assert mask.sum(axis=1).tolist() == [2, 2, 2, 2, 2]
    # For the first aircraft, the nearest two sit 300 m and 600 m away laterally, so the
    # nearer one appears in the first slot (positions are scaled by position_scale).
    lateral = np.abs(others[0, :, -1, 1]) * config.position_scale
    assert lateral[0] == pytest.approx(300.0, abs=1.0)
    assert lateral[1] == pytest.approx(600.0, abs=1.0)


def test_self_track_is_in_the_targets_own_frame():
    config = TransformerConfig()
    own, _, _, _ = scene_tensors(make_window(n_agents=2).obs, config, np.zeros(2))
    # Each aircraft ends at its own origin, whatever the airport coordinates were.
    assert own[:, -1, 0:3] == pytest.approx(np.zeros((2, 3)), abs=1e-6)


def test_samples_are_flattened_per_aircraft():
    config = TransformerConfig(max_neighbours=3)
    own, others, mask, targets = make_samples([make_window(3), make_window(2)], config)

    assert own.shape == (5, 11, config.n_features)
    assert others.shape == (5, 3, 11, config.n_features)
    assert mask.shape == (5, 3)
    assert targets.shape == (5, 12, 3)
    # Targets are scaled offsets in each aircraft's own frame.
    assert targets[0, -1, 0] == pytest.approx(40.0 * 120.0 / config.target_scale, rel=0.01)


# --- module -------------------------------------------------------------------


def test_forward_returns_hypotheses_and_logits():
    config = TransformerConfig(
        d_model=32, n_heads=2, n_temporal_layers=1, max_neighbours=3, n_modes=6
    )
    module = build_module(config)
    own, others, mask, _ = scene_tensors(make_window(3).obs, config, np.zeros(2))

    trajectories, logits = module(
        torch.from_numpy(own), torch.from_numpy(others), torch.from_numpy(mask)
    )
    assert trajectories.shape == (3, 6, 12, 3)
    assert logits.shape == (3, 6)
    assert torch.isfinite(trajectories).all()
    assert torch.isfinite(logits).all()


def test_single_mode_keeps_the_hypothesis_dimension():
    config = TransformerConfig(d_model=32, n_heads=2, n_temporal_layers=1, n_modes=1)
    module = build_module(config)
    own, others, mask, _ = scene_tensors(make_window(2).obs, config, np.zeros(2))

    trajectories, logits = module(
        torch.from_numpy(own), torch.from_numpy(others), torch.from_numpy(mask)
    )
    assert trajectories.shape == (2, 1, 12, 3)
    # One hypothesis means its probability is exactly 1.
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()
    assert probabilities.squeeze(-1) == pytest.approx(np.ones(2), abs=1e-6)


def test_aircraft_with_no_neighbours_does_not_produce_nans():
    # An all-masked attention row makes softmax undefined unless it is handled; a single
    # aircraft in a window is the common case, so this must not return NaN.
    config = TransformerConfig(d_model=32, n_heads=2, n_temporal_layers=1)
    module = build_module(config)
    own, others, mask, _ = scene_tensors(make_window(1).obs, config, np.zeros(2))

    trajectories, logits = module(
        torch.from_numpy(own), torch.from_numpy(others), torch.from_numpy(mask)
    )
    assert torch.isfinite(trajectories).all()
    assert torch.isfinite(logits).all()


def test_social_ablation_ignores_neighbours():
    # With use_social=False the neighbours must make no difference at all: that is what
    # makes the ablation an attribution rather than a guess.
    config = TransformerConfig(d_model=32, n_heads=2, n_temporal_layers=1, use_social=False)
    module = build_module(config).eval()  # dropout off: this must be a clean comparison
    own, others, mask, _ = scene_tensors(make_window(3).obs, config, np.zeros(2))

    with torch.no_grad():
        baseline, _ = module(
            torch.from_numpy(own), torch.from_numpy(others), torch.from_numpy(mask)
        )
        scrambled, _ = module(
            torch.from_numpy(own),
            torch.from_numpy(others) + 5.0,
            torch.from_numpy(mask),
        )
    assert torch.allclose(baseline, scrambled)


def test_social_model_does_use_neighbours():
    config = TransformerConfig(d_model=32, n_heads=2, n_temporal_layers=1, use_social=True)
    module = build_module(config).eval()
    own, others, mask, _ = scene_tensors(make_window(3).obs, config, np.zeros(2))

    with torch.no_grad():
        baseline, _ = module(
            torch.from_numpy(own), torch.from_numpy(others), torch.from_numpy(mask)
        )
        scrambled, _ = module(
            torch.from_numpy(own),
            torch.from_numpy(others) + 5.0,
            torch.from_numpy(mask),
        )
    assert not torch.allclose(baseline, scrambled)


# --- predictor ----------------------------------------------------------------


def test_predictor_matches_the_common_interface():
    config = TransformerConfig(d_model=32, n_heads=2, n_temporal_layers=1)
    predictor = TransformerPredictor(module=build_module(config), config=config)
    window = make_window(3)

    pred = predictor.with_wind(window.wind).predict(window.obs, HORIZONS)
    assert pred.shape == (3, 1, 12, 3)
    assert np.isfinite(pred).all()


def test_multimodal_predictor_returns_hypotheses_and_probabilities():
    config = TransformerConfig(d_model=32, n_heads=2, n_temporal_layers=1, n_modes=6)
    predictor = TransformerPredictor(module=build_module(config), config=config)
    window = make_window(3)

    pred = predictor.with_wind(window.wind).predict(window.obs, HORIZONS)
    assert pred.shape == (3, 6, 12, 3)

    probabilities = predictor.last_probabilities
    assert probabilities.shape == (3, 6)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(3), abs=1e-5)
    assert (probabilities >= 0).all()


def test_multimodal_predictions_interpolate_per_hypothesis():
    config = TransformerConfig(d_model=32, n_heads=2, n_temporal_layers=1, n_modes=4)
    predictor = TransformerPredictor(module=build_module(config), config=config)
    window = make_window(2)

    dense = predictor.with_wind(window.wind).predict(window.obs, np.arange(1.0, 121.0))
    native = predictor.with_wind(window.wind).predict(window.obs, HORIZONS)

    assert dense.shape == (2, 4, 120, 3)
    # Every hypothesis must line up with its own waypoints, not with another's.
    for k in range(4):
        assert dense[0, k, 9] == pytest.approx(native[0, k, 0], abs=1e-4)


def test_predictions_are_interpolated_onto_dense_horizons():
    config = TransformerConfig(d_model=32, n_heads=2, n_temporal_layers=1)
    predictor = TransformerPredictor(module=build_module(config), config=config)
    window = make_window(2)

    dense = predictor.with_wind(window.wind).predict(window.obs, np.arange(1.0, 121.0))
    native = predictor.with_wind(window.wind).predict(window.obs, HORIZONS)

    assert dense.shape == (2, 1, 120, 3)
    assert dense[0, 0, 9] == pytest.approx(native[0, 0, 0], abs=1e-4)


def test_predictions_return_to_airport_coordinates():
    # An untrained model predicts small offsets, so predictions should land near each
    # aircraft's last observed position rather than near the origin of the local frame.
    config = TransformerConfig(d_model=32, n_heads=2, n_temporal_layers=1)
    predictor = TransformerPredictor(module=build_module(config), config=config)
    window = make_window(2)

    pred = predictor.with_wind(window.wind).predict(window.obs, np.array([10.0]))
    for i in range(2):
        assert np.linalg.norm(pred[i, 0, 0] - window.obs[i, -1]) < 3000.0
