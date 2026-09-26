import numpy as np
import pytest

from pcas.eval.metrics import (
    MetricAccumulator,
    displacement,
    min_ade,
    min_fde,
    miss,
    per_horizon,
)


@pytest.fixture
def truth():
    # One agent, three waypoints along x.
    return np.array([[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]]])


def test_perfect_prediction_scores_zero(truth):
    assert min_ade(truth, truth) == pytest.approx(0.0)
    assert min_fde(truth, truth) == pytest.approx(0.0)


def test_known_errors(truth):
    pred = truth + np.array([0.0, 30.0, 40.0])  # 50 m off in 3D, 30 m horizontally
    assert min_ade(pred, truth) == pytest.approx(50.0)
    assert min_fde(pred, truth) == pytest.approx(50.0)
    assert min_ade(pred, truth, axes="horizontal") == pytest.approx(30.0)
    assert min_fde(pred, truth, axes="vertical") == pytest.approx(40.0)


def test_error_grows_with_horizon(truth):
    pred = truth.copy()
    pred[0, :, 1] = [0.0, 10.0, 20.0]
    table = per_horizon(pred, truth, [10, 20, 30])
    assert table["mean_m"].tolist() == pytest.approx([0.0, 10.0, 20.0])
    assert table["horizon_s"].tolist() == [10, 20, 30]


def test_best_of_k_uses_the_closest_hypothesis(truth):
    good = truth
    bad = truth + 1000.0
    pred = np.stack([bad[0], good[0]], axis=0)[None]  # (1, 2, T, 3)

    assert pred.shape == (1, 2, 3, 3)
    assert min_ade(pred, truth) == pytest.approx(0.0)
    assert min_fde(pred, truth) == pytest.approx(0.0)


def test_single_hypothesis_input_is_accepted(truth):
    assert displacement(truth, truth).shape == (1, 1, 3)
    assert displacement(truth[:, None], truth).shape == (1, 1, 3)


def test_minade_and_minfde_may_pick_different_hypotheses():
    truth = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    # First hypothesis is close throughout but drifts at the end; second is the reverse.
    pred = np.array(
        [
            [
                [[10.0, 0.0, 0.0], [90.0, 0.0, 0.0]],
                [[80.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
            ]
        ]
    )
    assert min_ade(pred, truth) == pytest.approx(50.0)  # first: (10+90)/2
    assert min_fde(pred, truth) == pytest.approx(20.0)  # second: better final point


def test_miss_uses_horizontal_distance_only(truth):
    # 200 m of vertical error is not a horizontal miss.
    assert not miss(truth + np.array([0.0, 0.0, 200.0]), truth, threshold_m=152.4)[0]
    assert miss(truth + np.array([200.0, 0.0, 0.0]), truth, threshold_m=152.4)[0]


def test_shape_mismatch_is_rejected(truth):
    with pytest.raises(ValueError, match="shape mismatch"):
        min_ade(truth[:, :2], truth)
    with pytest.raises(ValueError, match="expected"):
        displacement(np.zeros(3), truth)


def test_accumulator_pools_agents_not_windows(truth):
    acc = MetricAccumulator(horizons_s=[10, 20, 30])

    # One window with a single agent 100 m off, then one with three agents that are exact.
    acc.update(truth + np.array([100.0, 0.0, 0.0]), truth)
    exact = np.repeat(truth, 3, axis=0)
    acc.update(exact, exact)

    summary = acc.summary()
    assert summary["windows"] == 2
    assert summary["agents"] == 4
    # Pooled over 4 agents: 100/4 = 25. Averaging window means would give 50.
    assert summary["minADE_m"] == pytest.approx(25.0)


def test_accumulator_reports_miss_rate_and_percentiles(truth):
    acc = MetricAccumulator(horizons_s=[10, 20, 30], miss_threshold_m=152.4)
    acc.update(truth, truth)
    acc.update(truth + np.array([500.0, 0.0, 0.0]), truth)

    summary = acc.summary()
    assert summary["miss_rate_152m"] == pytest.approx(0.5)
    assert summary["median_FDE_m"] == pytest.approx(250.0)
    assert acc.horizon_table()["horizon_s"].tolist() == [10, 20, 30]


def test_empty_accumulator_refuses_to_summarise():
    acc = MetricAccumulator(horizons_s=[10])
    assert acc.empty
    with pytest.raises(ValueError, match="no windows"):
        acc.summary()
