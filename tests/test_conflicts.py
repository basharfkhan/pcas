import numpy as np
import pandas as pd
import pytest

from pcas.data.scenes import Window
from pcas.eval.conflicts import (
    NMAC,
    PROXIMITY,
    AlertScorer,
    ConflictCriterion,
    closure_rate_alerts,
    first_violation,
    pair_separations,
    predicted_alerts,
)

TIGHT = ConflictCriterion(horizontal_m=200.0, vertical_m=100.0, name="test")


def make_window(future_dense, obs=None, agent_ids=None):
    n = future_dense.shape[0]
    return Window(
        scene_id="2020-09-18_1200",
        date="2020-09-18",
        start_frame=0,
        agent_ids=agent_ids or tuple(f"A{k}" for k in range(n)),
        obs=obs if obs is not None else np.zeros((n, 11, 3)),
        future=future_dense[:, ::10, :],
        wind=np.zeros(2),
        future_dense=future_dense,
    )


def converging(n_steps=120, closing_speed=20.0, start_gap=1500.0, altitude_gap=0.0):
    """Two aircraft flying at each other along x, `start_gap` apart."""
    t = np.arange(1, n_steps + 1, dtype=float)
    a = np.stack(
        [-start_gap / 2 + closing_speed / 2 * t, np.zeros_like(t), np.full_like(t, 300.0)], -1
    )
    b = np.stack(
        [
            start_gap / 2 - closing_speed / 2 * t,
            np.zeros_like(t),
            np.full_like(t, 300.0 + altitude_gap),
        ],
        -1,
    )
    return np.stack([a, b])


# --- criteria and labelling ---------------------------------------------------


def test_standard_criteria_match_published_thresholds():
    assert NMAC.horizontal_m == pytest.approx(152.4)
    assert NMAC.vertical_m == pytest.approx(30.48)
    assert PROXIMITY.horizontal_m == pytest.approx(926.0)
    assert PROXIMITY.vertical_m == pytest.approx(152.4)


def test_both_thresholds_must_break_together():
    # Horizontally close but 300 m apart vertically is not a conflict.
    assert not PROXIMITY.violated(np.array(50.0), np.array(300.0))
    # Vertically close but 5 km apart horizontally is not either.
    assert not PROXIMITY.violated(np.array(5000.0), np.array(10.0))
    assert PROXIMITY.violated(np.array(50.0), np.array(10.0))


def test_pair_separations_shapes_and_values():
    traj = np.array([[[0.0, 0.0, 0.0]], [[300.0, 400.0, 50.0]]])
    horizontal, vertical, pairs = pair_separations(traj)
    assert pairs == [(0, 1)]
    assert horizontal[0, 0] == pytest.approx(500.0)
    assert vertical[0, 0] == pytest.approx(50.0)


def test_single_aircraft_has_no_pairs():
    horizontal, _, pairs = pair_separations(np.zeros((1, 5, 3)))
    assert pairs == []
    assert horizontal.shape == (0, 5)


def test_first_violation_reports_onset_and_minimum():
    traj = converging(closing_speed=20.0, start_gap=1500.0)
    times = np.arange(1, traj.shape[1] + 1, dtype=float)
    events = first_violation(traj, TIGHT, times)

    assert len(events) == 1
    row = events.iloc[0]
    # Gap closes at 20 m/s from 1500 m, so it drops under 200 m at t = 65 s.
    assert row["onset_s"] == pytest.approx(65.0, abs=1.0)
    assert row["min_horizontal_m"] < 200.0


def test_aircraft_that_stay_apart_produce_no_event():
    traj = converging(closing_speed=2.0, start_gap=5000.0)
    times = np.arange(1, traj.shape[1] + 1, dtype=float)
    assert first_violation(traj, TIGHT, times).empty


def test_vertical_separation_prevents_an_event():
    traj = converging(closing_speed=20.0, start_gap=1500.0, altitude_gap=400.0)
    times = np.arange(1, traj.shape[1] + 1, dtype=float)
    assert first_violation(traj, TIGHT, times).empty


# --- closure-rate alerting ----------------------------------------------------


def observed_converging(closing_speed=20.0, gap_at_end=1000.0, obs_len=11, altitude_gap=0.0):
    """Observation window for two aircraft closing head-on, `gap_at_end` apart at the end."""
    t = np.arange(obs_len, dtype=float)
    back = (obs_len - 1 - t) * closing_speed / 2
    a = np.stack([-gap_at_end / 2 - back, np.zeros_like(t), np.full_like(t, 300.0)], -1)
    b = np.stack(
        [gap_at_end / 2 + back, np.zeros_like(t), np.full_like(t, 300.0 + altitude_gap)], -1
    )
    return np.stack([a, b])


def test_closure_rate_alerts_on_a_head_on_pair():
    obs = observed_converging(closing_speed=20.0, gap_at_end=600.0)
    alerts = closure_rate_alerts(obs, TIGHT, tau_s=40.0)

    assert len(alerts) == 1
    # 600 m of gap at 20 m/s closes in 30 s, inside the 40 s horizon.
    assert alerts.iloc[0]["alert_time_s"] == pytest.approx(30.0, abs=1.5)


def test_closure_rate_stays_quiet_beyond_its_horizon():
    # Same geometry, but the closest approach is 100 s away.
    obs = observed_converging(closing_speed=20.0, gap_at_end=2000.0)
    assert closure_rate_alerts(obs, TIGHT, tau_s=40.0).empty
    assert len(closure_rate_alerts(obs, TIGHT, tau_s=120.0)) == 1


def test_closure_rate_ignores_diverging_traffic():
    obs = observed_converging(closing_speed=-20.0, gap_at_end=600.0)
    assert closure_rate_alerts(obs, TIGHT, tau_s=60.0).empty


def test_existing_violation_alerts_immediately():
    obs = observed_converging(closing_speed=0.0, gap_at_end=50.0)
    alerts = closure_rate_alerts(obs, TIGHT, tau_s=40.0)
    assert len(alerts) == 1
    assert alerts.iloc[0]["alert_time_s"] == 0.0


def test_closure_rate_respects_vertical_separation():
    obs = observed_converging(closing_speed=20.0, gap_at_end=600.0, altitude_gap=400.0)
    assert closure_rate_alerts(obs, TIGHT, tau_s=40.0).empty


# --- prediction-based alerting ------------------------------------------------


def test_predicted_alerts_use_the_predicted_trajectories():
    horizons = np.array([30.0, 60.0, 90.0])
    # Two aircraft predicted to pass 50 m apart at the second waypoint.
    pred = np.zeros((2, 1, 3, 3))
    pred[0, 0, :, 0] = [1000.0, 0.0, -1000.0]
    pred[1, 0, :, 0] = [-1000.0, 50.0, 1000.0]

    alerts = predicted_alerts(pred, horizons, TIGHT)
    assert len(alerts) == 1
    assert alerts.iloc[0]["alert_time_s"] == pytest.approx(60.0)


def test_hypothesis_share_becomes_alert_probability():
    horizons = np.array([30.0])
    # Four hypotheses, one of which collides.
    pred = np.zeros((2, 4, 1, 3))
    pred[0, :, 0, 0] = [0.0, 0.0, 0.0, 0.0]
    pred[1, :, 0, 0] = [5000.0, 5000.0, 5000.0, 10.0]

    assert predicted_alerts(pred, horizons, TIGHT, probability_threshold=0.5).empty
    loose = predicted_alerts(pred, horizons, TIGHT, probability_threshold=0.25)
    assert len(loose) == 1
    assert loose.iloc[0]["probability"] == pytest.approx(0.25)


# --- scoring ------------------------------------------------------------------


def test_scorer_counts_detection_false_alarms_and_lead_time():
    scorer = AlertScorer(criterion=TIGHT, window_stride_s=10.0)

    # Window 1: a real event at 65 s, correctly alerted.
    event_window = make_window(converging(closing_speed=20.0, start_gap=1500.0))
    scorer.update(event_window, pd.DataFrame([{"i": 0, "j": 1, "alert_time_s": 60.0}]), 120.0)

    # Window 2: no event, but an alert anyway.
    quiet_window = make_window(converging(closing_speed=0.0, start_gap=9000.0))
    scorer.update(quiet_window, pd.DataFrame([{"i": 0, "j": 1, "alert_time_s": 30.0}]), 120.0)

    summary = scorer.summary()
    assert summary["events"] == 1
    assert summary["detected"] == 1
    assert summary["detection_rate"] == pytest.approx(1.0)
    assert summary["false_alarms"] == 1
    assert summary["median_lead_time_s"] == pytest.approx(65.0, abs=1.0)
    # Two windows at a 10 s stride is 20 s of scene time, so one false alarm is 180 per hour.
    assert summary["false_alarms_per_hour"] == pytest.approx(180.0, rel=0.01)


def test_missed_event_lowers_detection_rate():
    scorer = AlertScorer(criterion=TIGHT)
    window = make_window(converging(closing_speed=20.0, start_gap=1500.0))
    scorer.update(window, pd.DataFrame(columns=["i", "j", "alert_time_s"]), 120.0)

    summary = scorer.summary()
    assert summary["events"] == 1
    assert summary["detected"] == 0
    assert summary["detection_rate"] == pytest.approx(0.0)


def test_events_beyond_the_horizon_are_not_counted():
    scorer = AlertScorer(criterion=TIGHT)
    window = make_window(converging(closing_speed=20.0, start_gap=1500.0))
    # The event is at 65 s, so a 30 s horizon should see no event at all.
    scorer.update(window, pd.DataFrame(columns=["i", "j", "alert_time_s"]), 30.0)
    assert scorer.summary()["events"] == 0


def test_every_pair_is_an_opportunity():
    scorer = AlertScorer(criterion=TIGHT)
    # Three aircraft far apart: 3 pairs, no events.
    far = np.zeros((3, 120, 3))
    far[1, :, 0] = 9000.0
    far[2, :, 0] = 18000.0
    scorer.update(
        far_window := make_window(far), pd.DataFrame(columns=["i", "j", "alert_time_s"]), 120.0
    )

    assert far_window.n_agents == 3
    assert scorer.summary()["pair_windows"] == 3


def test_detection_by_lead_time_buckets():
    scorer = AlertScorer(criterion=TIGHT)
    # Event at ~65 s, detected.
    scorer.update(
        make_window(converging(closing_speed=20.0, start_gap=1500.0)),
        pd.DataFrame([{"i": 0, "j": 1, "alert_time_s": 60.0}]),
        120.0,
    )
    # Event at ~15 s, missed.
    scorer.update(
        make_window(converging(closing_speed=20.0, start_gap=500.0)),
        pd.DataFrame(columns=["i", "j", "alert_time_s"]),
        120.0,
    )

    table = scorer.detection_by_lead_time()
    assert table["events"].sum() == 2
    assert table["detected"].sum() == 1


def test_window_without_dense_future_is_rejected():
    scorer = AlertScorer()
    window = make_window(converging())
    bare = Window(
        scene_id=window.scene_id,
        date=window.date,
        start_frame=0,
        agent_ids=window.agent_ids,
        obs=window.obs,
        future=window.future,
        wind=window.wind,
    )
    with pytest.raises(ValueError, match="future_dense"):
        scorer.update(bare, pd.DataFrame(columns=["i", "j", "alert_time_s"]), 120.0)


def test_joint_mode_probability_weights_hypothesis_pairs():
    horizons = np.array([30.0])
    # Aircraft 0 has two hypotheses: one collides with aircraft 1's only hypothesis.
    pred = np.zeros((2, 2, 1, 3))
    pred[0, 0, 0, 0] = 0.0  # on top of aircraft 1
    pred[0, 1, 0, 0] = 9000.0  # far away
    pred[1, :, 0, 0] = 0.0

    # Equal weights: one of two combinations violates, so 0.5.
    alerts = predicted_alerts(pred, horizons, TIGHT, probability_threshold=0.1)
    assert alerts.iloc[0]["probability"] == pytest.approx(0.5)

    # Weighted: the colliding hypothesis carries 10% of aircraft 0's probability.
    weighted = predicted_alerts(
        pred,
        horizons,
        TIGHT,
        probability_threshold=0.01,
        mode_probabilities=np.array([[0.1, 0.9], [1.0, 0.0]]),
    )
    assert weighted.iloc[0]["probability"] == pytest.approx(0.1)


def test_probability_threshold_filters_unlikely_conflicts():
    horizons = np.array([30.0])
    pred = np.zeros((2, 4, 1, 3))
    pred[0, 0, 0, 0] = 0.0
    pred[0, 1:, 0, 0] = 9000.0
    pred[1, :, 0, 0] = 0.0

    # One of aircraft 0's four hypotheses collides, and all four of aircraft 1's are
    # co-located, so 4 of the 16 combinations violate: 25%.
    assert predicted_alerts(pred, horizons, TIGHT, probability_threshold=0.5).empty
    loose = predicted_alerts(pred, horizons, TIGHT, probability_threshold=0.2)
    assert loose.iloc[0]["probability"] == pytest.approx(0.25)


def test_certain_conflict_has_probability_one():
    horizons = np.array([30.0])
    pred = np.zeros((2, 3, 1, 3))  # every hypothesis of both aircraft is co-located
    alerts = predicted_alerts(pred, horizons, TIGHT, probability_threshold=0.5)
    assert alerts.iloc[0]["probability"] == pytest.approx(1.0)


def test_reliability_compares_stated_probability_with_outcomes():
    scorer = AlertScorer(criterion=TIGHT, window_stride_s=10.0)

    # Two windows called 90% likely: one conflicts, one does not.
    event = make_window(converging(closing_speed=20.0, start_gap=1500.0))
    quiet = make_window(converging(closing_speed=0.0, start_gap=9000.0))
    for window in (event, quiet):
        scorer.update(
            window,
            pd.DataFrame([{"i": 0, "j": 1, "alert_time_s": 60.0, "probability": 0.9}]),
            120.0,
        )

    table = scorer.reliability()
    row = table.iloc[-1]
    assert row["n"] == 2
    assert row["predicted"] == pytest.approx(0.9)
    # Claimed 90%, happened 50% of the time: overconfident, and visibly so.
    assert row["observed"] == pytest.approx(0.5)


def test_reliability_is_empty_without_probabilities():
    scorer = AlertScorer(criterion=TIGHT)
    scorer.update(
        make_window(converging()), pd.DataFrame(columns=["i", "j", "alert_time_s"]), 120.0
    )
    assert scorer.reliability().empty
