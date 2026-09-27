"""Conflict labelling, closure-rate alerting, and lead time vs false alarms.

This is the metric PCAS is actually judged on. Trajectory error is a means to it.

**What counts as a conflict.** Two thresholds, both configurable:

- `NMAC` (near midair collision): 500 ft horizontal, 100 ft vertical. The standard
  definition of "too close" in aviation safety work. Rare, so statistics on it are thin.
- `PROXIMITY` (the default): 0.5 nm horizontal, 500 ft vertical. Not dangerous by itself,
  but it is the kind of convergence a pilot at a non-towered field wants to hear about
  early, and it happens often enough to measure.

**What counts as an alert.** Both the baseline and the model are scored the same way: at
the last observed instant, predict where every aircraft goes, then check every pair for a
threshold violation inside the horizon. The difference is only in how the prediction is
made.

- `closure_rate_alerts` is the TCAS-style comparison: straight-line relative motion, time
  to closest point of approach, alert if the projected miss distance breaks the threshold
  within `tau_s`. This is, in substance, what certified traffic alerting does, and it is
  the baseline the headline result quotes.
- `predicted_alerts` takes any predictor's output, so constant velocity, Kalman and later
  the Transformer are all scored by identical code. With K hypotheses, the share that
  violate the threshold becomes the alert probability, which is what a threshold sweep
  then trades off against false alarms.

**Lead time** is measured from the alert instant (the last observed sample) to the moment
the aircraft actually violate the threshold in the recorded data. A predictor that only
sees 20 s ahead cannot earn a 90 s lead time, which is the whole point of the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEET = 0.3048
NM = 1852.0


@dataclass(frozen=True)
class ConflictCriterion:
    """Separation thresholds. A conflict needs BOTH to be broken at the same instant."""

    horizontal_m: float
    vertical_m: float
    name: str

    def violated(self, horizontal: np.ndarray, vertical: np.ndarray) -> np.ndarray:
        return (horizontal < self.horizontal_m) & (vertical < self.vertical_m)


NMAC = ConflictCriterion(horizontal_m=500 * FEET, vertical_m=100 * FEET, name="nmac")
PROXIMITY = ConflictCriterion(horizontal_m=0.5 * NM, vertical_m=500 * FEET, name="proximity")


def pair_separations(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Horizontal and vertical separation over time for every aircraft pair.

    `traj` is (n_agents, T, 3). Returns (horizontal, vertical) of shape (n_pairs, T) plus
    the pair index list.
    """
    traj = np.asarray(traj, dtype=float)
    n = traj.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        empty = np.zeros((0, traj.shape[1]))
        return empty, empty, pairs

    a = np.array([traj[i] for i, _ in pairs])
    b = np.array([traj[j] for _, j in pairs])
    horizontal = np.linalg.norm(a[..., :2] - b[..., :2], axis=-1)
    vertical = np.abs(a[..., 2] - b[..., 2])
    return horizontal, vertical, pairs


def first_violation(
    traj: np.ndarray, criterion: ConflictCriterion, times_s: np.ndarray
) -> pd.DataFrame:
    """Earliest threshold violation per pair, or nothing for pairs that stay apart."""
    horizontal, vertical, pairs = pair_separations(traj)
    if not pairs:
        return pd.DataFrame(columns=["i", "j", "onset_s", "min_horizontal_m", "min_vertical_m"])

    violated = criterion.violated(horizontal, vertical)
    rows = []
    for k, (i, j) in enumerate(pairs):
        hits = np.flatnonzero(violated[k])
        if hits.size == 0:
            continue
        rows.append(
            {
                "i": i,
                "j": j,
                "onset_s": float(times_s[hits[0]]),
                "min_horizontal_m": float(horizontal[k].min()),
                "min_vertical_m": float(vertical[k].min()),
            }
        )
    return pd.DataFrame(rows, columns=["i", "j", "onset_s", "min_horizontal_m", "min_vertical_m"])


def closure_rate_alerts(
    obs: np.ndarray,
    criterion: ConflictCriterion,
    tau_s: float = 40.0,
    fit_len: int | None = None,
) -> pd.DataFrame:
    """TCAS-style alerting: straight-line closure to the closest point of approach.

    For each pair, take relative position and velocity at the last observed sample, solve
    for the time of closest approach, and alert when that falls within `tau_s` and the
    projected separation there breaks the threshold. Pairs already in violation alert
    immediately, as a real system would.

    `tau_s` is the alerting horizon. TCAS II uses roughly 25 to 40 s depending on
    altitude; sweeping it traces the baseline's own lead time / false alarm trade-off.
    """
    from pcas.models.baselines import _fit_velocity

    position, velocity = _fit_velocity(obs, fit_len)
    n = position.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    rows = []
    for i, j in pairs:
        dp = position[i] - position[j]
        dv = velocity[i] - velocity[j]

        speed_sq = float(dv @ dv)
        # Identical velocities never reach a closest approach; treat as "now".
        t_cpa = 0.0 if speed_sq < 1e-9 else float(-(dp @ dv) / speed_sq)

        # Separation now, and at the projected closest approach.
        now_h = float(np.linalg.norm(dp[:2]))
        now_v = float(abs(dp[2]))
        if bool(criterion.violated(np.array(now_h), np.array(now_v))):
            rows.append({"i": i, "j": j, "alert_time_s": 0.0, "tau_s": 0.0})
            continue

        if t_cpa <= 0.0 or t_cpa > tau_s:
            continue  # diverging, or too far ahead to alert on

        at_cpa = dp + dv * t_cpa
        cpa_h = float(np.linalg.norm(at_cpa[:2]))
        cpa_v = float(abs(at_cpa[2]))
        if bool(criterion.violated(np.array(cpa_h), np.array(cpa_v))):
            rows.append({"i": i, "j": j, "alert_time_s": t_cpa, "tau_s": t_cpa})

    return pd.DataFrame(rows, columns=["i", "j", "alert_time_s", "tau_s"])


def predicted_alerts(
    pred: np.ndarray,
    horizons_s: np.ndarray,
    criterion: ConflictCriterion,
    probability_threshold: float = 0.5,
    mode_probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    """Alerts from any predictor's trajectories, with a conflict probability.

    `pred` is (n_agents, K, T, 3). Two aircraft each carrying K hypotheses give K x K
    possible futures for the pair, so the probability that they conflict is the weight of
    the combinations in which they do:

        P(conflict) = sum over (k, m) of p_i(k) * p_j(m) * [hypotheses k and m violate]

    Treating each aircraft's hypotheses as independent is an approximation, since two
    aircraft sequencing with each other are correlated, but it is an honest one and it turns
    a yes/no alert into a number that can be weighed: "30% chance inside the next minute".

    `mode_probabilities` is (n_agents, K). Without it every hypothesis is equally likely,
    which reduces to the fraction of violating combinations.
    """
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 3:
        pred = pred[:, None, :, :]
    n_agents, n_hyp, _, _ = pred.shape
    horizons = np.asarray(horizons_s, dtype=float)

    if mode_probabilities is None:
        weights = np.full((n_agents, n_hyp), 1.0 / n_hyp)
    else:
        weights = np.asarray(mode_probabilities, dtype=float).reshape(n_agents, n_hyp)

    rows = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            probability, earliest = 0.0, np.inf
            for k in range(n_hyp):
                for m in range(n_hyp):
                    horizontal = np.linalg.norm(pred[i, k, :, :2] - pred[j, m, :, :2], axis=-1)
                    vertical = np.abs(pred[i, k, :, 2] - pred[j, m, :, 2])
                    hits = np.flatnonzero(criterion.violated(horizontal, vertical))
                    if hits.size:
                        probability += float(weights[i, k] * weights[j, m])
                        earliest = min(earliest, float(horizons[hits[0]]))

            if probability >= probability_threshold and np.isfinite(earliest):
                rows.append({"i": i, "j": j, "alert_time_s": earliest, "probability": probability})

    return pd.DataFrame(rows, columns=["i", "j", "alert_time_s", "probability"])


class AlertScorer:
    """Scores alerts against recorded outcomes across many windows.

    Counting is per aircraft PAIR per window. A window with five aircraft holds ten pairs,
    and treating that as one opportunity would hide most of both the events and the false
    alarms.
    """

    def __init__(self, criterion: ConflictCriterion = PROXIMITY, window_stride_s: float = 10.0):
        self.criterion = criterion
        self.window_stride_s = window_stride_s
        self._rows: list[dict] = []
        self._windows = 0

    def update(
        self, window, alerts: pd.DataFrame, horizon_s: float, times_s: np.ndarray | None = None
    ) -> None:
        """Compare one window's alerts with what actually happened."""
        if window.future_dense is None:
            raise ValueError("window has no future_dense; conflicts need 1 Hz truth")

        dense_times = (
            times_s
            if times_s is not None
            else np.arange(1, window.future_dense.shape[1] + 1, dtype=float)
        )
        truth = first_violation(window.future_dense, self.criterion, dense_times)
        truth = truth[truth["onset_s"] <= horizon_s]

        truth_pairs = {(int(r.i), int(r.j)): float(r.onset_s) for r in truth.itertuples()}
        alert_pairs = {(int(r.i), int(r.j)): float(r.alert_time_s) for r in alerts.itertuples()}
        alert_probability = {
            (int(r.i), int(r.j)): float(getattr(r, "probability", 1.0)) for r in alerts.itertuples()
        }

        n = window.n_agents
        for i in range(n):
            for j in range(i + 1, n):
                onset = truth_pairs.get((i, j))
                alerted = (i, j) in alert_pairs
                self._rows.append(
                    {
                        "scene_id": window.scene_id,
                        "date": window.date,
                        "start_frame": window.start_frame,
                        "event": onset is not None,
                        "onset_s": onset,
                        "alerted": alerted,
                        "alert_time_s": alert_pairs.get((i, j)),
                        "probability": alert_probability.get((i, j)),
                    }
                )
        self._windows += 1

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def summary(self) -> dict[str, float]:
        df = self.frame()
        if df.empty:
            raise ValueError("nothing scored")

        events = df[df["event"]]
        quiet = df[~df["event"]]
        detected = events[events["alerted"]]
        false_alarms = int(quiet["alerted"].sum())

        nan = float("nan")
        n_events, n_quiet = len(events), len(quiet)
        # Windows overlap by design (stride < window length), so "per hour" is expressed in
        # scene time: each window advances the clock by the stride.
        scene_hours = self._windows * self.window_stride_s / 3600.0

        out = {
            "pair_windows": float(len(df)),
            "windows": float(self._windows),
            "events": float(n_events),
            "detected": float(len(detected)),
            "detection_rate": len(detected) / n_events if n_events else nan,
            "false_alarms": float(false_alarms),
            "false_alarm_rate": false_alarms / n_quiet if n_quiet else nan,
            "false_alarms_per_hour": false_alarms / scene_hours if scene_hours else nan,
            "median_lead_time_s": float(detected["onset_s"].median()) if len(detected) else nan,
            "mean_lead_time_s": float(detected["onset_s"].mean()) if len(detected) else nan,
        }
        return out

    def reliability(self, bins=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> pd.DataFrame:
        """Predicted conflict probability against how often a conflict actually followed.

        A multimodal model states a number, and the number is only worth stating if it means
        something: of the pairs called 30% likely, close to 30% should go on to conflict.
        Only alerted pairs carry a probability, so this describes the calibration of the
        alerts raised rather than of every pair in the airspace.
        """
        df = self.frame()
        if "probability" not in df.columns:
            return pd.DataFrame(columns=["probability_bin", "n", "predicted", "observed"])

        alerted = df[df["alerted"] & df["probability"].notna()].copy()
        if alerted.empty:
            return pd.DataFrame(columns=["probability_bin", "n", "predicted", "observed"])

        alerted["bin"] = pd.cut(alerted["probability"], bins=list(bins), include_lowest=True)
        grouped = alerted.groupby("bin", observed=True).agg(
            n=("event", "size"),
            predicted=("probability", "mean"),
            observed=("event", "mean"),
        )
        return grouped.reset_index().rename(columns={"bin": "probability_bin"})

    def detection_by_lead_time(self, edges=(0, 30, 60, 90, 120)) -> pd.DataFrame:
        """Detection rate split by how far ahead the event actually was.

        This is where a longer-horizon predictor should separate from closure-rate logic:
        both should catch an event 10 s out, but only a model that understands the pattern
        can catch one 90 s out.
        """
        df = self.frame()
        events = df[df["event"]].copy()
        if events.empty:
            return pd.DataFrame(columns=["lead_bucket_s", "events", "detected", "detection_rate"])

        events["bucket"] = pd.cut(events["onset_s"], bins=list(edges), right=True)
        grouped = events.groupby("bucket", observed=True)["alerted"].agg(["size", "sum"])
        return pd.DataFrame(
            {
                "lead_bucket_s": [str(b) for b in grouped.index],
                "events": grouped["size"].to_numpy(),
                "detected": grouped["sum"].to_numpy(),
                "detection_rate": (grouped["sum"] / grouped["size"]).to_numpy(),
            }
        )
