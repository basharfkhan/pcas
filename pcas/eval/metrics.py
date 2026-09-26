"""Trajectory prediction metrics.

Built before any model, so every predictor is scored by the same code. Conventions:

- Predictions are `(n_agents, K, n_waypoints, 3)` in metres, K hypotheses per aircraft.
  A single-hypothesis predictor passes `(n_agents, n_waypoints, 3)` and gets K = 1.
- Truth is `(n_agents, n_waypoints, 3)`.
- Errors are reported in 3D, horizontally, and vertically. Keeping them apart matters
  here: a light aircraft's altitude is quantised to 100 ft in ADS-B, so mixing vertical
  error into one number hides where a model is actually wrong.
- With K > 1, metrics are "best of K" (minADE/minFDE), the standard in motion
  forecasting. Reporting mean-over-K instead would punish a model for offering a
  hypothesis that did not happen, which is the whole point of multimodal prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Horizontal miss threshold, roughly the NMAC horizontal bound (500 ft).
DEFAULT_MISS_M = 152.4


def _as_hypotheses(pred: np.ndarray) -> np.ndarray:
    """Normalise (A, T, 3) to (A, 1, T, 3); leave (A, K, T, 3) alone."""
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 3:
        return pred[:, None, :, :]
    if pred.ndim == 4:
        return pred
    raise ValueError(f"expected (A, T, 3) or (A, K, T, 3), got shape {pred.shape}")


def displacement(pred: np.ndarray, truth: np.ndarray, axes: str = "3d") -> np.ndarray:
    """Per-agent, per-hypothesis, per-waypoint error in metres: (A, K, T).

    `axes` is "3d", "horizontal" (x, y) or "vertical" (|dz|).
    """
    pred = _as_hypotheses(pred)
    truth = np.asarray(truth, dtype=float)
    if truth.ndim != 3:
        raise ValueError(f"truth must be (A, T, 3), got {truth.shape}")
    if pred.shape[0] != truth.shape[0] or pred.shape[2] != truth.shape[1]:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs truth {truth.shape}")

    delta = pred - truth[:, None, :, :]
    if axes == "3d":
        return np.linalg.norm(delta, axis=-1)
    if axes == "horizontal":
        return np.linalg.norm(delta[..., :2], axis=-1)
    if axes == "vertical":
        return np.abs(delta[..., 2])
    raise ValueError(f"unknown axes {axes!r}")


def min_ade(pred: np.ndarray, truth: np.ndarray, axes: str = "3d") -> np.ndarray:
    """Average displacement over the horizon, best hypothesis per agent: (A,)."""
    return displacement(pred, truth, axes).mean(axis=2).min(axis=1)


def min_fde(pred: np.ndarray, truth: np.ndarray, axes: str = "3d") -> np.ndarray:
    """Final displacement error, best hypothesis per agent: (A,).

    The best hypothesis is chosen by the final point, not by the average, so minADE and
    minFDE may pick different hypotheses. That is the usual convention.
    """
    return displacement(pred, truth, axes)[:, :, -1].min(axis=1)


def miss(pred: np.ndarray, truth: np.ndarray, threshold_m: float = DEFAULT_MISS_M) -> np.ndarray:
    """Per agent: did every hypothesis end further than `threshold_m` away? (A,) bool."""
    return min_fde(pred, truth, axes="horizontal") > threshold_m


def per_horizon(
    pred: np.ndarray, truth: np.ndarray, horizons_s: np.ndarray, axes: str = "3d"
) -> pd.DataFrame:
    """Error as a function of how far ahead the waypoint is."""
    err = displacement(pred, truth, axes).min(axis=1)  # best hypothesis per agent
    horizons = np.asarray(horizons_s)
    if err.shape[1] != len(horizons):
        raise ValueError(f"{err.shape[1]} waypoints but {len(horizons)} horizons")

    return pd.DataFrame(
        {
            "horizon_s": horizons,
            "mean_m": err.mean(axis=0),
            "median_m": np.median(err, axis=0),
            "p95_m": np.percentile(err, 95, axis=0),
            "n": err.shape[0],
        }
    )


class MetricAccumulator:
    """Collects per-agent errors across many windows, then summarises once.

    Averaging per-window means and then averaging those would weight a window holding one
    aircraft the same as one holding twenty, so agents are pooled instead.
    """

    def __init__(self, horizons_s: np.ndarray, miss_threshold_m: float = DEFAULT_MISS_M):
        self.horizons_s = np.asarray(horizons_s)
        self.miss_threshold_m = miss_threshold_m
        self._err_3d: list[np.ndarray] = []
        self._err_h: list[np.ndarray] = []
        self._err_v: list[np.ndarray] = []
        self._misses: list[np.ndarray] = []
        self._n_agents = 0
        self._n_windows = 0

    def update(self, pred: np.ndarray, truth: np.ndarray) -> None:
        self._err_3d.append(displacement(pred, truth, "3d").min(axis=1))
        self._err_h.append(displacement(pred, truth, "horizontal").min(axis=1))
        self._err_v.append(displacement(pred, truth, "vertical").min(axis=1))
        self._misses.append(miss(pred, truth, self.miss_threshold_m))
        self._n_agents += truth.shape[0]
        self._n_windows += 1

    @property
    def empty(self) -> bool:
        return self._n_windows == 0

    def summary(self) -> dict[str, float]:
        if self.empty:
            raise ValueError("no windows accumulated")

        err_3d = np.concatenate(self._err_3d)
        err_h = np.concatenate(self._err_h)
        err_v = np.concatenate(self._err_v)
        misses = np.concatenate(self._misses)

        return {
            "windows": float(self._n_windows),
            "agents": float(self._n_agents),
            "minADE_m": float(err_3d.mean()),
            "minFDE_m": float(err_3d[:, -1].mean()),
            "minADE_horizontal_m": float(err_h.mean()),
            "minFDE_horizontal_m": float(err_h[:, -1].mean()),
            "minFDE_vertical_m": float(err_v[:, -1].mean()),
            "median_FDE_m": float(np.median(err_3d[:, -1])),
            "p95_FDE_m": float(np.percentile(err_3d[:, -1], 95)),
            f"miss_rate_{int(self.miss_threshold_m)}m": float(misses.mean()),
        }

    def horizon_table(self) -> pd.DataFrame:
        err = np.concatenate(self._err_3d)
        return pd.DataFrame(
            {
                "horizon_s": self.horizons_s,
                "mean_m": err.mean(axis=0),
                "median_m": np.median(err, axis=0),
                "p95_m": np.percentile(err, 95, axis=0),
                "n": len(err),
            }
        )
