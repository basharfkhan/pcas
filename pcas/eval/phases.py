"""Label what an aircraft is doing, so errors can be read by flight phase.

An average error over every window hides the thing worth knowing. A light aircraft transiting
overhead at 3,000 ft is nearly free to predict; the same aircraft turning base at 800 ft is
where conflicts happen and where a model earns its place. Splitting error by phase says which
of those the model is good at.

Phases are assigned from motion alone, since the data carries no flight plans: altitude above
the field, climb rate, distance from the runway, and turn rate. The rules are deliberately
simple and stated here rather than tuned, because the point is to slice the errors honestly,
not to build a classifier.
"""

from __future__ import annotations

import numpy as np

# Thresholds, in the units the pipeline uses: metres, metres per second, degrees per second.
PATTERN_RADIUS_M = 5000.0
PATTERN_CEILING_M = 700.0  # above field elevation; a circuit is flown at roughly 300 to 450 m
APPROACH_RADIUS_M = 3000.0
APPROACH_CEILING_M = 300.0
CLIMB_RATE_MS = 1.5
TURN_RATE_DEG_S = 1.5

PHASES = ("final approach", "pattern turn", "pattern", "climb", "descent", "transit")


def motion_features(obs: np.ndarray, field_elev_m: float = 0.0) -> dict[str, np.ndarray]:
    """Speed, climb rate, turn rate, range and height, per aircraft, from an observed window."""
    obs = np.asarray(obs, dtype=float)
    steps = np.diff(obs, axis=1)

    speed = np.linalg.norm(steps[..., :2], axis=-1).mean(axis=1)
    climb = steps[..., 2].mean(axis=1)

    heading = np.unwrap(np.arctan2(steps[..., 1], steps[..., 0]), axis=1)
    turn = (
        np.degrees(np.diff(heading, axis=1)).mean(axis=1)
        if heading.shape[1] > 1
        else np.zeros(obs.shape[0])
    )

    last = obs[:, -1, :]
    return {
        "speed_ms": speed,
        "climb_ms": climb,
        "turn_deg_s": turn,
        "range_m": np.linalg.norm(last[:, :2], axis=-1),
        "height_m": last[:, 2] - field_elev_m,
    }


def classify(obs: np.ndarray, field_elev_m: float = 0.0) -> np.ndarray:
    """One phase label per aircraft in the window.

    Checked most specific first: an aircraft low and close to the runway is on final however
    it is turning, and a turning aircraft in the circuit is a pattern turn rather than merely
    a descent.
    """
    f = motion_features(obs, field_elev_m)
    n = obs.shape[0]
    labels = np.full(n, "transit", dtype=object)

    in_pattern = (f["range_m"] < PATTERN_RADIUS_M) & (f["height_m"] < PATTERN_CEILING_M)
    on_final = (
        (f["range_m"] < APPROACH_RADIUS_M)
        & (f["height_m"] < APPROACH_CEILING_M)
        & (f["climb_ms"] < 0)
    )
    turning = np.abs(f["turn_deg_s"]) >= TURN_RATE_DEG_S

    labels[f["climb_ms"] >= CLIMB_RATE_MS] = "climb"
    labels[f["climb_ms"] <= -CLIMB_RATE_MS] = "descent"
    labels[in_pattern] = "pattern"
    labels[in_pattern & turning] = "pattern turn"
    labels[on_final] = "final approach"

    return labels
