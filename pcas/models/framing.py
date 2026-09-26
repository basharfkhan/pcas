"""Agent-centric framing, shared by every learned model.

A network fed raw airport coordinates has to learn the same manoeuvre separately for every
position and heading it can occur at. Re-expressing each aircraft's window in its own frame
removes that burden: translate so the last observed position is the origin, then rotate so
the aircraft is currently heading along +x.

The transform is invertible, so predictions come back to airport coordinates for scoring.
The runway frame still matters and is not discarded: the heading used for the rotation is
kept as a feature, so the model can still know whether an aircraft is on downwind for
runway 8 or final for 26.
"""

from __future__ import annotations

import numpy as np


def heading_from(obs: np.ndarray, fit_len: int = 4) -> np.ndarray:
    """Current heading per agent, in radians, from the last few observed steps.

    Averaging a few steps rather than taking the final one keeps ADS-B position noise from
    swinging the frame around.
    """
    obs = np.asarray(obs, dtype=float)
    tail = obs[:, -(fit_len + 1) :, :]
    steps = np.diff(tail, axis=1)
    mean_step = steps[..., :2].mean(axis=1)

    # A stationary aircraft has no meaningful heading; keep it at zero rather than NaN.
    still = np.linalg.norm(mean_step, axis=-1) < 1e-6
    heading = np.arctan2(mean_step[:, 1], mean_step[:, 0])
    return np.where(still, 0.0, heading)


def to_agent_frame(
    obs: np.ndarray, target: np.ndarray | None = None, fit_len: int = 4
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Move observations (and optionally targets) into each agent's own frame.

    Returns (obs_local, target_local, frame) where `frame` carries what `to_world` needs.
    """
    obs = np.asarray(obs, dtype=float)
    origin = obs[:, -1, :].copy()
    heading = heading_from(obs, fit_len)

    cos, sin = np.cos(-heading), np.sin(-heading)
    obs_local = _rotate_translate(obs, origin, cos, sin)
    target_local = (
        None if target is None else _rotate_translate(np.asarray(target, float), origin, cos, sin)
    )
    return obs_local, target_local, {"origin": origin, "heading": heading}


def to_world(pred_local: np.ndarray, frame: dict) -> np.ndarray:
    """Invert `to_agent_frame` for predictions shaped (A, T, 3) or (A, K, T, 3)."""
    pred_local = np.asarray(pred_local, dtype=float)
    origin, heading = frame["origin"], frame["heading"]

    squeeze = pred_local.ndim == 3
    if squeeze:
        pred_local = pred_local[:, None, :, :]

    cos, sin = np.cos(heading), np.sin(heading)
    x = pred_local[..., 0] * cos[:, None, None] - pred_local[..., 1] * sin[:, None, None]
    y = pred_local[..., 0] * sin[:, None, None] + pred_local[..., 1] * cos[:, None, None]
    z = pred_local[..., 2]

    out = np.stack(
        [
            x + origin[:, None, None, 0],
            y + origin[:, None, None, 1],
            z + origin[:, None, None, 2],
        ],
        axis=-1,
    )
    return out[:, 0] if squeeze else out


def _rotate_translate(
    points: np.ndarray, origin: np.ndarray, cos: np.ndarray, sin: np.ndarray
) -> np.ndarray:
    shifted = points - origin[:, None, :]
    x = shifted[..., 0] * cos[:, None] - shifted[..., 1] * sin[:, None]
    y = shifted[..., 0] * sin[:, None] + shifted[..., 1] * cos[:, None]
    return np.stack([x, y, shifted[..., 2]], axis=-1)


def features(obs_local: np.ndarray, wind: np.ndarray | None = None) -> np.ndarray:
    """Per-step model input: position, step velocity, and optional wind.

    Velocity is given explicitly rather than left for the network to difference out of
    positions, which is a free head start on an 11-step window.
    """
    obs_local = np.asarray(obs_local, dtype=float)
    velocity = np.diff(obs_local, axis=1, prepend=obs_local[:, :1, :])
    parts = [obs_local, velocity]

    if wind is not None:
        wind = np.asarray(wind, dtype=float).reshape(1, 1, -1)
        parts.append(np.broadcast_to(wind, (*obs_local.shape[:2], wind.shape[-1])))

    return np.concatenate(parts, axis=-1)
