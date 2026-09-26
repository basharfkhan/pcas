"""Physics baselines.

These are what a learned model has to beat, and they are not strawmen: for the first
20 to 30 seconds, an aircraft in cruise really does keep going straight, which is exactly
the assumption certified traffic alerting makes. The interesting question is where that
assumption breaks down, which is in the pattern, in turns, and further out in time.

Every predictor takes observed positions `(n_agents, obs_len, 3)` in metres at 1 Hz and a
list of horizons in seconds past the last observation, and returns `(n_agents, K, T, 3)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_INTERVAL_S = 1.0


class Predictor:
    """Interface shared by baselines and, later, the learned models."""

    name = "predictor"

    def predict(self, obs: np.ndarray, horizons_s: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _fit_velocity(obs: np.ndarray, fit_len: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares position and velocity at the last observed sample.

    A finite difference over the final two samples would be simpler, but ADS-B positions
    are noisy and altitude is quantised, so a fit over several samples is steadier.
    """
    obs = np.asarray(obs, dtype=float)
    if fit_len is not None:
        obs = obs[:, -fit_len:, :]

    n = obs.shape[1]
    if n < 2:
        raise ValueError("need at least two observations to estimate velocity")

    t = np.arange(n, dtype=float) * SAMPLE_INTERVAL_S
    t_centred = t - t.mean()
    denom = (t_centred**2).sum()

    mean_pos = obs.mean(axis=1)
    velocity = (obs * t_centred[None, :, None]).sum(axis=1) / denom
    # Extrapolate the fitted line to the final observed instant.
    position = mean_pos + velocity * (t[-1] - t.mean())
    return position, velocity


@dataclass
class ConstantVelocity(Predictor):
    """Straight line at the current velocity. The dead-reckoning baseline."""

    fit_len: int | None = None
    name: str = "constant_velocity"

    def predict(self, obs: np.ndarray, horizons_s: np.ndarray) -> np.ndarray:
        position, velocity = _fit_velocity(obs, self.fit_len)
        h = np.asarray(horizons_s, dtype=float)
        pred = position[:, None, :] + velocity[:, None, :] * h[None, :, None]
        return pred[:, None, :, :]


@dataclass
class ConstantTurnRate(Predictor):
    """Constant speed, constant turn rate in the horizontal plane; constant climb rate.

    Aircraft in a traffic pattern spend much of their time in standard-rate turns, which
    a straight-line model cannot represent at all.
    """

    fit_len: int | None = None
    name: str = "constant_turn_rate"

    def predict(self, obs: np.ndarray, horizons_s: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float)
        window = obs if self.fit_len is None else obs[:, -self.fit_len :, :]
        h = np.asarray(horizons_s, dtype=float)

        steps = np.diff(window, axis=1)
        speed = np.linalg.norm(steps[..., :2], axis=-1).mean(axis=1) / SAMPLE_INTERVAL_S
        climb = steps[..., 2].mean(axis=1) / SAMPLE_INTERVAL_S

        heading = np.arctan2(steps[..., 1], steps[..., 0])
        # Unwrap before differencing so a pass through +/-pi is not read as a huge turn.
        unwrapped = np.unwrap(heading, axis=1)
        turn_rate = (
            np.diff(unwrapped, axis=1).mean(axis=1) / SAMPLE_INTERVAL_S
            if unwrapped.shape[1] > 1
            else np.zeros(obs.shape[0])
        )

        last = obs[:, -1, :]
        heading_now = unwrapped[:, -1]

        # Integrate a circular arc. Where the turn rate is ~0 this degenerates to a
        # straight line, so it is handled separately to avoid dividing by zero.
        theta = heading_now[:, None] + turn_rate[:, None] * h[None, :]
        turning = np.abs(turn_rate) > 1e-6
        safe_rate = np.where(turning, turn_rate, 1.0)[:, None]

        dx = np.where(
            turning[:, None],
            (np.sin(theta) - np.sin(heading_now)[:, None]) / safe_rate,
            np.cos(heading_now)[:, None] * h[None, :],
        )
        dy = np.where(
            turning[:, None],
            (np.cos(heading_now)[:, None] - np.cos(theta)) / safe_rate,
            np.sin(heading_now)[:, None] * h[None, :],
        )

        x = last[:, 0:1] + speed[:, None] * dx
        y = last[:, 1:2] + speed[:, None] * dy
        z = last[:, 2:3] + climb[:, None] * h[None, :]

        return np.stack([x, y, z], axis=-1)[:, None, :, :]


@dataclass
class KalmanConstantVelocity(Predictor):
    """Constant-velocity Kalman filter over the observed window, then propagated.

    Same motion model as ConstantVelocity, but it weighs measurement noise against
    process noise instead of fitting every sample equally, so it reacts faster to a
    recent change in velocity.
    """

    position_noise_m: float = 25.0
    acceleration_noise: float = 1.5
    name: str = "kalman_cv"

    def predict(self, obs: np.ndarray, horizons_s: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float)
        n_agents, obs_len, _ = obs.shape
        h = np.asarray(horizons_s, dtype=float)
        out = np.empty((n_agents, 1, len(h), 3))

        dt = SAMPLE_INTERVAL_S
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        # Discrete white-noise acceleration model.
        q = self.acceleration_noise**2 * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        r = self.position_noise_m**2
        measure = np.array([[1.0, 0.0]])

        for a in range(n_agents):
            for axis in range(3):
                series = obs[a, :, axis]
                state = np.array([series[0], (series[1] - series[0]) / dt])
                cov = np.array([[r, 0.0], [0.0, (self.position_noise_m / dt) ** 2]])

                for k in range(1, obs_len):
                    state = transition @ state
                    cov = transition @ cov @ transition.T + q

                    residual = series[k] - (measure @ state)[0]
                    gain = (cov @ measure.T) / ((measure @ cov @ measure.T)[0, 0] + r)
                    state = state + (gain[:, 0] * residual)
                    cov = cov - gain @ measure @ cov

                out[a, 0, :, axis] = state[0] + state[1] * h

        return out


DEFAULT_BASELINES: tuple[Predictor, ...] = (
    ConstantVelocity(),
    ConstantVelocity(fit_len=4, name="constant_velocity_short_fit"),
    ConstantTurnRate(),
    KalmanConstantVelocity(),
)
