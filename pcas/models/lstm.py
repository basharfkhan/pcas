"""Single-aircraft LSTM baseline.

Deliberately not multi-agent: this measures what learning alone buys over physics, so that
when the Transformer adds attention between aircraft, the gain from *interaction* can be
separated from the gain from *learning*. Reporting a social model against physics only
would conflate the two.

The model sees one aircraft's 11 s window in that aircraft's own frame and predicts 12
waypoints at 10 s spacing. It predicts offsets from the last observed position, so
"keep doing what you are doing" is near the origin of its output space rather than
something it has to learn to produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pcas.models.baselines import Predictor
from pcas.models.framing import features, to_agent_frame, to_world


@dataclass
class LSTMConfig:
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    n_waypoints: int = 12
    use_wind: bool = True
    # Metres. Positions and velocities are divided by this before the network sees them:
    # raw values run to thousands of metres, which saturates activations.
    position_scale: float = 1000.0
    velocity_scale: float = 50.0
    # Targets are scaled too. Without this the head has to emit raw metres (thousands),
    # which leaves almost every sample in the Huber loss's linear regime: gradients arrive
    # with near-constant magnitude and the output layer barely moves. A 2-layer net limps
    # through it; a 3-layer net collapses to predicting a constant.
    target_scale: float = 1000.0


def build_module(config: LSTMConfig):
    """Construct the torch module. Imported lazily so the package works without torch."""
    import torch
    from torch import nn

    class TrajectoryLSTM(nn.Module):
        def __init__(self, cfg: LSTMConfig):
            super().__init__()
            self.cfg = cfg
            in_features = 6 + (2 if cfg.use_wind else 0)
            self.encoder = nn.LSTM(
                input_size=in_features,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.ReLU(),
                nn.Linear(cfg.hidden_size, cfg.n_waypoints * 3),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, (hidden, _) = self.encoder(x)
            out = self.head(hidden[-1])
            return out.view(-1, self.cfg.n_waypoints, 3)

    return TrajectoryLSTM(config)


def normalise(feats: np.ndarray, config: LSTMConfig) -> np.ndarray:
    """Scale positions, velocities and wind into a comparable range."""
    scaled = feats.copy()
    scaled[..., 0:3] /= config.position_scale
    scaled[..., 3:6] /= config.velocity_scale
    return scaled


def make_samples(windows, config: LSTMConfig) -> tuple[np.ndarray, np.ndarray]:
    """Flatten windows into per-aircraft training samples.

    Returns (inputs, targets): inputs are (N, obs_len, n_features) normalised model inputs,
    targets are (N, n_waypoints, 3) offsets in the aircraft's own frame, divided by
    `config.target_scale`. Multiply by that scale to get metres back.
    """
    inputs, targets = [], []

    for window in windows:
        obs_local, future_local, _ = to_agent_frame(window.obs, window.future)
        wind = window.wind if config.use_wind else None
        feats = normalise(features(obs_local, wind), config)
        inputs.append(feats)
        targets.append(future_local / config.target_scale)

    if not inputs:
        n_features = 6 + (2 if config.use_wind else 0)
        return (
            np.zeros((0, 0, n_features), dtype=np.float32),
            np.zeros((0, config.n_waypoints, 3), dtype=np.float32),
        )

    return (
        np.concatenate(inputs).astype(np.float32),
        np.concatenate(targets).astype(np.float32),
    )


@dataclass
class LSTMPredictor(Predictor):
    """Wraps a trained module in the same interface as the physics baselines."""

    module: object = None
    config: LSTMConfig = field(default_factory=LSTMConfig)
    waypoint_horizons: tuple[float, ...] = tuple(float(10 * k) for k in range(1, 13))
    device: str = "cpu"
    name: str = "lstm"

    def predict(self, obs: np.ndarray, horizons_s: np.ndarray) -> np.ndarray:
        import torch

        obs_local, _, frame = to_agent_frame(obs)
        wind = getattr(self, "_wind", None) if self.config.use_wind else None
        if self.config.use_wind and wind is None:
            wind = np.zeros(2)
        feats = normalise(features(obs_local, wind), self.config).astype(np.float32)

        self.module.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(feats).to(self.device)
            local = self.module(tensor).cpu().numpy() * self.config.target_scale

        world = to_world(local, frame)
        return self._resample(world, horizons_s)[:, None, :, :]

    def with_wind(self, wind: np.ndarray) -> LSTMPredictor:
        """Attach the window's wind vector before predicting."""
        self._wind = wind
        return self

    def _resample(self, waypoints: np.ndarray, horizons_s: np.ndarray) -> np.ndarray:
        """Interpolate the 10 s waypoints onto whatever horizons are asked for.

        Alerting checks every second, so the dense horizons must be filled in. Linear
        interpolation between waypoints is the honest choice: it adds no information the
        model did not produce.
        """
        horizons = np.asarray(horizons_s, dtype=float)
        native = np.asarray(self.waypoint_horizons, dtype=float)
        if np.array_equal(horizons, native):
            return waypoints

        out = np.empty((waypoints.shape[0], len(horizons), 3))
        for axis in range(3):
            for a in range(waypoints.shape[0]):
                out[a, :, axis] = np.interp(horizons, native, waypoints[a, :, axis])
        return out


def load_predictor(path: str | Path, device: str | None = None) -> LSTMPredictor:
    """Load a trained checkpoint into a predictor usable by every evaluation script."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    config = LSTMConfig(**blob["config"])

    module = build_module(config)
    module.load_state_dict(blob["state_dict"])
    module.to(device).eval()

    return LSTMPredictor(module=module, config=config, device=device)
