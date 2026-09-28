"""Multi-agent Transformer with social attention.

This is the model the earlier results argued for. Three measurements said the single-aircraft
LSTM was limited by its inputs rather than its size: 14x the parameters bought 0.8%, fixing
the optimisation moved final error 0.5%, and the aircraft it might conflict with was not an
input at all. So the change here is not more capacity, it is more context.

Shape of it, per target aircraft:

1. Every aircraft in the window is re-expressed in the target's frame (`scene_frames`), so
   neighbour positions are real relative geometry rather than private coordinates.
2. A shared temporal Transformer encodes each aircraft's 11 s track into one vector.
3. The target's vector attends over its neighbours' vectors (social attention), producing a
   context vector. `use_social=False` zeroes that context, which is the ablation: the same
   weights, the same data, the same loss, with the neighbours removed. Any gain that appears
   between the two is attributable to interaction rather than to learning in general.
4. An MLP decodes 12 waypoint offsets in the target's frame, as the LSTM does, so "carry on
   as you are" sits near the origin of the output space.

Neighbours are the nearest `max_neighbours` at the last observed instant. Aircraft further
away in a terminal area are rarely the ones you are about to converge with, and the cap keeps
attention cost bounded when a busy scene holds twenty aircraft.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pcas.models.baselines import Predictor
from pcas.models.framing import features, scene_frames, to_world

log = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    d_model: int = 128
    n_heads: int = 4
    n_temporal_layers: int = 2
    n_social_layers: int = 1
    ff_multiplier: int = 4
    dropout: float = 0.1
    max_neighbours: int = 6
    n_waypoints: int = 12
    # Number of trajectory hypotheses. 1 keeps the single-mode behaviour (and loads the
    # checkpoints trained before this existed); 6 is the usual choice in motion forecasting.
    # An aircraft on downwind may turn base or extend, and averaging those two futures
    # produces a path it would never fly, so a single mode is a modelling error rather than
    # merely a weaker model.
    n_modes: int = 1
    use_wind: bool = True
    use_social: bool = True
    position_scale: float = 1000.0
    velocity_scale: float = 50.0
    target_scale: float = 1000.0

    @property
    def n_features(self) -> int:
        return 6 + (2 if self.use_wind else 0)


def build_module(config: TransformerConfig):
    """Construct the torch module. Imported lazily so the package works without torch."""
    import torch
    from torch import nn

    class SocialTransformer(nn.Module):
        def __init__(self, cfg: TransformerConfig):
            super().__init__()
            self.cfg = cfg

            self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)
            # Learned positional encoding: the window is a fixed 11 steps, so there is no
            # need for a scheme that extrapolates to unseen lengths.
            self.time_embedding = nn.Parameter(torch.zeros(1, 11, cfg.d_model))
            nn.init.normal_(self.time_embedding, std=0.02)

            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * cfg.ff_multiplier,
                dropout=cfg.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=cfg.n_temporal_layers)

            self.social_layers = nn.ModuleList(
                nn.MultiheadAttention(
                    cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
                )
                for _ in range(cfg.n_social_layers)
            )
            self.social_norms = nn.ModuleList(
                nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_social_layers)
            )

            self.head = nn.Sequential(
                nn.Linear(cfg.d_model * 2, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.n_modes * cfg.n_waypoints * 3),
            )
            # One logit per hypothesis. With n_modes == 1 this is a constant and the
            # softmax over it is 1.0, so single-mode behaviour is unchanged.
            self.mode_logits = nn.Sequential(
                nn.Linear(cfg.d_model * 2, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.n_modes),
            )

        def encode(self, tracks: torch.Tensor) -> torch.Tensor:
            """(B, T, F) -> (B, d_model), pooling the encoded sequence at its last step."""
            hidden = self.input_proj(tracks) + self.time_embedding[:, : tracks.shape[1]]
            encoded = self.temporal(hidden)
            return encoded[:, -1]

        def forward(
            self,
            self_track: torch.Tensor,  # (B, T, F)
            neighbours: torch.Tensor,  # (B, N, T, F)
            neighbour_mask: torch.Tensor,  # (B, N) True where a neighbour exists
        ) -> torch.Tensor:
            batch, n_neighbours = neighbours.shape[:2]

            own = self.encode(self_track)  # (B, D)

            if self.cfg.use_social and n_neighbours:
                flat = neighbours.reshape(batch * n_neighbours, *neighbours.shape[2:])
                encoded = self.encode(flat).reshape(batch, n_neighbours, -1)

                context = own.unsqueeze(1)
                # A row with no neighbours at all would make softmax undefined, so it is
                # allowed to attend to a padded slot and masked out afterwards instead.
                empty = ~neighbour_mask.any(dim=1)
                key_padding = ~neighbour_mask
                key_padding = key_padding & ~empty.unsqueeze(1)

                for attention, norm in zip(self.social_layers, self.social_norms, strict=True):
                    attended, _ = attention(context, encoded, encoded, key_padding_mask=key_padding)
                    context = norm(context + attended)

                context = context.squeeze(1)
                context = torch.where(empty.unsqueeze(1), torch.zeros_like(context), context)
            else:
                context = torch.zeros_like(own)

            pooled = torch.cat([own, context], dim=-1)
            trajectories = self.head(pooled).view(-1, self.cfg.n_modes, self.cfg.n_waypoints, 3)
            logits = self.mode_logits(pooled)
            return trajectories, logits

    return SocialTransformer(config)


def scene_tensors(
    obs: np.ndarray, config: TransformerConfig, wind: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build one model input per aircraft in a window.

    Returns `(self_track, neighbours, mask, frame)` with shapes
    (A, T, F), (A, N, T, F) and (A, N), plus the frame that inverts the transform.
    """
    obs = np.asarray(obs, dtype=float)
    n_agents = obs.shape[0]
    local, frame = scene_frames(obs)

    n_slots = config.max_neighbours
    feature_dim = config.n_features
    n_steps = obs.shape[1]

    self_track = np.zeros((n_agents, n_steps, feature_dim), dtype=np.float32)
    neighbours = np.zeros((n_agents, n_slots, n_steps, feature_dim), dtype=np.float32)
    mask = np.zeros((n_agents, n_slots), dtype=bool)

    for target in range(n_agents):
        seen = local[target]  # (A, T, 3)
        self_track[target] = _featurise(seen[target : target + 1], config, wind)[0]

        others = [j for j in range(n_agents) if j != target]
        if not others:
            continue

        # Nearest first, by horizontal distance at the last observed instant.
        distance = np.linalg.norm(seen[others, -1, :2], axis=-1)
        nearest = [others[k] for k in np.argsort(distance)[:n_slots]]

        chosen = _featurise(seen[nearest], config, wind)
        neighbours[target, : len(nearest)] = chosen
        mask[target, : len(nearest)] = True

    return self_track, neighbours, mask, frame


def _featurise(
    tracks: np.ndarray, config: TransformerConfig, wind: np.ndarray | None
) -> np.ndarray:
    """Positions, velocities and optional wind, scaled for the network."""
    feats = features(tracks, wind if config.use_wind else None)
    feats = feats.copy()
    feats[..., 0:3] /= config.position_scale
    feats[..., 3:6] /= config.velocity_scale
    return feats.astype(np.float32)


def make_samples(windows, config: TransformerConfig):
    """Flatten windows into per-aircraft samples: inputs, neighbours, mask, targets."""
    self_tracks, neighbours, masks, targets = [], [], [], []

    for window in windows:
        wind = window.wind if config.use_wind else None
        own, others, mask, _ = scene_tensors(window.obs, config, wind)

        # Targets are offsets in each aircraft's own frame: the diagonal of the scene frame
        # transform applied to the future.
        local_future = _future_in_own_frame(window)
        self_tracks.append(own)
        neighbours.append(others)
        masks.append(mask)
        targets.append((local_future / config.target_scale).astype(np.float32))

    if not self_tracks:
        empty_f = np.zeros((0, 11, config.n_features), dtype=np.float32)
        return (
            empty_f,
            np.zeros((0, config.max_neighbours, 11, config.n_features), dtype=np.float32),
            np.zeros((0, config.max_neighbours), dtype=bool),
            np.zeros((0, config.n_waypoints, 3), dtype=np.float32),
        )

    return (
        np.concatenate(self_tracks),
        np.concatenate(neighbours),
        np.concatenate(masks),
        np.concatenate(targets),
    )


def _future_in_own_frame(window) -> np.ndarray:
    """The prediction target, in each aircraft's own frame."""
    from pcas.models.framing import to_agent_frame

    _, future_local, _ = to_agent_frame(window.obs, window.future)
    return future_local


@dataclass
class TransformerPredictor(Predictor):
    """Wraps a trained module in the same interface as every other predictor."""

    module: object = None
    config: TransformerConfig = field(default_factory=TransformerConfig)
    waypoint_horizons: tuple[float, ...] = tuple(float(10 * k) for k in range(1, 13))
    device: str = "cpu"
    name: str = "transformer"
    _wind: np.ndarray | None = None
    # Probabilities of the hypotheses returned by the last predict() call, shape (A, K).
    last_probabilities: np.ndarray | None = None

    def with_wind(self, wind: np.ndarray) -> TransformerPredictor:
        self._wind = wind
        return self

    def predict(self, obs: np.ndarray, horizons_s: np.ndarray) -> np.ndarray:
        import torch

        wind = self._wind if self.config.use_wind else None
        if self.config.use_wind and wind is None:
            wind = np.zeros(2)

        own, others, mask, frame = scene_tensors(obs, self.config, wind)

        self.module.eval()
        with torch.no_grad():
            trajectories, logits = self.module(
                torch.from_numpy(own).to(self.device),
                torch.from_numpy(others).to(self.device),
                torch.from_numpy(mask).to(self.device),
            )
            local = trajectories.cpu().numpy() * self.config.target_scale
            self.last_probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        # (A, K, T, 3): to_world already understands a hypothesis dimension.
        world = to_world(local, frame)
        return self._resample_modes(world, horizons_s)

    def _resample_modes(self, waypoints: np.ndarray, horizons_s: np.ndarray) -> np.ndarray:
        """Interpolate (A, K, T, 3) waypoints onto the requested horizons.

        Alerting checks every second, so the 10 s waypoints must be filled in. Linear
        interpolation adds no information the model did not produce.
        """
        horizons = np.asarray(horizons_s, dtype=float)
        native = np.asarray(self.waypoint_horizons, dtype=float)
        if np.array_equal(horizons, native):
            return waypoints

        n_agents, n_modes = waypoints.shape[:2]
        out = np.empty((n_agents, n_modes, len(horizons), 3))
        for a in range(n_agents):
            for k in range(n_modes):
                for axis in range(3):
                    out[a, k, :, axis] = np.interp(horizons, native, waypoints[a, k, :, axis])
        return out


def load_predictor(path: str | Path, device: str | None = None) -> TransformerPredictor:
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    config = TransformerConfig(**blob["config"])

    module = build_module(config)
    _load_state(module, blob["state_dict"], config)
    module.to(device).eval()

    return TransformerPredictor(module=module, config=config, device=device)


def _load_state(module, state_dict, config: TransformerConfig) -> None:
    """Load weights, tolerating only the one incompatibility that is provably harmless.

    Checkpoints trained before multimodal output have no `mode_logits` head. For a
    single-mode model that head is inert: a softmax over one logit is 1.0 whatever the
    weights are, so leaving it at its initialisation cannot change a prediction. Any other
    missing or unexpected key means the checkpoint does not match the architecture, and
    loading it anyway would silently produce a differently-shaped model.
    """
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if not missing and not unexpected:
        return

    inert = config.n_modes == 1 and all(key.startswith("mode_logits.") for key in missing)
    if unexpected or not inert:
        raise RuntimeError(
            f"checkpoint does not match the architecture: missing {list(missing)}, "
            f"unexpected {list(unexpected)}"
        )
    if missing:
        log.info(
            "checkpoint predates multimodal output; its inert mode_logits head is left "
            "at initialisation (n_modes=1, so its softmax is 1.0 regardless)"
        )
