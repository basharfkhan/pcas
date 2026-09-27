"""Load any trained checkpoint, whichever model produced it.

Evaluation scripts should not need to know which architecture they are scoring: everything
implements the same `Predictor` interface, so the only question is which class to rebuild.
Checkpoints record that in a `model` key; the earliest ones predate it and are LSTMs.
"""

from __future__ import annotations

from pathlib import Path

from pcas.models.baselines import Predictor


def load_predictor(path: str | Path, device: str | None = None) -> Predictor:
    import torch

    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    kind = blob.get("model", "lstm")

    if kind == "lstm":
        from pcas.models.lstm import load_predictor as load_lstm

        return load_lstm(path, device)

    if kind == "transformer":
        from pcas.models.transformer import load_predictor as load_transformer

        return load_transformer(path, device)

    raise ValueError(f"unknown model kind {kind!r} in {path}")
