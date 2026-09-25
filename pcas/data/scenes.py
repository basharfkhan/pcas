"""Turn scene files into training windows.

A window is a slice of time containing *every aircraft airborne during it* - that is what
makes the problem multi-agent. Defaults follow TrajAirNet so results stay comparable:
11 s observed, 120 s predicted, sampled every 10 s (12 future waypoints).

Two rules keep the data honest:

1. A window must be frame-contiguous. Gaps in the feed would otherwise be interpolated
   over silently, teaching the model motion that never happened.
2. An aircraft joins a window only if it is present for *every* frame in it. Padding
   absent aircraft with zeros would put phantom traffic at the runway threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OBS_LEN = 11
PRED_LEN = 120
PRED_STEP = 10


@dataclass(frozen=True)
class Window:
    """One multi-agent prediction problem.

    obs:     (n_agents, obs_len, 3)   positions in metres, x along runway
    future:  (n_agents, n_waypoints, 3)
    wind:    (2,)                     mean windx/windy over the observed span, m/s
    """

    scene_id: str
    date: str | None
    start_frame: int
    agent_ids: tuple[str, ...]
    obs: np.ndarray
    future: np.ndarray
    wind: np.ndarray

    @property
    def n_agents(self) -> int:
        return len(self.agent_ids)


def future_offsets(obs_len: int = OBS_LEN, pred_len: int = PRED_LEN, pred_step: int = PRED_STEP):
    """Frame offsets of the predicted waypoints, relative to the window start.

    The last observed frame is at offset obs_len - 1; waypoints march out from there.
    """
    if pred_len % pred_step:
        raise ValueError("pred_len must be a whole number of pred_step intervals")
    last_obs = obs_len - 1
    return [last_obs + k * pred_step for k in range(1, pred_len // pred_step + 1)]


def build_windows(
    scene,
    obs_len: int = OBS_LEN,
    pred_len: int = PRED_LEN,
    pred_step: int = PRED_STEP,
    stride: int = 10,
    min_agents: int = 1,
) -> list[Window]:
    """Slide a window over one scene and emit every usable multi-agent problem."""
    frames = scene.frames
    if frames.empty:
        return []

    offsets = future_offsets(obs_len, pred_len, pred_step)
    span = obs_len + pred_len  # frames the window must cover
    frame_ids = np.sort(frames["frame"].unique())
    if len(frame_ids) < span:
        return []

    positions = {
        (int(row.frame), row.agent_id): (row.x_m, row.y_m, row.z_m)
        for row in frames.itertuples(index=False)
    }
    wind_by_frame = (
        frames.groupby("frame")[["windx", "windy"]].mean().astype("float32").to_dict("index")
    )
    agents_by_frame = frames.groupby("frame")["agent_id"].apply(set).to_dict()

    windows: list[Window] = []

    for i in range(0, len(frame_ids) - span + 1, stride):
        start = int(frame_ids[i])
        wanted = np.arange(start, start + span)
        # Rule 1: contiguous frames only.
        if not np.array_equal(frame_ids[i : i + span], wanted):
            continue

        obs_frames = wanted[:obs_len]
        future_frames = [start + off for off in offsets]

        # Rule 2: aircraft present for the whole window.
        present: set[str] | None = None
        for frame in (*obs_frames, *future_frames):
            here = agents_by_frame.get(frame, set())
            present = set(here) if present is None else present & here
            if not present:
                break
        if not present or len(present) < min_agents:
            continue

        agent_ids = tuple(sorted(present))
        obs = np.array(
            [[positions[(f, a)] for f in obs_frames] for a in agent_ids], dtype=np.float32
        )
        future = np.array(
            [[positions[(f, a)] for f in future_frames] for a in agent_ids], dtype=np.float32
        )
        wind = np.array(
            [
                np.mean([wind_by_frame[f]["windx"] for f in obs_frames]),
                np.mean([wind_by_frame[f]["windy"] for f in obs_frames]),
            ],
            dtype=np.float32,
        )

        windows.append(
            Window(
                scene_id=scene.scene_id,
                date=scene.date,
                start_frame=start,
                agent_ids=agent_ids,
                obs=obs,
                future=future,
                wind=wind,
            )
        )

    return windows


def windows_to_frame(windows: list[Window]) -> pd.DataFrame:
    """Flat summary of a window list, for EDA and sanity checks."""
    return pd.DataFrame(
        [
            {
                "scene_id": w.scene_id,
                "date": w.date,
                "start_frame": w.start_frame,
                "n_agents": w.n_agents,
                "wind_x": float(w.wind[0]),
                "wind_y": float(w.wind[1]),
            }
            for w in windows
        ]
    )
