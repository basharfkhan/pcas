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

    obs:          (n_agents, obs_len, 3)      positions in metres, x along runway
    future:       (n_agents, n_waypoints, 3)  the prediction target, at pred_step spacing
    future_dense: (n_agents, pred_len, 3)     every second of the same span
    wind:         (2,)                        mean windx/windy over the observed span, m/s

    `future` is what models predict; `future_dense` is what conflicts are labelled on. A
    10 s sampling can step straight over a close approach, so the label needs 1 Hz.
    """

    scene_id: str
    date: str | None
    start_frame: int
    agent_ids: tuple[str, ...]
    obs: np.ndarray
    future: np.ndarray
    wind: np.ndarray
    future_dense: np.ndarray | None = None

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
    min_motion_m: float = 50.0,
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

        # Rule 2: aircraft present at EVERY second of the window, not merely at the
        # sampled waypoints. Conflict labelling reads all of them, and an aircraft that
        # vanishes mid-window has no separation to measure there.
        present: set[str] | None = None
        for frame in wanted:
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
        # Every second of the predicted span, for conflict labelling.
        dense_frames = wanted[obs_len:]
        future_dense = np.array(
            [[positions[(f, a)] for f in dense_frames] for a in agent_ids], dtype=np.float32
        )

        # Drop aircraft that barely move across this window. The track-level filter in
        # adsb.py cannot catch a transponder that freezes for part of an otherwise moving
        # track, and a motionless target is trivially predictable: it would flatter every
        # model and every baseline equally, which just adds noise to the comparison.
        if min_motion_m > 0:
            path = np.concatenate([obs, future], axis=1)
            moved = np.linalg.norm(np.diff(path, axis=1), axis=2).sum(axis=1)
            keep = moved >= min_motion_m
            if not keep.any() or keep.sum() < min_agents:
                continue
            if not keep.all():
                agent_ids = tuple(a for a, k in zip(agent_ids, keep, strict=True) if k)
                obs, future, future_dense = obs[keep], future[keep], future_dense[keep]
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
                future_dense=future_dense,
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
