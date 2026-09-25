"""Scene reading, windowing and splits, on synthetic scene files.

The fixtures are written in TrajAir's format (frame, id, x/y/z in km, wind in m/s at 1 Hz)
so these tests pin down the pipeline's behaviour before the real 420 MB download.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pcas.data.scenes import build_windows, future_offsets, windows_to_frame
from pcas.data.splits import (
    MissingSceneDates,
    chronological_split,
    day_split,
    describe_split,
)
from pcas.data.trajair import find_scenes, parse_scene_date, read_scene


def write_scene(path: Path, n_frames: int, agents=("A1", "A2"), skip_frames=(), drop=None) -> Path:
    """Write a synthetic scene: each agent flies a straight line at 1 Hz.

    `skip_frames` punches a gap in the whole scene; `drop` maps agent -> frames where that
    one agent is missing.
    """
    drop = drop or {}
    lines = []
    for frame in range(n_frames):
        if frame in skip_frames:
            continue
        for i, agent in enumerate(agents):
            if frame in drop.get(agent, ()):
                continue
            x = 0.05 * frame + i  # km
            y = 0.02 * frame - i
            z = 0.3 + 0.001 * frame
            lines.append(f"{frame} {agent} {x:.4f} {y:.4f} {z:.4f} 3.5 -1.5")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def scene(tmp_path):
    return read_scene(write_scene(tmp_path / "2020-09-18_scene1.txt", n_frames=200))


def test_read_scene_converts_km_to_metres(scene):
    row = scene.frames.iloc[0]
    assert row["x_m"] == pytest.approx(0.0)
    assert scene.frames.loc[scene.frames["frame"] == 100, "x_m"].iloc[0] == pytest.approx(5000.0)
    # The km columns are gone, so nothing downstream can mix units.
    assert not [c for c in scene.frames.columns if c.endswith("_km")]


def test_read_scene_metadata(scene):
    assert scene.n_agents == 2
    assert scene.duration_s == 200
    assert scene.date == "2020-09-18"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("2020-09-18_scene1", "2020-09-18"),
        ("20200918_7", "2020-09-18"),
        ("09_18_2020_scene", "2020-09-18"),
        ("scene_42", None),
    ],
)
def test_parse_scene_date(name, expected):
    assert parse_scene_date(name) == expected


def test_future_offsets_follow_trajairnet_defaults():
    offsets = future_offsets(obs_len=11, pred_len=120, pred_step=10)
    assert len(offsets) == 12
    assert offsets[0] == 20  # last observed frame (10) + 10 s
    assert offsets[-1] == 130  # 120 s past the last observed frame


def test_future_offsets_rejects_ragged_horizon():
    with pytest.raises(ValueError, match="whole number"):
        future_offsets(pred_len=125, pred_step=10)


def test_build_windows_shapes(scene):
    windows = build_windows(scene, stride=10)
    assert windows

    w = windows[0]
    assert w.obs.shape == (2, 11, 3)
    assert w.future.shape == (2, 12, 3)
    assert w.wind.tolist() == pytest.approx([3.5, -1.5])
    assert w.agent_ids == ("A1", "A2")
    assert w.date == "2020-09-18"


def test_window_future_starts_after_last_observed_frame(scene):
    w = build_windows(scene, stride=10)[0]
    # Synthetic motion is 50 m/s along x, so the first waypoint is 10 s past frame 10.
    assert float(w.obs[0, -1, 0]) == pytest.approx(500.0, abs=1e-3)
    assert float(w.future[0, 0, 0]) == pytest.approx(1000.0, abs=1e-3)
    assert float(w.future[0, -1, 0]) == pytest.approx(6500.0, abs=1e-3)


def test_windows_do_not_span_a_feed_gap(tmp_path):
    # A gap at frame 60 must not be bridged: no window may contain it.
    scene = read_scene(write_scene(tmp_path / "2020-09-19_gap.txt", n_frames=400, skip_frames={60}))
    windows = build_windows(scene, stride=1)
    assert windows
    for w in windows:
        assert not (w.start_frame <= 60 < w.start_frame + 131)


def test_agent_missing_part_of_the_window_is_excluded(tmp_path):
    # A2 disappears for a single frame inside the first window's span.
    scene = read_scene(
        write_scene(tmp_path / "2020-09-20_drop.txt", n_frames=200, drop={"A2": {5}})
    )
    first = build_windows(scene, stride=131)[0]
    assert first.agent_ids == ("A1",)
    assert first.obs.shape[0] == 1


def test_scene_too_short_yields_no_windows(tmp_path):
    scene = read_scene(write_scene(tmp_path / "2020-09-21_short.txt", n_frames=100))
    assert build_windows(scene) == []


def test_min_agents_filters_single_aircraft_windows(tmp_path):
    scene = read_scene(write_scene(tmp_path / "2020-09-22_solo.txt", n_frames=200, agents=("A1",)))
    assert build_windows(scene, min_agents=1)
    assert build_windows(scene, min_agents=2) == []


def test_windows_to_frame_summary(scene):
    frame = windows_to_frame(build_windows(scene, stride=10))
    assert set(frame.columns) == {"scene_id", "date", "start_frame", "n_agents", "wind_x", "wind_y"}
    assert (frame["n_agents"] == 2).all()


# --- splits -------------------------------------------------------------------


@pytest.fixture
def dated_paths(tmp_path):
    paths = []
    for day in range(18, 28):
        for scene_no in range(3):
            paths.append(tmp_path / f"2020-09-{day:02d}_scene{scene_no}.txt")
    return paths


def test_day_split_never_shares_a_day(dated_paths):
    split = day_split(dated_paths, test_frac=0.3, seed=0)
    summary = describe_split(split)

    assert summary["overlap"]["days"] == 0
    assert summary["test"]["days"] == 3
    assert summary["train"]["days"] == 7
    assert summary["train"]["scenes"] + summary["test"]["scenes"] == len(dated_paths)


def test_day_split_is_deterministic_for_a_seed(dated_paths):
    assert day_split(dated_paths, seed=7) == day_split(dated_paths, seed=7)
    assert day_split(dated_paths, seed=7) != day_split(dated_paths, seed=8)


def test_chronological_split_holds_out_the_last_days(dated_paths):
    split = chronological_split(dated_paths, test_frac=0.3)
    test_days = {parse_scene_date(p.stem) for p in split["test"]}
    train_days = {parse_scene_date(p.stem) for p in split["train"]}

    assert test_days == {"2020-09-25", "2020-09-26", "2020-09-27"}
    assert max(train_days) < min(test_days)


def test_undated_filenames_refuse_to_split(tmp_path):
    with pytest.raises(MissingSceneDates, match="no parsable date"):
        day_split([tmp_path / "scene_1.txt", tmp_path / "2020-09-18_a.txt"])


def test_find_scenes_reads_official_split_folders(tmp_path):
    write_scene(tmp_path / "processed_data" / "train" / "2020-09-18_a.txt", n_frames=20)
    write_scene(tmp_path / "processed_data" / "test" / "2020-09-19_b.txt", n_frames=20)
    (tmp_path / "processed_data" / "README.txt").write_text("ignore me", encoding="utf-8")

    assert [p.name for p in find_scenes(tmp_path, "train")] == ["2020-09-18_a.txt"]
    assert [p.name for p in find_scenes(tmp_path, "test")] == ["2020-09-19_b.txt"]
    # README.txt must not be mistaken for a scene.
    assert len(find_scenes(tmp_path)) == 2
