"""Read TrajAir scene files.

Format (from the dataset page): whitespace-separated, 1 Hz, one row per aircraft per
frame:

    Frame #, Aircraft ID, x (km), y (km), z (km), windx (m/s), windy (m/s)

Positions are already in an airport-centred inertial frame with x along the runway, so no
geodetic conversion is needed here - only km to metres, so that every dataset in this
project speaks metres.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

COLUMNS = ("frame", "agent_id", "x_km", "y_km", "z_km", "windx", "windy")

KM_TO_M = 1000.0

# Scene filenames carry the collection date. The exact convention is verified against the
# real download (see tests); these cover the plausible spellings.
_DATE_PATTERNS = (
    re.compile(r"(?P<y>20\d{2})[-_]?(?P<m>\d{2})[-_]?(?P<d>\d{2})"),
    re.compile(r"(?P<m>\d{2})[-_](?P<d>\d{2})[-_](?P<y>20\d{2})"),
)


@dataclass(frozen=True)
class Scene:
    """One scene file: a continuous stretch of traffic at the airport."""

    scene_id: str
    path: Path
    date: str | None
    frames: pd.DataFrame

    @property
    def n_agents(self) -> int:
        return int(self.frames["agent_id"].nunique())

    @property
    def duration_s(self) -> int:
        """Scene length in seconds (frames are 1 Hz)."""
        if self.frames.empty:
            return 0
        return int(self.frames["frame"].max() - self.frames["frame"].min() + 1)


def parse_scene_date(name: str) -> str | None:
    """Pull an ISO date out of a scene filename, or None if it carries none."""
    for pattern in _DATE_PATTERNS:
        match = pattern.search(name)
        if match:
            return f"{match['y']}-{match['m']}-{match['d']}"
    return None


def read_scene(path: str | Path) -> Scene:
    """Read one scene file into metres, sorted by frame then agent."""
    path = Path(path)
    frames = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        dtype={"frame": "int64", "agent_id": "string"},
    )

    for axis in ("x", "y", "z"):
        frames[f"{axis}_m"] = frames[f"{axis}_km"] * KM_TO_M
    frames = frames.drop(columns=["x_km", "y_km", "z_km"])

    frames = frames.sort_values(["frame", "agent_id"], ignore_index=True)

    return Scene(
        scene_id=path.stem,
        path=path,
        date=parse_scene_date(path.stem),
        frames=frames,
    )


def find_scenes(root: str | Path, split: str | None = None) -> list[Path]:
    """List scene files under a TrajAir subset.

    `split` picks the dataset's own train/test folders; None searches the whole tree.
    """
    root = Path(root)
    base = root / "processed_data" / split if split else root
    return sorted(p for p in base.rglob("*.txt") if p.name.lower() != "readme.txt")
