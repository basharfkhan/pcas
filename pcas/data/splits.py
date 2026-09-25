"""Train/test splits, and why there is more than one.

TrajAir ships a random 70/30 split over scene files. Published numbers use it, so PCAS
reports it too - but it leaks: overlapping windows cut from the same day (same aircraft,
same traffic pattern, often the same flight) can land on both sides, so a model gets
credit for recalling traffic it already saw.

`day_split` assigns whole days to one side. `chronological_split` goes further and holds
out the *last* days, which is the honest question for a deployed system: does it work on
traffic it has never seen, later in time?

Reporting the official split alongside a day-based one - and the gap between them - is a
result in itself.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

from pcas.data.trajair import parse_scene_date


class MissingSceneDates(ValueError):
    """Raised when filenames carry no date, so a leak-free split cannot be built."""


def _dates_for(paths: Iterable[Path]) -> dict[Path, str]:
    paths = list(paths)
    dated = {p: parse_scene_date(p.stem) for p in paths}
    undated = [p.name for p, d in dated.items() if d is None]
    if undated:
        raise MissingSceneDates(
            f"{len(undated)} of {len(paths)} scene files have no parsable date "
            f"(e.g. {undated[:3]}); add the real filename convention to "
            "pcas.data.trajair._DATE_PATTERNS"
        )
    return dated  # type: ignore[return-value]


def official_split(root: str | Path) -> dict[str, list[Path]]:
    """The dataset's own train/test folders. Comparable to published results; leaky."""
    from pcas.data.trajair import find_scenes

    return {"train": find_scenes(root, "train"), "test": find_scenes(root, "test")}


def day_split(
    paths: Iterable[Path], test_frac: float = 0.3, seed: int = 0
) -> dict[str, list[Path]]:
    """Hold out whole randomly chosen days, so no day appears on both sides."""
    dated = _dates_for(paths)
    days = sorted(set(dated.values()))

    rng = random.Random(seed)
    shuffled = days[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(len(days) * test_frac))
    test_days = set(shuffled[:n_test])

    return {
        "train": sorted(p for p, d in dated.items() if d not in test_days),
        "test": sorted(p for p, d in dated.items() if d in test_days),
    }


def chronological_split(paths: Iterable[Path], test_frac: float = 0.3) -> dict[str, list[Path]]:
    """Hold out the final days - the strictest, and the closest to deployment."""
    dated = _dates_for(paths)
    days = sorted(set(dated.values()))

    n_test = max(1, round(len(days) * test_frac))
    test_days = set(days[-n_test:])

    return {
        "train": sorted(p for p, d in dated.items() if d not in test_days),
        "test": sorted(p for p, d in dated.items() if d in test_days),
    }


def describe_split(split: dict[str, list[Path]]) -> dict[str, dict[str, int]]:
    """Scene and day counts per side, plus any day appearing on both (should be zero)."""
    summary: dict[str, dict[str, int]] = {}
    day_sets: dict[str, set[str]] = {}

    for name, paths in split.items():
        days = {d for d in (parse_scene_date(p.stem) for p in paths) if d is not None}
        day_sets[name] = days
        summary[name] = {"scenes": len(paths), "days": len(days)}

    if "train" in day_sets and "test" in day_sets:
        summary["overlap"] = {"days": len(day_sets["train"] & day_sets["test"])}

    return summary
