"""Locating the days in a TrajAir subset, without reading them all.

The 111-day subset holds ~2 GB of raw CSVs. Building every window and then keeping the
held-out days would need gigabytes of RAM, so the split is decided from the folder names
(`raw_data/09-18-20_adsb`) and only the wanted days are read.
"""

from __future__ import annotations

import re
from pathlib import Path

_FOLDER_DATE = re.compile(r"^(?P<m>\d{2})-(?P<d>\d{2})-(?P<y>\d{2})")


def raw_day_files(subset: str | Path) -> dict[str, Path]:
    """ISO date -> raw CSV path, for every day in the subset."""
    root = Path(subset) / "raw_data"
    days: dict[str, Path] = {}

    for folder in sorted(root.iterdir()) if root.exists() else []:
        if not folder.is_dir():
            continue
        match = _FOLDER_DATE.match(folder.name)
        if not match:
            continue
        csvs = sorted(folder.glob("*.csv"))
        if not csvs:
            continue
        days[f"20{match['y']}-{match['m']}-{match['d']}"] = csvs[0]

    return dict(sorted(days.items()))


def chronological_days(subset: str | Path, test_days: int) -> tuple[list[str], list[str]]:
    """Split the subset's dates into (train, test), holding out the final `test_days`."""
    dates = list(raw_day_files(subset))
    if test_days >= len(dates):
        raise ValueError(f"asked to hold out {test_days} of only {len(dates)} days")
    return dates[:-test_days], dates[-test_days:]
