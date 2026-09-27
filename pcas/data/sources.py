"""One way to open either ADS-B dataset.

Scripts should not care whether a day came from TrajAir or TartanAviation: the readers
differ, the cleaned `RawDay` does not. This keeps the choice to a single flag so that every
baseline, model and alerting run can be pointed at either source and stay comparable.

A note on mixing them: TartanAviation's KBTP recordings are close to a superset of TrajAir.
98 of TrajAir's 109 days appear in it, so pooling the two would double-count those days and,
worse, could place the same day on both sides of a split. Pick one source per run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pcas.data.adsb import RawDay, read_raw_day
from pcas.data.subsets import raw_day_files
from pcas.data.tartan import find_sessions, read_session

KINDS = ("trajair", "tartan")


@dataclass
class Source:
    """A dataset, its usable days, and how to read one."""

    kind: str
    root: Path
    airport: str
    days: dict[str, list[Path]]
    _read: Callable[[list[Path], str], RawDay]

    def dates(self) -> list[str]:
        return sorted(self.days)

    def read(self, date: str) -> RawDay:
        return self._read(self.days[date], date)

    def split(self, test_days: int) -> tuple[list[str], list[str]]:
        """Chronological split: the final `test_days` are held out."""
        dates = self.dates()
        if test_days >= len(dates):
            raise ValueError(f"asked to hold out {test_days} of only {len(dates)} days")
        return dates[:-test_days], dates[-test_days:]


def open_source(kind: str, root: str | Path, airport: str = "kbtp") -> Source:
    root = Path(root)

    if kind == "trajair":
        days = {date: [path] for date, path in raw_day_files(root).items()}
        return Source(kind, root, airport, days, lambda paths, date: read_raw_day(paths[0]))

    if kind == "tartan":
        days = find_sessions(root, airport)
        return Source(
            kind,
            root,
            airport,
            days,
            lambda paths, date: read_session(paths, date=date, airport=airport, weather_root=root),
        )

    raise ValueError(f"unknown source {kind!r}; expected one of {KINDS}")
