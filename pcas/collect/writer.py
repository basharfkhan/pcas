"""Buffered Parquet writer.

The feed is polled every ~15 s, which would be ~5,700 tiny files per dataset per day if
each snapshot were written straight to disk. Rows are buffered in memory and flushed
every few minutes into date-partitioned Parquet instead.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


class ParquetBuffer:
    """Accumulates rows for one dataset and flushes them to date-partitioned Parquet."""

    def __init__(self, root: Path, name: str, compression: str = "zstd") -> None:
        self.root = Path(root)
        self.name = name
        self.compression = compression
        self._rows: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def extend(self, rows: list[dict[str, Any]]) -> None:
        self._rows.extend(rows)

    def flush(self, now: datetime | None = None) -> Path | None:
        """Write buffered rows to a single Parquet file. Returns the path, or None if empty."""
        if not self._rows:
            return None

        now = now or datetime.now(UTC)
        partition = self.root / self.name / f"date={now:%Y-%m-%d}"
        partition.mkdir(parents=True, exist_ok=True)
        path = partition / f"{self.name}-{now:%Y%m%dT%H%M%S}.parquet"

        frame = pd.DataFrame(self._rows)
        frame.to_parquet(path, compression=self.compression, index=False)

        log.info("wrote %d rows to %s", len(frame), path)
        self._rows.clear()
        return path
