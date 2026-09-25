"""Configuration loading for PCAS."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# VATSIM asks consumers not to poll faster than the feed updates (~15 s).
MIN_POLL_SECONDS = 15


@dataclass(frozen=True)
class CollectorConfig:
    """Settings for the VATSIM datafeed collector."""

    data_dir: Path
    poll_seconds: int = 15
    flush_minutes: int = 5
    request_timeout: int = 20
    # Optional [lat_min, lon_min, lat_max, lon_max] filter for pilot rows.
    bbox: list[float] | None = None
    # Controllers are kept regardless of bbox - their positions are not meaningful,
    # and which facilities are online is the whole point of collecting them.
    status_url: str = "https://status.vatsim.net/status.json"
    fallback_data_url: str = "https://data.vatsim.net/v3/vatsim-data.json"

    def __post_init__(self) -> None:
        if self.poll_seconds < MIN_POLL_SECONDS:
            raise ValueError(
                f"poll_seconds must be >= {MIN_POLL_SECONDS} to respect VATSIM's feed guidance"
            )
        if self.bbox is not None and len(self.bbox) != 4:
            raise ValueError("bbox must be [lat_min, lon_min, lat_max, lon_max]")


@dataclass(frozen=True)
class Config:
    collector: CollectorConfig
    tracked_airports: list[str] = field(default_factory=list)


def load_config(path: str | Path = "configs/collector.yaml") -> Config:
    """Load config from YAML. PCAS_DATA_DIR overrides the data directory."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    collector_raw = dict(raw.get("collector", {}))

    data_dir = os.environ.get("PCAS_DATA_DIR") or collector_raw.pop("data_dir")
    collector = CollectorConfig(data_dir=Path(data_dir).expanduser(), **collector_raw)
    return Config(collector=collector, tracked_airports=raw.get("tracked_airports", []))
