"""Long-running VATSIM datafeed collector.

    python -m pcas.collect --config configs/collector.yaml

Runs until interrupted, flushing whatever is buffered on the way out. Safe to restart:
each flush writes its own file, so a crash costs at most one flush interval.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import UTC, datetime
from types import FrameType

import requests

from pcas.collect.vatsim import (
    fetch_datafeed,
    parse_controllers,
    parse_pilots,
    resolve_data_url,
    snapshot_timestamp,
)
from pcas.collect.writer import ParquetBuffer
from pcas.config import load_config

log = logging.getLogger("pcas.collect")

_stopping = False


def _handle_stop(signum: int, frame: FrameType | None) -> None:
    global _stopping
    _stopping = True
    log.info("signal %s received, finishing current cycle", signum)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect the VATSIM datafeed to Parquet.")
    parser.add_argument("--config", default="configs/collector.yaml")
    parser.add_argument("--once", action="store_true", help="Poll a single time and exit.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = load_config(args.config).collector
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    log.info("writing to %s, polling every %ds", cfg.data_dir, cfg.poll_seconds)

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    session = requests.Session()
    data_url = resolve_data_url(session, cfg.status_url, cfg.fallback_data_url, cfg.request_timeout)
    log.info("datafeed mirror: %s", data_url)

    pilots = ParquetBuffer(cfg.data_dir, "pilots")
    controllers = ParquetBuffer(cfg.data_dir, "controllers")

    last_snapshot: str | None = None
    last_flush = time.monotonic()
    flush_seconds = cfg.flush_minutes * 60
    consecutive_failures = 0

    while not _stopping:
        cycle_start = time.monotonic()

        try:
            feed = fetch_datafeed(session, data_url, cfg.request_timeout)
            consecutive_failures = 0

            snapshot = snapshot_timestamp(feed)
            if snapshot is not None and snapshot == last_snapshot:
                log.debug("snapshot %s unchanged, skipping", snapshot)
            else:
                last_snapshot = snapshot
                pilot_rows = parse_pilots(feed, cfg.bbox)
                pilots.extend(pilot_rows)
                controllers.extend(parse_controllers(feed))
                log.debug("snapshot %s: %d pilots buffered", snapshot, len(pilot_rows))

        except (requests.RequestException, ValueError) as exc:
            consecutive_failures += 1
            log.warning("fetch failed (%d in a row): %s", consecutive_failures, exc)
            # Back off, and re-resolve the mirror in case this one is unhealthy.
            if consecutive_failures >= 3:
                data_url = resolve_data_url(
                    session, cfg.status_url, cfg.fallback_data_url, cfg.request_timeout
                )
                log.info("switched datafeed mirror: %s", data_url)
            time.sleep(min(60, cfg.poll_seconds * consecutive_failures))

        now = time.monotonic()
        if now - last_flush >= flush_seconds:
            stamp = datetime.now(UTC)
            pilots.flush(stamp)
            controllers.flush(stamp)
            last_flush = now

        if args.once:
            break

        elapsed = time.monotonic() - cycle_start
        time.sleep(max(0.0, cfg.poll_seconds - elapsed))

    stamp = datetime.now(UTC)
    pilots.flush(stamp)
    controllers.flush(stamp)
    log.info("collector stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
