"""Fetch and parse the public VATSIM datafeed.

The feed is a full snapshot of every connected pilot and controller, refreshed roughly
every 15 seconds. Two things matter for PCAS:

1. Pilot position reports give trajectories (coarse - ~15 s between samples).
2. The controller list says which facilities are staffed at that instant, which is the
   natural experiment real ADS-B data cannot provide.

Personal data (the `name` field on each connection) is deliberately not stored; the
numeric CID is kept only so a single aircraft's samples can be stitched into a track.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import requests

log = logging.getLogger(__name__)

USER_AGENT = "pcas-research-collector/0.1 (portfolio research project)"

# Placeholder frequency used by observer connections - not a working controller.
OBSERVER_FREQUENCY = "199.998"

PILOT_FIELDS = (
    "cid",
    "callsign",
    "latitude",
    "longitude",
    "altitude",
    "groundspeed",
    "heading",
    "transponder",
    "qnh_i_hg",
    "logon_time",
    "last_updated",
)

CONTROLLER_FIELDS = (
    "cid",
    "callsign",
    "frequency",
    "facility",
    "rating",
    "visual_range",
    "logon_time",
    "last_updated",
)

FLIGHT_PLAN_FIELDS = {
    "aircraft_short": "aircraft",
    "departure": "departure",
    "arrival": "arrival",
    "flight_rules": "flight_rules",
}


def resolve_data_url(
    session: requests.Session, status_url: str, fallback: str, timeout: int
) -> str:
    """Pick a datafeed mirror from status.json, as VATSIM's docs ask consumers to do."""
    try:
        resp = session.get(status_url, timeout=timeout)
        resp.raise_for_status()
        urls = resp.json().get("data", {}).get("v3", [])
        if urls:
            return random.choice(urls)
    except (requests.RequestException, ValueError) as exc:
        log.warning("status.json lookup failed (%s); using fallback URL", exc)
    return fallback


def fetch_datafeed(session: requests.Session, url: str, timeout: int) -> dict[str, Any]:
    resp = session.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.json()


def snapshot_timestamp(feed: dict[str, Any]) -> str | None:
    """The feed's own generation time - used to skip snapshots we already stored."""
    return feed.get("general", {}).get("update_timestamp")


def _in_bbox(lat: Any, lon: Any, bbox: list[float]) -> bool:
    if lat is None or lon is None:
        return False
    lat_min, lon_min, lat_max, lon_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def parse_pilots(feed: dict[str, Any], bbox: list[float] | None = None) -> list[dict[str, Any]]:
    """Flatten the pilot list into rows, dropping the free-text name field."""
    snapshot_ts = snapshot_timestamp(feed)
    rows: list[dict[str, Any]] = []

    for pilot in feed.get("pilots", []):
        if bbox is not None and not _in_bbox(pilot.get("latitude"), pilot.get("longitude"), bbox):
            continue

        row: dict[str, Any] = {"snapshot_ts": snapshot_ts}
        row.update({key: pilot.get(key) for key in PILOT_FIELDS})

        plan = pilot.get("flight_plan") or {}
        for source, target in FLIGHT_PLAN_FIELDS.items():
            row[target] = plan.get(source)

        rows.append(row)

    return rows


def parse_controllers(feed: dict[str, Any]) -> list[dict[str, Any]]:
    """Controllers plus ATIS connections - both mean 'this field is not uncontrolled'.

    The feed's `controllers` list also contains observers (people watching, not working
    traffic), who show up on the placeholder frequency 199.998. They are kept but flagged,
    because an observer online does NOT make a field controlled - counting them would
    corrupt the controlled-vs-uncontrolled comparison.
    """
    snapshot_ts = snapshot_timestamp(feed)
    rows: list[dict[str, Any]] = []

    for kind in ("controllers", "atis"):
        for controller in feed.get(kind, []):
            row: dict[str, Any] = {"snapshot_ts": snapshot_ts, "kind": kind}
            row.update({key: controller.get(key) for key in CONTROLLER_FIELDS})
            row["is_observer"] = row.get("frequency") == OBSERVER_FREQUENCY
            rows.append(row)

    return rows
