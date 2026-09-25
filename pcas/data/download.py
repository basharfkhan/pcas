"""Download TrajAir from Figshare.

TrajAirNet's README suggests `wget .../articles/14866251/versions/1`, which asks Figshare
to build a zip of *everything* asynchronously. The article API lists each file with its
size and MD5 instead, so this fetches only the subsets asked for and verifies them.

    python -m pcas.data.download --subset 7days1 --dest data/trajair

TrajAir is CC BY 4.0: cite Patrikar et al., "Predicting Like A Pilot" (2021).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import zipfile
from pathlib import Path

import requests

log = logging.getLogger("pcas.data.download")

ARTICLE_ID = 14866251
ARTICLE_API = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
CHUNK = 1 << 20

SUBSETS = ("7days1", "7days2", "7days3", "7days4", "111_days", "weather_data")


def list_files(session: requests.Session) -> dict[str, dict]:
    """Map filename -> Figshare file record (download_url, size, md5)."""
    resp = session.get(ARTICLE_API, timeout=30)
    resp.raise_for_status()
    return {f["name"]: f for f in resp.json()["files"]}


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - Figshare publishes MD5, not our choice
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(session: requests.Session, record: dict, dest: Path) -> Path:
    """Download one Figshare file, skipping it when a verified copy is already present."""
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / record["name"]
    expected = record.get("computed_md5") or record.get("supplied_md5")

    if target.exists() and expected and _md5(target) == expected:
        log.info("%s already present and verified", target.name)
        return target

    size_mb = record["size"] / 1e6
    log.info("downloading %s (%.1f MB)", record["name"], size_mb)

    with session.get(record["download_url"], stream=True, timeout=60) as resp:
        resp.raise_for_status()
        written = 0
        with target.open("wb") as handle:
            for block in resp.iter_content(CHUNK):
                handle.write(block)
                written += len(block)
                if written % (50 * CHUNK) < CHUNK:
                    log.info("  %.0f/%.0f MB", written / 1e6, size_mb)

    if expected:
        actual = _md5(target)
        if actual != expected:
            target.unlink(missing_ok=True)
            raise OSError(f"{record['name']}: MD5 mismatch (expected {expected}, got {actual})")
        log.info("%s verified", target.name)

    return target


def extract(archive: Path, dest: Path) -> Path:
    """Extract a subset zip, refusing any entry that escapes the destination."""
    out = dest / archive.stem
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            resolved = (out / member).resolve()
            if not resolved.is_relative_to(out.resolve()):
                raise OSError(f"{archive.name}: refusing path traversal entry {member!r}")
        zf.extractall(out)

    log.info("extracted %s -> %s", archive.name, out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the TrajAir dataset.")
    parser.add_argument("--subset", action="append", choices=SUBSETS, help="Repeatable.")
    parser.add_argument("--dest", default="data/trajair")
    parser.add_argument("--list", action="store_true", help="List available files and exit.")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    session = requests.Session()
    available = list_files(session)

    if args.list or not args.subset:
        total = sum(f["size"] for f in available.values())
        for name, rec in sorted(available.items()):
            print(f"{name:<24} {rec['size'] / 1e6:>8.1f} MB")
        print(f"{'TOTAL':<24} {total / 1e9:>8.2f} GB")
        if not args.subset:
            print("\nNothing downloaded: pass --subset (e.g. --subset 7days1).")
        return 0

    dest = Path(args.dest)
    for subset in args.subset:
        name = f"{subset}.zip"
        if name not in available:
            log.error("%s not found in article %s", name, ARTICLE_ID)
            return 1
        archive = download_file(session, available[name], dest)
        extract(archive, dest)
        if not args.keep_zip:
            archive.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
