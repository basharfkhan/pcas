"""Download TartanAviation ADS-B data.

TartanAviation is TrajAir's successor from the same lab: 661 days of ADS-B across two
airports rather than 111 at one, and crucially the second airport (KAGC) is **towered**.
That gives a towered-vs-non-towered comparison on real aircraft, where the VATSIM feed can
only offer simulated traffic, and it gives a second field to test whether a model learned
traffic behaviour or merely memorised KBTP's geometry.

The data sits in a public, S3-compatible bucket at CMU. Their own script pulls it with
boto3/minio; plain HTTPS works just as well and spares the dependency. Listing the bucket
is not permitted, so the object paths are enumerated exactly as their `download.py` does.

Raw layout matches TrajAir closely (`kbtp/raw/2022/11-02-20/1.csv`), with an extra year
level and no `_adsb` folder suffix.

    python -m pcas.data.download_tartan --location both --dest data/tartan

CC BY 4.0: cite Patrikar et al., "TartanAviation" (2024).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import zipfile
from pathlib import Path

import requests

log = logging.getLogger("pcas.data.download_tartan")

BUCKET = (
    "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/"
    "AUTH_ac8533a83cff4d48bc8c608ad222d330/tartanaviation-adsb"
)
CHUNK = 1 << 20

# Enumerated rather than listed: the container denies listing. KBTP has an extra year.
RAW_OBJECTS = {
    "kbtp": ("kbtp/raw/2020", "kbtp/raw/2021", "kbtp/raw/2022"),
    "kagc": ("kagc/raw/2021", "kagc/raw/2022"),
}


ATTEMPTS = 5


def _with_retry(call, what: str):
    """Retry around a flaky endpoint: this host drops connections mid-transfer."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return call()
        except (requests.RequestException, OSError) as exc:
            if attempt == ATTEMPTS:
                raise
            wait = min(30, 2**attempt)
            log.warning(
                "%s failed (%s), retrying in %ds [%d/%d]", what, exc, wait, attempt, ATTEMPTS
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def remote_size(session: requests.Session, object_path: str) -> int | None:
    def head():
        resp = session.head(f"{BUCKET}/{object_path}.zip", timeout=30)
        if resp.status_code != 200:
            return None
        length = resp.headers.get("content-length")
        return int(length) if length else None

    return _with_retry(head, f"HEAD {object_path}")


def download(session: requests.Session, object_path: str, dest: Path) -> Path:
    """Stream one object to disk, skipping it when a complete copy is already present.

    No checksums are published for these objects, so completeness is judged by byte count
    against the server's content-length. That catches the common failure (a truncated
    download) without pretending to verify integrity.
    """
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"{object_path.replace('/', '_')}.zip"
    expected = remote_size(session, object_path)
    if expected is None:
        raise OSError(f"{object_path}: not found in bucket")

    if target.exists() and target.stat().st_size == expected:
        log.info("%s already complete (%.2f GB)", target.name, expected / 1e9)
        return target

    log.info("downloading %s (%.2f GB)", object_path, expected / 1e9)

    def fetch() -> int:
        written = 0
        with session.get(f"{BUCKET}/{object_path}.zip", stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with target.open("wb") as handle:
                for block in resp.iter_content(CHUNK):
                    handle.write(block)
                    written += len(block)
                    if written % (100 * CHUNK) < CHUNK:
                        log.info("  %.1f/%.1f GB", written / 1e9, expected / 1e9)
        if written != expected:
            # Truncated: raise so the retry wrapper starts the object over.
            raise OSError(f"{object_path}: got {written} bytes, expected {expected}")
        return written

    _with_retry(fetch, f"GET {object_path}")
    return target


def _safe_target(out: Path, member: str) -> Path:
    target = (out / member).resolve()
    if not target.is_relative_to(out.resolve()):
        raise OSError(f"refusing path traversal entry {member!r}")
    return target


def _extract_deflate64(archive: Path, out: Path) -> None:
    """Extract an archive using Deflate64, which the standard library cannot read.

    Some of these archives use compression method 9 (Deflate64, a WinZip extension).
    Python's zipfile raises NotImplementedError on it, and - worse - bsdtar writes files
    of the correct SIZE filled with padding instead of failing, which looks like a
    successful extraction until you read the contents. stream-unzip implements the method
    properly, in pure Python, so it is slow but correct.
    """
    from stream_unzip import stream_unzip

    def blocks():
        with archive.open("rb") as handle:
            while chunk := handle.read(CHUNK):
                yield chunk

    written_bytes, files = 0, 0
    for name, _size, chunks in stream_unzip(blocks()):
        member = name.decode("utf-8", "replace")
        if member.endswith("/"):
            continue
        target = _safe_target(out, member)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            for chunk in chunks:
                handle.write(chunk)
                written_bytes += len(chunk)
        files += 1
        if files % 25 == 0:
            log.info("  %d files, %.2f GB written", files, written_bytes / 1e9)

    log.info("  %d files, %.2f GB written", files, written_bytes / 1e9)


def extract(archive: Path, dest: Path) -> Path:
    """Extract, refusing any entry that would escape the destination."""
    out = dest / archive.stem
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            _safe_target(out, member)
        needs_deflate64 = any(info.compress_type == 9 for info in zf.infolist())
        if not needs_deflate64:
            zf.extractall(out)
            log.info("extracted %s", archive.name)
            return out

    log.info("%s uses Deflate64; extracting with stream-unzip (slower)", archive.name)
    _extract_deflate64(archive, out)
    log.info("extracted %s", archive.name)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download TartanAviation raw ADS-B.")
    parser.add_argument("--location", choices=["kbtp", "kagc", "both"], default="both")
    parser.add_argument("--dest", default="data/tartan")
    parser.add_argument("--sizes-only", action="store_true", help="Report sizes and exit.")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    locations = list(RAW_OBJECTS) if args.location == "both" else [args.location]
    objects = [path for loc in locations for path in RAW_OBJECTS[loc]]
    session = requests.Session()

    if args.sizes_only:
        total = 0
        for path in objects:
            size = remote_size(session, path)
            print(f"{path + '.zip':<24} {(size or 0) / 1e9:7.2f} GB")
            total += size or 0
        print(f"{'TOTAL':<24} {total / 1e9:7.2f} GB compressed")
        return 0

    dest = Path(args.dest)
    for path in objects:
        archive = download(session, path, dest)
        extract(archive, dest)
        if not args.keep_zip:
            archive.unlink(missing_ok=True)

    log.info("done: %s", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
