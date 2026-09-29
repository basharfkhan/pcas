"""Controlled against uncontrolled: close convergences at fields with and without a tower.

Three passes over the collected feed, because the second one needs the first one's answer:

1. Stationary aircraft give every field's position and elevation (`estimate_field_positions`).
2. Those volumes are then used to pull out near-field traffic, which is a tiny fraction of
   the 21M position reports and the only part worth keeping in memory.
3. Controller callsigns say which of those fields were staffed at each instant.

The output is a per-(field, snapshot) panel cached under `artifacts/staffing/`, so the
comparison can be re-cut later without re-reading the parquet files.

    python scripts/staffing_analysis.py --data-dir C:/Users/basha/pcas-data

Add `--reuse` to skip straight to the comparison using the cached panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pcas.analysis.staffing import (
    CRITERIA,
    build_panel,
    crude_rates,
    estimate_field_positions,
    first_stationary_samples,
    near_field_traffic,
    parse_controller_positions,
    placebo_ratios,
    staffing_by_snapshot,
    standardised_comparison,
    stratified_rates,
)
from pcas.collect.vatsim import OBSERVER_FREQUENCY

PILOT_COLUMNS = [
    "snapshot_ts",
    "cid",
    "callsign",
    "latitude",
    "longitude",
    "altitude",
    "groundspeed",
    "logon_time",
    "departure",
]

# Cruise traffic cannot be in a field's pattern, and reading it costs the same as reading
# the traffic that matters. This cap is applied before anything is kept.
PREFILTER_ALT_FT = 20_000.0


def pilot_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("pilots/date=*/*.parquet"))


def controller_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("controllers/date=*/*.parquet"))


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read the columns a file actually has; the collector's schema gained fields over time."""
    available = set(pq.ParquetFile(path).schema.names)
    return pd.read_parquet(path, columns=[c for c in columns if c in available])


def collect_field_positions(files: list[Path], min_samples: int) -> pd.DataFrame:
    parked = []
    for index, path in enumerate(files, start=1):
        parked.append(first_stationary_samples(_read(path, PILOT_COLUMNS)))
        if index % 200 == 0:
            print(f"  parked pass {index}/{len(files)}")
    if not parked:
        return pd.DataFrame()

    pooled = pd.concat(parked, ignore_index=True).sort_values("snapshot_ts")
    pooled = pooled.drop_duplicates(subset=["cid", "callsign", "logon_time"], keep="first")
    return estimate_field_positions(pooled, min_samples=min_samples)


def collect_traffic(files: list[Path], fields: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    for index, path in enumerate(files, start=1):
        frame = _read(path, PILOT_COLUMNS)
        chunks.append(near_field_traffic(frame[frame["altitude"] < PREFILTER_ALT_FT], fields))
        if index % 200 == 0:
            print(f"  traffic pass {index}/{len(files)}")
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def collect_staffing(files: list[Path], fields: pd.Series) -> pd.DataFrame:
    columns = ["snapshot_ts", "callsign", "frequency", "is_observer"]
    frames = []
    for path in files:
        frame = _read(path, columns).reindex(columns=columns)
        # An observer online does not make a field controlled, and rows written before the
        # collector had the flag are re-derived from the frequency exactly as it does.
        flagged = frame["is_observer"]
        backfill = frame["frequency"] == OBSERVER_FREQUENCY
        is_observer = flagged.where(flagged.notna(), backfill).fillna(False).astype(bool)
        frames.append(frame[~is_observer])
    if not frames:
        return pd.DataFrame()
    working = pd.concat(frames, ignore_index=True)
    return staffing_by_snapshot(parse_controller_positions(working), fields)


def gate_sweep(panel: pd.DataFrame, arm: str, by_field: bool = True) -> pd.DataFrame:
    """One standardised comparison per separation gate.

    The point of the sweep is that the conclusion should not move when the line moves. If
    the ratio swings across gates, the finding is an artefact of the threshold and has to be
    reported as one. A gradient is different from a swing: a gate wide enough to catch
    ordinary traffic density should show less of an effect, because stratification has
    already removed density.
    """
    rows = []
    for criterion in CRITERIA:
        metric = f"close_{criterion.name}"
        strata = stratified_rates(panel, arm=arm, metric=metric, by_field=by_field)
        pooled = standardised_comparison(strata)
        rows.append({"gate": criterion.name, "events": int(panel[metric].sum()), **pooled})
    return pd.DataFrame(rows)


def field_decomposition(panel: pd.DataFrame, arm: str, metric: str) -> pd.DataFrame:
    """Each field's contribution to the pooled ratio, heaviest first.

    A pooled number that rests on two or three airports is a statement about those airports.
    This is how that gets checked rather than assumed.
    """
    strata = stratified_rates(panel, arm=arm, metric=metric, by_field=True)
    usable = strata.dropna(subset=["rate_off", "rate_on"])
    usable = usable[(usable["pairs_off"] > 0) & (usable["pairs_on"] > 0)]
    if usable.empty:
        return pd.DataFrame(columns=["icao", "weight", "rate_off", "rate_on", "ratio"])

    weight = (usable["pairs_off"] * usable["pairs_on"]) / (usable["pairs_off"] + usable["pairs_on"])
    scored = usable.assign(
        weight=weight,
        weighted_off=usable["rate_off"] * weight,
        weighted_on=usable["rate_on"] * weight,
    )
    per_field = scored.groupby("icao")[["weight", "weighted_off", "weighted_on"]].sum()
    per_field["rate_off"] = per_field["weighted_off"] / per_field["weight"]
    per_field["rate_on"] = per_field["weighted_on"] / per_field["weight"]
    per_field["ratio"] = per_field["rate_on"] / per_field["rate_off"].replace(0, float("nan"))
    per_field["weight_share"] = per_field["weight"] / per_field["weight"].sum()
    columns = ["weight", "weight_share", "rate_off", "rate_on", "ratio"]
    return per_field.sort_values("weight", ascending=False)[columns].reset_index()


def robustness(panel: pd.DataFrame, arm: str, metric: str) -> None:
    """Everything that could make the headline ratio wrong, checked in one place."""
    print(f"\n### Robustness [{arm}, {metric}]\n")

    def ratio(frame: pd.DataFrame) -> float:
        pooled = standardised_comparison(
            stratified_rates(frame, arm=arm, metric=metric, by_field=True)
        )
        return pooled.get("ratio_on_over_off", float("nan"))

    print(f"  within-field, all traffic levels      {ratio(panel):.3f}")

    # Exactly two aircraft: the aircraft-count stratum is then an exact match rather than a
    # bin, so no residual density difference can survive inside it.
    print(f"  exactly two aircraft in the volume    {ratio(panel[panel['aircraft'] == 2]):.3f}")

    volume = panel.groupby("icao")["pairs"].sum().sort_values(ascending=False)
    hubs = set(volume.index[:25])
    print(f"  top 25 fields by pairs                {ratio(panel[panel['icao'].isin(hubs)]):.3f}")
    print(f"  every other field                     {ratio(panel[~panel['icao'].isin(hubs)]):.3f}")

    print("\n  placebo (staffing shuffled within field x count x hour):")
    shuffled = placebo_ratios(panel, arm=arm, metric=metric)
    print("   ", [round(r, 3) for r in shuffled])

    decomposition = field_decomposition(panel, arm, metric)
    print(f"\n  fields contributing: {len(decomposition)}")
    print(decomposition.head(10).round(3).to_string(index=False))
    top = set(decomposition["icao"].head(5))
    print(f"  weight in the top 5: {decomposition['weight_share'].head(5).sum():.3f}")
    print("\n  leave-one-field-out, heaviest five:")
    for name in sorted(top):
        print(f"    without {name:6} {ratio(panel[panel['icao'] != name]):.3f}")


def report(panel: pd.DataFrame, arm: str, label: str, metric: str) -> None:
    print(f"\n### {label} [{metric}]\n")
    print("Crude, ignoring the confound:")
    print(crude_rates(panel, arm=arm, metric=metric).round(3).to_string(index=False))

    strata = stratified_rates(panel, arm=arm, metric=metric)
    print("\nBy stratum (close pairs per 1,000 pairs):")
    print(strata.round(3).to_string(index=False))

    print("\nStandardised:")
    for key, value in standardised_comparison(strata).items():
        print(f"  {key:34} {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="C:/Users/basha/pcas-data")
    parser.add_argument("--out-dir", default="artifacts/staffing")
    parser.add_argument("--reuse", action="store_true", help="use the cached inputs")
    parser.add_argument("--min-field-samples", type=int, default=20)
    parser.add_argument("--metric", default="close_near_1nm", help="gate for the detailed tables")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fields_path = out_dir / "fields.parquet"
    traffic_path = out_dir / "traffic.parquet"
    staffing_path = out_dir / "staffing.parquet"
    cached = all(p.exists() for p in (fields_path, traffic_path, staffing_path))

    if args.reuse and cached:
        fields = pd.read_parquet(fields_path)
        traffic = pd.read_parquet(traffic_path)
        staffing = pd.read_parquet(staffing_path)
        print(f"reused {len(fields)} fields, {len(traffic):,} traffic rows")
    else:
        files = pilot_files(data_dir)
        print(f"{len(files)} pilot files")

        print("pass 1: field positions from stationary aircraft")
        fields = collect_field_positions(files, args.min_field_samples)
        fields.to_parquet(fields_path, index=False)
        print(f"  {len(fields)} fields located")

        print("pass 2: airborne traffic inside those volumes")
        traffic = collect_traffic(files, fields)
        traffic.to_parquet(traffic_path, index=False)
        print(f"  {len(traffic):,} aircraft-snapshot rows")

        print("pass 3: staffing")
        staffing = collect_staffing(controller_files(data_dir), fields["icao"])
        staffing.to_parquet(staffing_path, index=False)
        print(f"  {len(staffing):,} staffed field-snapshots")

    panel = build_panel(traffic, staffing)
    panel.to_parquet(out_dir / "panel.parquet", index=False)

    print(f"\npanel: {len(panel):,} field-snapshots over {panel['icao'].nunique()} fields")
    print(f"pairs: {int(panel['pairs'].sum()):,}")
    for criterion in CRITERIA:
        print(f"  inside {criterion.name}: {int(panel['close_' + criterion.name].sum()):,}")

    # Fields that are always staffed or never staffed carry no within-field contrast, so
    # they are counted separately from the ones the natural experiment actually uses.
    per_field = panel.groupby("icao")["tower"].agg(["mean", "size"])
    mixed = per_field[(per_field["mean"] > 0) & (per_field["mean"] < 1)]
    print(f"fields with both arms: {len(mixed)} of {len(per_field)}")

    both = panel.assign(any_control=panel["tower"] | panel["approach"])

    print("\n### Does the answer depend on where the line is drawn?\n")
    for arm, frame in (("tower", panel), ("any_control", both)):
        print(f"{arm}, across fields:")
        print(gate_sweep(frame, arm, by_field=False).round(3).to_string(index=False))
        print(f"{arm}, within field:")
        print(gate_sweep(frame, arm, by_field=True).round(3).to_string(index=False))

    robustness(panel, "tower", args.metric)
    robustness(both, "any_control", args.metric)

    report(panel, "tower", "Tower online against no tower", args.metric)
    report(panel, "approach", "Approach or departure online against neither", args.metric)
    report(both, "any_control", "Any separating controller against none", args.metric)


if __name__ == "__main__":
    main()
