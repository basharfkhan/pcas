# PCAS: Predictive Collision Awareness System

Learned conflict warning for airports **without a control tower**, where most midair
collisions happen and where certified collision avoidance systems are least useful.

> Status: **early**. The VATSIM collector is running; the model is not built yet.

## The problem

Most US airports have no tower. Pilots there separate themselves by looking out the window
and announcing their positions on a shared frequency. Meanwhile:

- **TCAS II** the certified system - is carried mainly by airliners and larger turbine
  aircraft, not the trainers and light singles flying the pattern at these fields.
- Even where it is carried, TCAS II **inhibits resolution advisories below roughly 1,000 ft
  AGL**, because telling an aircraft to descend near the ground is dangerous. The traffic
  pattern is flown at and below that altitude.
- Its logic is **geometric**: time-to-closest-approach from current closure rate. That works
  for two airliners converging in cruise. In a pattern, where aircraft turn constantly, it
  both misses conflicts that have not developed yet and fires on turns that resolve
  themselves.

PCAS asks whether a model that has *learned the shape of traffic at an airport* can warn
earlier than closure-rate logic at the same false-alarm rate.

**Headline metric:** warning lead time vs. false-alarm rate against a TCAS-style
closure-rate baseline. Trajectory error (minADE/minFDE) is a supporting metric, not the
result.

This is a research prototype, not a safety system. The direction is the same one the
industry is already taking - **ACAS X**, TCAS's successor, replaces geometric logic with
probabilistic prediction, and **ACAS Xu** targets exactly this airspace for drones.

*Not affiliated with, or related to, the discontinued Zaon PCAS product.*

## Approach

| Stage | Data | Why |
|---|---|---|
| Model training and evaluation | **ADS-B** (TrajAir benchmark, KBTP; OpenSky later) | Real flights, high sample rate, published baselines to compare against |
| Natural experiment + live demo | **VATSIM datafeed** | The only source where *controller presence is a variable*: the same airport is uncontrolled one hour and staffed the next |

The model is a multi-agent Transformer: attention over time within each aircraft's track,
and attention across aircraft in the same scene, predicting several possible futures with
calibrated probabilities. Those futures are turned into a conflict probability between
aircraft pairs, which is what actually raises an alert.

## Why VATSIM

Real ADS-B cannot answer "does traffic become more predictable when a controller is
online?", because a given airport's tower status barely changes. On VATSIM it changes hourly.
The feed publishes both pilot positions and which controllers are connected, so the same
field can be compared staffed vs. unstaffed:

- pattern conformance and spacing on final
- how often aircraft come close to each other
- whether an ML warning matters more when nobody is watching

**Caveats stated up front:** the feed updates only every ~15 s, which is coarse for
close-approach detection, and simulator pilots do not always fly like real ones. VATSIM
carries the analysis and the live demo; ADS-B carries the quantitative results.

## The collector

Polls the public VATSIM datafeed every 15 s (its refresh rate - the config refuses to go
faster) and appends to date-partitioned Parquet. Pilot rows keep position, altitude,
groundspeed, heading and flight plan; controller rows record which facilities are online.
The free-text `name` field on each connection is **not stored**, only the numeric CID, so
one aircraft's samples can be stitched into a track.

```bash
pip install -e ".[dev]"
python -m pcas.collect --once          # single poll, verify it works
python -m pcas.collect                 # run continuously; Ctrl-C flushes and exits
```

Output layout:

```
$PCAS_DATA_DIR/
  pilots/date=2026-09-25/pilots-20260925T140500.parquet
  controllers/date=2026-09-25/controllers-20260925T140500.parquet
```

Configure in `configs/collector.yaml`, or override the destination with `PCAS_DATA_DIR`.
The default data directory sits outside OneDrive on purpose, this grows by tens of MB per
day and syncing every flush would thrash the sync client.

Start it early: the controlled-vs-uncontrolled analysis needs weeks of accumulated history.

## The ADS-B pipeline

Two tracks, deliberately kept separate.

**1. TrajAir's `processed_data`, for comparability.** Scene files are already in an
airport-centred frame (km, 1 Hz), so the loader only converts to metres. Used with the
dataset's own train/test split so numbers line up with published results (TrajAirNet,
[ASCENT](https://arxiv.org/abs/2603.16550)).

That split leaks, and we measured it: scene files are numbered with no date, and the 70/30
split is random over scenes, so **all 7 days of `7days1` appear on both sides**. A model
can be tested on the same day's traffic, often the same aircraft in the same pattern, that
it trained on. Published numbers on this split should be read with that in mind.

**2. Our own pipeline from `raw_data`, for the honest numbers.** The raw CSVs are one file
per day with absolute UTC timestamps, so rebuilding from them gives real dates, real times
(conflict labelling needs them to pair aircraft), and visible filtering choices:

| Step | Rule |
|---|---|
| Frame | lat/lon to metres, x along the runway (`geo.py`) |
| Ground | drop samples within 30 m of the field elevation, itself estimated from the data because the raw altitudes are not reliably MSL |
| Terminal area | keep traffic within 15 km; the receiver hears enroute aircraft out to 110 km |
| Gaps | split a track at any hole over 5 s instead of interpolating across it |
| Resample | interpolate onto whole seconds, 1 Hz |
| Frozen tracks | drop tracks whose whole path is under 200 m (stuck transponders) |

On `7days1` (7 days): **375 scenes, 17,683 windows, 6,097 of them multi-agent.** Windows are
11 s observed and 120 s predicted at 10 s steps, matching TrajAirNet's defaults.

Scene ids carry the date (`2020-09-24_2031`), so day-based and chronological splits read it
straight off the name and cannot leak a day across both sides.

**Known issue:** the frozen-track filter works per track, so a track that moves overall but
freezes for a stretch still gets through. A per-window check belongs with the baselines,
where a stationary target would otherwise flatter every metric.

## Baseline results

Physics baselines on `7days1`, with the last 2 days held out chronologically (6,324 test
windows, 10,015 aircraft). Errors in metres, best of K = 1 hypothesis.

| model | minADE | minFDE (120 s) | horizontal | vertical | median FDE | p95 FDE |
|---|---|---|---|---|---|---|
| constant velocity | 1587 | 3621 | 3597 | 210 | 3289 | 8170 |
| constant velocity (4 s fit) | 1609 | 3664 | 3622 | 277 | 3285 | 8172 |
| constant turn rate | 1661 | 3754 | 3739 | 203 | 3624 | 7475 |
| Kalman (constant velocity) | **1578** | **3607** | 3583 | 209 | 3284 | 8153 |

Error grows steeply with horizon (Kalman, mean/median):

| horizon | 10 s | 30 s | 60 s | 90 s | 120 s |
|---|---|---|---|---|---|
| mean | 97 | 423 | 1234 | 2331 | 3607 |
| median | 48 | 194 | 706 | 1749 | 3284 |

Three things this says:

1. **The physics assumption dies somewhere past 30 s.** Under 20 s, dead reckoning is
   decent, which is why closure-rate alerting works for its intended job. At 120 s it is
   off by kilometres, and 120 s is where a pilot could still act on a warning.
2. **Turning is where the error lives.** Aircraft turning at 1 deg/s or more during the
   observed window have a median 120 s error of 5053 m, against 2124 m for aircraft that
   look straight. Note even the "straight" ones are badly wrong: they turn *after* the
   observation window, in the pattern. A model that has learned the pattern should.
3. **Constant turn rate is not automatically better.** Extrapolating a turn for 120 s
   overshoots when the aircraft rolls out, so it loses to plain constant velocity on
   average while having the lowest p95. Both are beatable.

Caveat on comparing to published numbers: these come from our own raw-data pipeline, which
keeps any traffic within 15 km (including transiting aircraft above 120 kt), not from
TrajAir's filtered `processed_data`. A like-for-like run on their processed data and
official split is still to do, and belongs beside the leakage finding above.

## Headline result so far: where closure-rate logic runs out

Alerting comparison on `7days1`, last 2 days held out, 2,563 test windows holding 2 or more
aircraft. Conflict = 0.5 nm horizontal and 500 ft vertical broken at the same instant.
Both methods alert at the last observed instant and are scored by identical code.

| method | events | detected | detection rate | false alarms/hour | median lead time |
|---|---|---|---|---|---|
| closure rate (tau 20 s) | 23 | 14 | 0.61 | 0.3 | 1 s |
| closure rate (tau 40 s) | 33 | 16 | 0.49 | 1.8 | 1 s |
| closure rate (tau 120 s) | 77 | 26 | 0.34 | 7.3 | 9 s |
| constant velocity (120 s) | 77 | 25 | 0.33 | 8.8 | 19 s |
| Kalman CV (120 s) | 77 | **27** | 0.35 | 8.7 | **19 s** |

Split by how far ahead the conflict actually was:

| actual lead time | closure rate (tau 40 s) | Kalman CV (120 s) |
|---|---|---|
| 0 to 30 s | 16 of 29 (0.55) | 15 of 29 (0.52) |
| 30 to 60 s | **0 of 12 (0.00)** | 4 of 12 (0.33) |
| 60 to 90 s | **0 of 18 (0.00)** | 5 of 18 (0.28) |
| 90 to 120 s | **0 of 18 (0.00)** | 3 of 18 (0.17) |

That zero column is the point of the project. Closure-rate logic is not bad at its job: it
catches the majority of conflicts inside 30 s, at a very low false alarm rate. It simply
cannot see past its horizon, and its detections arrive with a median lead time of 1 s,
meaning the aircraft are already converging as it fires. Extending tau to 120 s does not
fix that: it just alerts on more pairs, and its lead time stays short.

Even a dumb straight-line predictor over 120 s finds some of what closure-rate logic
misses, at a cost of roughly 9 false alarms per hour against 1.8. **The learned model's job
is to hold that longer horizon while pushing the false alarm rate back down.** Headroom is
large: the best method here detects 35% of conflicts overall.

Caveat, stated plainly: 77 events across 2 days is a thin sample, so these numbers are
provisional and the buckets are small. The full 111-day dataset is the fix, and it is the
next thing to pull.

## Roadmap

- [x] Repo scaffold, CI, VATSIM collector
- [x] ADS-B pipeline: runway-relative metres, scene building, day-based splits (no window leakage)
- [x] Metrics harness and physics baselines (constant velocity, constant turn rate, Kalman)
- [x] TCAS-style closure-rate alerting baseline, and conflict labelling
- [ ] LSTM baseline
- [ ] Like-for-like run on TrajAir's processed data and official split
- [ ] Multi-agent Transformer with social attention
- [ ] Multimodal predictions with calibrated uncertainty
- [ ] Ablations, error analysis by flight phase, failure gallery
- [ ] Controlled vs. uncontrolled analysis on VATSIM
- [ ] Live demo: predicted conflicts on live VATSIM traffic

## Data sources and terms

- **VATSIM datafeed**: public status/data endpoints, polled no faster than every 15 s per
  VATSIM's guidance. Review their data usage terms before publishing the live demo.
- **TrajAir**: general aviation trajectory dataset (CMU AirLab, KBTP), cite on use.
- **OpenSky Network**: historical access requires a research account.
