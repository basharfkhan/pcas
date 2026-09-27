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

`pcas.collect.coverage` reports what was actually captured, per hour of the day. This
matters more than it sounds: the collector runs on a desktop that sleeps, and VATSIM
controller staffing peaks in the evening, so holes landing at the same hours every night
would make the controlled-vs-uncontrolled comparison measure collector uptime rather than
air traffic. Coverage is reported against full collection on every calendar day of the
span, including days missed entirely, so the gaps are stated rather than hidden.

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

Primary dataset is **TartanAviation KBTP**: 368 recording sessions (2020-08 to 2022-10), of
which 298 train, 30 validate and the last **40 are held out chronologically**. That is
500,294 training samples, 4.1x what TrajAir's 111 days gave. Errors in metres, best of K = 1,
measured over 83,958 test windows and 142,720 aircraft.

| model | minADE | minFDE (120 s) | horizontal | vertical | median FDE | p95 FDE |
|---|---|---|---|---|---|---|
| constant velocity | 1594 | 3637 | 3571 | 257 | 3058 | 8346 |
| constant velocity (4 s fit) | 1741 | 3869 | 3760 | 360 | 3261 | 8507 |
| constant turn rate | 1829 | 4058 | 4004 | 250 | 3841 | 7930 |
| Kalman (constant velocity) | 1586 | 3626 | 3559 | 258 | 3045 | 8322 |
| **LSTM** (single aircraft, 3.1M params) | **1130** | **2539** | 2525 | **123** | 2065 | 6208 |

The LSTM cuts trajectory error by 29% against the best physics baseline and 52% vertically.
It also replicates: trained and tested on TrajAir's 111 days instead, the same architecture
gave minADE 1095 m and minFDE 2491 m, within 2% of these numbers on a different, smaller
test set.

Three measurements say this model is limited by its inputs rather than its size or its
optimisation, all pointing the same way:

- **Capacity:** 223k parameters reach a validation FDE of 2485 m; 3.13M reach 2465 m.
- **Optimisation:** fixing the target scaling (below) cut the epochs needed by roughly an
  order of magnitude but moved final test FDE by about 0.5%.
- **What it can see:** one aircraft's own 11 seconds. The aircraft it might conflict with is
  not an input at all.

## Headline result: the learned model owns the 30 to 60 s band, and nothing owns 90 s

Alerting on the 40 held-out TartanAviation KBTP sessions: 37,256 test windows holding 2 or
more aircraft, 88,955 aircraft pairs, **1,686 conflicts**. Conflict = 0.5 nm horizontal and
500 ft vertical broken at the same instant. Every method alerts at the last observed instant,
is scored by identical code, and is checked every second.

**Detection rates only mean something at a matched false alarm rate**, so each method is
swept over its own sensitivity knob (`artifacts/tradeoff_tartan.csv`, from
`scripts/alert_tradeoff.py`): closure-rate alerting sweeps tau, and the predictors sweep how
far ahead they may raise an alert. Reading across at comparable budgets:

| false alarms/hour | method | 0 to 30 s | 30 to 60 s | 60 to 90 s | 90 to 120 s |
|---|---|---|---|---|---|
| ~1.7 | closure rate (tau 40) | 0.675 | 0.091 | | |
| ~2.1 | Kalman (horizon 30 s) | **0.730** | | | |
| ~2.1 | LSTM (horizon 30 s) | 0.637 | | | |
| ~3.4 | closure rate (tau 60) | 0.689 | 0.156 | | |
| ~3.3 | Kalman (horizon 40 s) | **0.736** | 0.174 | | |
| ~3.8 | LSTM (horizon 40 s) | 0.655 | **0.248** | | |
| ~5.7 | closure rate (tau 90) | 0.693 | 0.202 | 0.090 | |
| ~5.8 | Kalman (horizon 60 s) | **0.740** | 0.228 | | |
| ~7.1 | LSTM (horizon 60 s) | 0.662 | **0.309** | | |
| ~7.1 | closure rate (tau 120) | 0.693 | 0.205 | 0.122 | 0.052 |
| ~10.0 | Kalman (horizon 120 s) | 0.746 | 0.249 | 0.151 | 0.063 |
| ~12.6 | LSTM (horizon 90 s) | 0.667 | **0.367** | **0.177** | |
| ~18.7 | LSTM (horizon 120 s) | 0.668 | 0.376 | 0.209 | 0.092 |

Five things this supports, and two it does not:

1. **Close in, physics wins.** In the 0 to 30 s band the Kalman filter leads at every budget
   (0.73 to 0.75), closure-rate alerting follows (0.68 to 0.69), and the LSTM trails
   (0.64 to 0.67). Nothing in that band needs a neural network.
2. **The learned model's clear win is 30 to 60 s.** At ~3.5 false alarms per hour it detects
   0.248 against 0.156 for closure-rate logic, and at ~7 per hour 0.309 against 0.205. That
   is the band where an aircraft's *intent* matters and straight-line motion has stopped
   being informative.
3. **Lead time follows.** Median lead time is 1 s for every closure-rate setting and 1 to 3 s
   for Kalman, against 3 to 24 s for the LSTM. The physics methods fire when the aircraft are
   already converging.
4. **Below about 2 false alarms per hour, use physics.** The learned model cannot be made
   that quiet without losing the horizon that makes it useful.
5. **Better trajectories are not the same as better alerts.** The LSTM cuts trajectory error
   by 29% and still loses inside 30 s. Average error is dominated by ordinary cruise;
   conflicts happen in turning traffic near the field. Optimising the first does not serve
   the second, which is why this project reports alerting metrics rather than FDE alone.

Not supported:

- **The 60 to 90 s band is not a clear win.** Matched on false alarms, the LSTM's 0.177 at
  12.6 per hour sits near Kalman's 0.151 at 10.0. An earlier, smaller sample made this look
  like a 3x improvement. It is not.
- **Nothing works at 90 to 120 s.** The best figure at any budget is 0.092, and the physics
  baselines reach 0.052 to 0.063. Two minutes of warning is out of reach for every method
  here, including the learned one.

### Corrections this section has been through

Recorded because the method matters more than the number.

**The sample was too small, twice.** A 7-day version reported 28% and 17% detection in the
60 to 90 s and 90 to 120 s bands; at 528 conflicts those fell to 5% and 2%, and at 1,686
conflicts the ordering firmed up but the 60 to 90 s advantage shrank to nothing much. Every
conflict statistic here was quoted too confidently at least once before it settled.

**The sensitivity knob was wrong, and it produced a false negative.** The first sweep
quietened the predictors by demanding the predicted separation break the threshold by a
tighter margin. A trajectory off by kilometres at 90 s cannot be asked to predict a near
miss, so tightening destroyed recall rather than trimming false alarms, and the conclusion
written down was that the learned model never beats closure-rate logic. Sweeping the horizon
instead, the direct analogue of tau, reverses that in the 30 to 60 s band. Both knobs stay in
the CSV so the difference is visible.

**Scoring was unfair to the predictors.** Closure-rate alerting solves for the closest point
of approach analytically, while predicted trajectories were only checked at their 10 s
waypoints, so they stepped over brief violations. Alerting now checks every second, which
moved near-term detection by about 20 points.

### Training setup, and a bug worth naming

Each aircraft's window is re-expressed in its own frame: translated so the last observed
position is the origin, rotated so it is heading along +x. Without that the network relearns
the same manoeuvre at every position and heading on the field. Targets are offsets from the
last observed position, so "carry on as you are" sits near the origin of the output space.

Inputs were normalised but **targets were not**, and that cost a training run. With a Huber
delta of 100 m against targets of thousands of metres, almost every sample sat in the loss's
linear regime, so gradients arrived with near-constant magnitude and the output layer barely
moved. The 2-layer model limped through, which is why it was still improving when it hit its
epoch cap; the 3-layer model collapsed to predicting a constant and scored worse than Kalman.
Scaling the targets fixed it.

## What the data is actually like

Both datasets are real recordings, and most of the work was making them safe to model. Six
distinct defects, each of which used to end a run:

| defect | where | how it was handled |
|---|---|---|
| Deflate64 archives | TartanAviation zips | `zipfile` refuses them, and **bsdtar writes correctly-sized files full of padding while only warning** (3.2 GB of plausible, empty CSVs). `stream-unzip` reads them properly |
| Line truncated mid-field | a full-day session file | Drop lines with unbalanced quotes: only ever a truncated tail |
| Binary file in place of CSV | kbtp 2020-09-01, `23.csv` | Header check, skip the file |
| `Altitude` 0 meaning "missing" | both datasets | Excluded: left in, it put the field elevation at 0 m against a true 382 m and switched the ground filter off |
| `Lon` recorded as `-` | TartanAviation sessions | Coerce per row, drop unparsable rows |
| Sessions sampled at 4 to 6 s | mostly KAGC | `RawDay.median_gap_s` flags them: 1 Hz tracks are impossible, and the gap rule otherwise shatters every track and yields nothing silently |
| Field elevation estimated from cruise traffic | a quiet KBTP session | Estimate rejected when implausible (one session put "ground" at 10,666 m, filtering out every aircraft) |

About 1% of sessions are unreadable, so reads go through `iter_days`, which skips and counts
them rather than ending a job that has spent an hour building windows.

**KAGC is parked.** The towered airport was the reason to take this dataset on, since it
promised a towered-versus-non-towered comparison on real aircraft instead of simulated
traffic. It does not survive the data: 7 of 10 sampled sessions report every 4 to 6 s, and the
ones that do not produce a handful of windows with **no multi-agent windows at all**. That
comparison stays with VATSIM, with its simulator caveat stated.

## Roadmap

- [x] Repo scaffold, CI, VATSIM collector
- [x] ADS-B pipeline: runway-relative metres, scene building, day-based splits (no window leakage)
- [x] Metrics harness and physics baselines (constant velocity, constant turn rate, Kalman)
- [x] TCAS-style closure-rate alerting baseline, and conflict labelling
- [x] LSTM baseline (single aircraft, no attention)
- [x] Full 111-day TrajAir dataset
- [x] TartanAviation KBTP: 368 sessions, 4.1x the training data, 1,686 test conflicts
- [ ] Conflict-weighted loss (the LSTM loses inside 30 s because cruise dominates its loss)
- [ ] Like-for-like run on TrajAir's processed data and official split
- [ ] Multi-agent Transformer with social attention (the capacity result says this, not more capacity, is the lever)
- [ ] Multimodal predictions with calibrated uncertainty
- [ ] Ablations, error analysis by flight phase, failure gallery
- [ ] Controlled vs. uncontrolled analysis on VATSIM
- [ ] Live demo: predicted conflicts on live VATSIM traffic

## Data sources and terms

- **VATSIM datafeed**: public status/data endpoints, polled no faster than every 15 s per
  VATSIM's guidance. Review their data usage terms before publishing the live demo.
- **TrajAir**: general aviation trajectory dataset (CMU AirLab, KBTP), cite on use.
- **OpenSky Network**: historical access requires a research account.
