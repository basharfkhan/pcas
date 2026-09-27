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

## Results

Primary dataset is **TartanAviation KBTP**: 368 recording sessions (2020-08 to 2022-10), of
which 298 train, 30 validate and the last **40 are held out chronologically**. That is
500,294 training samples, 4.1x what TrajAir's 111 days gave. Errors in metres, best of K = 1,
over 83,958 test windows and 142,720 aircraft.

| model | params | minADE | minFDE (120 s) | vertical | median FDE |
|---|---|---|---|---|---|
| constant velocity | | 1594 | 3637 | 257 | 3058 |
| constant turn rate | | 1829 | 4058 | 250 | 3841 |
| Kalman (constant velocity) | | 1586 | 3626 | 258 | 3045 |
| LSTM, single aircraft | 3.1M | 1130 | 2539 | 123 | 2065 |
| Transformer, neighbours hidden | 503k | 1130 | 2538 | 125 | 2052 |
| **Transformer, social attention** | 503k | **925** | **2004** | **108** | **1342** |

## The central finding: context, not capacity

Four measurements, in the order they were made, each narrowing where the limit actually was:

| question | measurement | answer |
|---|---|---|
| Is it model size? | 223k params: val FDE 2485 m. 3.13M params: 2465 m | No: 14x capacity buys 0.8% |
| Is it optimisation? | Fixing target scaling cut epochs-to-converge ~10x | No: final test FDE moved 0.5% |
| Is it the architecture? | Transformer with neighbours **hidden**: minFDE 2538 m | No: matches the LSTM's 2539 m |
| Is it the missing context? | Same Transformer with neighbours **visible**: 2004 m | **Yes: 21% better** |

The third row is what makes this an attribution rather than a story. The ablation shares the
architecture, the cached input tensors, the loss, the schedule and the split with the social
model; only the neighbour tensor is masked. It lands within noise of the LSTM on every metric
measured, including each lead-time band (60 to 90 s detection: 0.225 against the LSTM's
0.225). So the gain is not "a Transformer beats an LSTM". It is "seeing the other aircraft
beats not seeing it".

Why that should be true is not mysterious. Aircraft in a traffic pattern are not independent:
a pilot extends the downwind to follow slower traffic, turns base early to fit in front of
someone, or goes around because the runway is occupied. None of that is inferable from one
aircraft's own last 11 seconds, and all of it is fairly predictable given the aircraft it is
sequencing with. It also explains why the gain grows with horizon: over 10 s an aircraft
simply continues, while over 90 s what it does is largely decided by who else is there.

## Alerting: where each method is actually worth using

Alerting on the 40 held-out sessions: 37,256 test windows holding 2 or more aircraft, 88,955
aircraft pairs, **1,686 conflicts**. Conflict = 0.5 nm horizontal and 500 ft vertical broken
at the same instant. Every method alerts at the last observed instant, is scored by identical
code, and is checked every second.

**Detection rates only mean something at a matched false alarm rate**, so each method is swept
over its own sensitivity knob (`artifacts/tradeoff_transformer.csv`, from
`scripts/alert_tradeoff.py`). Reading across at comparable budgets:

| false alarms/hour | method | 0 to 30 s | 30 to 60 s | 60 to 90 s | 90 to 120 s |
|---|---|---|---|---|---|
| ~0.3 | Kalman (horizon 10 s) | **0.872** | | | |
| ~1.7 | closure rate (tau 40) | 0.675 | 0.091 | | |
| ~1.7 | Transformer (horizon 30 s) | 0.665 | | | |
| ~2.1 | Kalman (horizon 30 s) | 0.730 | | | |
| ~3.4 | closure rate (tau 60) | 0.689 | 0.156 | | |
| ~3.3 | Kalman (horizon 40 s) | **0.736** | 0.174 | | |
| ~3.2 | **Transformer (horizon 40 s)** | 0.687 | **0.380** | | |
| ~5.7 | closure rate (tau 90) | 0.693 | 0.202 | 0.090 | |
| ~7.5 | **Transformer (horizon 60 s)** | 0.696 | **0.413** | | |
| ~10.0 | Kalman (horizon 120 s) | **0.746** | 0.249 | 0.151 | 0.063 |
| ~10.8 | **Transformer (margin 0.70)** | 0.348 | 0.303 | **0.251** | **0.126** |
| ~15.6 | **Transformer (horizon 90 s)** | 0.706 | **0.503** | **0.315** | |
| ~26.0 | Transformer (horizon 120 s) | 0.709 | 0.517 | 0.418 | 0.259 |

Read as advice about which method to deploy where:

1. **Under ~2 false alarms per hour, use physics.** Kalman detects 0.872 of conflicts arriving
   inside 10 s at 0.3 per hour. Nothing learned competes at that budget, and for a last-resort
   alert that is the right budget.
2. **Inside 30 s, physics still wins.** Kalman leads at every budget (0.73 to 0.75) with the
   Transformer close behind (0.67 to 0.71). Closure-rate logic sits between them. This band
   does not need a neural network.
3. **From 30 to 90 s, the learned model wins clearly, and at matched cost.** At ~3.3 per hour
   it detects 0.380 of 30 to 60 s conflicts against 0.174 for Kalman and 0.156 for closure-rate
   logic. At ~10.8 per hour it reaches 0.251 in the 60 to 90 s band against Kalman's 0.151 at
   10.0. Median lead time runs 3 to 23 s against 1 to 3 s for the physics methods.
4. **Past 90 s something finally works, barely.** 0.126 at ~10.8 false alarms per hour, twice
   Kalman's 0.063 at a comparable budget. Two minutes of warning remains mostly out of reach.
5. **The knob matters as much as the model.** Limiting the horizon keeps a predictor inside the
   regime where it is accurate; tightening the separation margin instead wrecks near-term
   detection (the Transformer drops to 0.348 at 0 to 30 s) while buying the long bands. Which
   knob to use is an operational choice, not a detail.

Two earlier conclusions in this file were withdrawn when this model arrived: "the 60 to 90 s
band is not a clear win" and "nothing works at 90 to 120 s". Both were true of the physics
baselines and the single-aircraft LSTM. Neither survived giving a model the other aircraft.

## Multimodal predictions, and an honest look at the probabilities

An aircraft on downwind either turns base or extends, and the average of those two futures is
a path it would never fly. So the model can emit K hypotheses with a probability each, trained
winner-takes-all with a cross-entropy over the mode logits. With two aircraft carrying K
hypotheses each, a pair has K x K possible futures, and the conflict probability is the weight
of the combinations that violate the threshold. The alert becomes a number rather than a
yes/no.

**The probability threshold turns out to be the best sensitivity knob available.** Compare the
three ways of quietening a predictor, at around 10 false alarms per hour:

| method | FA/hour | 0 to 30 s | 30 to 60 s | 60 to 90 s | 90 to 120 s |
|---|---|---|---|---|---|
| Kalman (horizon 120 s) | 10.0 | **0.746** | 0.249 | 0.151 | 0.063 |
| Transformer, 1 mode (margin knob) | 10.8 | 0.348 | 0.303 | 0.251 | 0.126 |
| **Transformer, 6 modes (P >= 0.50)** | 9.3 | 0.700 | **0.422** | 0.235 | **0.138** |

The single-mode model can only be made quiet by tightening its separation margin, which wrecks
near-term detection (0.348). The multimodal model is made quiet by raising the probability bar
instead, which keeps 0.700 close in while still beating the Kalman filter 1.7x at 30 to 60 s
and 2.2x past 90 s. Sweeping further trades false alarms for recall smoothly:

| P >= | FA/hour | 0 to 30 s | 30 to 60 s | 60 to 90 s | 90 to 120 s | median lead |
|---|---|---|---|---|---|---|
| 0.50 | 9.3 | 0.700 | 0.422 | 0.235 | 0.138 | 14 s |
| 0.35 | 19.3 | 0.736 | 0.523 | 0.360 | 0.241 | 23 s |
| 0.20 | 43.0 | 0.784 | 0.653 | 0.534 | 0.376 | 30 s |
| 0.05 | 116.9 | 0.844 | 0.847 | 0.788 | 0.606 | 39 s |

Those high-recall rows are not deployable at 40 to 120 false alarms per hour, but they do say
something worth knowing: the information needed to catch most conflicts two minutes ahead is
present in the data. What is missing is a way to be selective about it.

### The probabilities are not calibrated

`AlertScorer.reliability()` compares what the model claimed against what happened:

| stated probability | conflicts actually followed | pairs |
|---|---|---|
| 0.28 | 0.07 | 3,135 |
| 0.49 | 0.15 | 1,174 |
| 0.69 | 0.27 | 579 |
| 0.94 | 0.70 | 535 |

**Overconfident by a factor of 2 to 4 across the range.** The ordering is sound, so a higher
number really does mean a likelier conflict and thresholding it works, which is why the sweep
above behaves sensibly. But these numbers cannot be shown to a pilot as percentages, and this
project is not going to print "30% chance" next to a figure that means 7%.

Three candidates for why, in the order worth testing: the two aircraft's hypotheses are
combined as if independent, when aircraft sequencing with each other are correlated;
winner-takes-all training optimises the winning trajectory and never asks the mode
probabilities to be calibrated; and a hard conflict threshold turns a near miss into a
coin-flip that the model has no way to express. Post-hoc calibration on the validation split
(isotonic or Platt) is the cheap first move, and it is the next thing on the roadmap.

A note on comparing numbers across models: minADE and minFDE over K hypotheses are best-of-K,
so the 6-mode model's minFDE of 840 m is not comparable to the single-mode model's 2004 m.
Alerting at a matched false alarm rate is the comparison that stays fair, because extra
hypotheses produce extra alerts as well as extra chances to be right.

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

**Two conclusions were overturned by a better model, not by a better measurement.** With
only physics baselines and a single-aircraft LSTM in hand, this file said the 60 to 90 s band
was not a clear win and that nothing worked past 90 s. Social attention reached 0.251 and
0.126 in those bands at a matched false alarm rate. A negative result about the models you
have is not a negative result about the problem.

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
- [x] Multi-agent Transformer with social attention, plus the ablation that attributes the gain to context
- [x] Multimodal predictions: K hypotheses, joint-mode conflict probability, reliability curve
- [ ] Calibrate those probabilities (they are 2 to 4x overconfident; isotonic on the validation split)
- [ ] Ablations, error analysis by flight phase, failure gallery
- [ ] Controlled vs. uncontrolled analysis on VATSIM
- [ ] Live demo: predicted conflicts on live VATSIM traffic

## Data sources and terms

- **VATSIM datafeed**: public status/data endpoints, polled no faster than every 15 s per
  VATSIM's guidance. Review their data usage terms before publishing the live demo.
- **TrajAir**: general aviation trajectory dataset (CMU AirLab, KBTP), cite on use.
- **OpenSky Network**: historical access requires a research account.
