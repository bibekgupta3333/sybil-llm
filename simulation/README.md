# RoadFM-Lite · Window Simulator

Interactive replay of the prepared trajectory windows (20 timesteps × 13 kinematic features) that
RoadFM-Lite pretrains on, label-free. Built to answer "what does one window actually look like?" —
position, speed, acceleration, heading and the derived deltas over time, how stride-10 windows tile an
identity's drive, and what a forged Sybil identity looks like in motion next to an honest one.

TypeScript + Vite, zero runtime dependencies (Canvas 2D). Light Power BI-style theme matches `tracker.html`.

## Run

```bash
cd simulation
npm install          # once
npm run dev          # http://localhost:5173
npm run test         # Vitest, core/ unit tests
npm run typecheck    # tsc --noEmit
npm run build        # static bundle in dist/  (npm run preview to serve it)
```

## Views

| Panel | What it shows |
|---|---|
| **Map replay** | Trail coloured by ‖acceleration‖, vehicle marker, amber heading arrow (unit vector), blue velocity vector, dots at every window start (stride 10), dashed green box = benign road extent, **dashed red segments = teleports** (implied speed > 60 m/s). Auto-fits the identity; `L` locks to the road extent (off-screen vehicles become edge ticks with distance). |
| **Kinematics strips** | speed, ‖acl‖, `dt`, ‖Δpos‖, residual ‖Δpos/dt − spd‖ on a shared step axis, alternating bands per stride block, playhead-synced readouts, red ticks on flagged steps. |
| **Window tensor** | The current 20×13 window as the model sees it — z-scored with the train-split stats (`N` toggles raw). Highlighted column = current step. Caption gives the `X_windows.npy` row. |
| **Physics consistency** | Live gauges for implied speed, claimed speed, residual and heading-vs-velocity angle, a plausibility verdict, and identity-level teleport / inconsistency rates. This is the label-free signal the pretext tasks should expose. |
| **Provenance strip** | `sender_uid`, physical vehicle, "k of N pseudonyms survived `min_seq_len=20`", run, `X_windows` row range, tags, **data source** (`prepared` vs `raw_veremi`, see below), current step/window/time. Every pixel traces to a row. |

Modes: **Single** (one pseudonym), **Sybil sender** (all surviving pseudonyms of one physical vehicle,
time-aligned on absolute simulation time — the Sybil act itself), **Compare** (two identities on one
clock), **Map compare** (one physical vehicle's real path next to its fabricated broadcast, on the map —
see below), **Window compare** (a frozen ground-truth window next to a frozen attack window, no shared
clock — see below).

Keyboard: `Space` play/pause · `←/→` step · `[`/`]` window · `Home`/`End` · `L` lock map · `G` compress
gaps > 2 s · `N` heatmap normalized/raw. "Export map frame as PNG" saves the current map for slides.

## v2 — scenario picker, window navigation, zoom/pan, Window Compare

- **Scenario picker.** The sidebar's "Scenario" section lists every VeReMi run (`subfolder` × time
  `group`) with its benign/attacker identity counts. Selecting one scopes every identity picker (Single,
  Sybil sender, Compare, Window compare) to that run only, and switches the map's dashed context box and
  `L` lock target from the manifest-wide benign road extent to that scenario's own bounding box. "All
  scenarios" clears it.
- **Window scrubber.** A tick strip under the map caption — one tick per stride-10 window of the identity
  the clock is following. Click a tick to jump straight to that window; a tick is tinted amber/red if any
  step inside it is flagged inconsistent/teleporting, so defect windows are visible before you press play.
- **Manual zoom/pan.** Mouse wheel zooms the map around the cursor; click-drag pans. Both persist until the
  scene or lock-to-road target changes (which re-fits); "Reset map zoom/pan" in the sidebar restores the
  last auto-fit on demand.
- **Window compare mode.** Pick a benign identity + one of its windows on the left, an attacker identity +
  one of its windows on the right (both pickers scope to the active scenario, and the attacker side has its
  own family filter). Each side is `Identity.windowSlice(k)` — a real, frozen 20-step window — fed into its
  own heatmap, mini kinematics strip, physics-consistency panel and provenance strip, with a `◂ step / step ▸`
  control to walk its 20 steps independently. No shared playback clock: this is for holding a clean window
  and a forged one side by side, not for watching either replay.

## v3 — prepared data vs. raw VeReMi, and Map Compare

Every prepared window in this app — `X_windows.npy`, and everything shown before v3 — is built from
`traceGroundTruthJSON`: each vehicle's **true physical position**, even for confirmed attackers. A Sybil
attacker's **fabricated** broadcast position only exists in a *neighbor's* raw received-message log
(`traceJSON-*.json`, a type-3 entry under a `senderPseudo` ground truth never recorded for that sender).
That signal was outside the pipeline entirely until now.

- **`data_source` on every identity** (`"prepared"` or `"raw_veremi"`), shown in the provenance strip.
  `raw_veremi` identities are read directly from a neighbor's broadcast log, tagged `forged_broadcast`,
  and are never part of `X_windows.npy` (their `X_windows rows` reads `−1`).
- **Paired identities.** For physical vehicles where a fabricated pseudonym was found, the prepared
  (ground-truth) identity and its `raw_veremi` companion record each other's id in
  `paired_identity_id`. The provenance strip shows a "→ view the other source for this vehicle" button
  wherever a pair exists (and an explicit "no raw-broadcast divergence exists for benign vehicles" line
  where it doesn't — benign vehicles never fabricate anything, so there's nothing to pair).
- **Map Compare mode.** Pick one of the paired physical vehicles; the map shows exactly two trails —
  blue "ground truth" (prepared data) and red "attack broadcast" (raw VeReMi) — on the same clock. A
  "Strips / heatmap / consistency follow" toggle switches which trail the other panels inspect, so you
  can watch the physics-consistency gauges break down (residual, heading-vs-velocity) on the fabricated
  trail specifically.
- **Real finding from the reference sample (seed 42):** of the 4 attack families, only **GridSybil** produced any
  resolvable prepared/raw-VeReMi pairs (20, from a candidate pool drawn across every sampled GridSybil
  vehicle) — DataReplaySybil, DoSRandomSybil and DoSDisruptiveSybil attackers in this sample don't
  broadcast under an *additional* fabricated pseudonym the way GridSybil does; they tamper with content
  under their own existing identity instead. Re-running the exporter may find different pairs (or none)
  depending on the sample seed and `forged_pairs_per_cell`.

## v4 — Benign vs attack, three maps

- **Benign vs attack mode.** Pick a match; the map panel splits into three maps — **x · Benign vehicle**,
  **y · Attack broadcast**, **Both** — on one clock and one shared extent (the union of both trails), so a
  point means the same place in every panel. Wheel/drag zoom and pan move all three together. The
  "follow" toggle picks which trace drives the strips, heatmap and consistency panel.
- **The two are different vehicles.** The benign trace is a Benign-labeled vehicle from the same run; the
  attack trace is an attacker's fabricated broadcast (raw VeReMi). Both are clipped to the time window
  they share (`Identity.timeSlice`, snapped to a window boundary so `X_windows` rows stay truthful).
- **Every attack type, labeled.** The sidebar filters by attack type (GridSybil, DataReplay, DoSRandom,
  DoSDisruptive, with counts), groups matches under a header per type, and shows each pair's run folder
  (e.g. `DoSRandomSybil_0709/VeReMi_25200_28800_…`). A "Comparing" card names the attack type, its observed
  pattern (from `docs/research-notes/data_understanding/attack-taxonomy.md`), the folder, both vehicles and
  the window. The map titles and caption carry the same type and folder.
- **What the attack trace is, per type.** GridSybil: one fake pseudonym ground truth never records.
  DataReplay / DoSRandom / DoSDisruptive: they rotate pseudonyms every 1–2 messages, so the trace is every
  message heard from that vehicle across all its pseudonyms (`attack_trace: "all_pseudonyms"`).
- **How matches are chosen.** For each fabricated broadcast, the exporter picks the Benign identity in
  the same run with the longest time overlap, requiring at least one full 20-step window on both sides
  (`BenignMatcher`). Vehicles not already in the sample are added and tagged `overlap_benign`. In the
  reference sample (seed 42) 90 of 92 attack traces got a match: GridSybil 19, DataReplay 24, DoSRandom 24, DoSDisruptive 23.
- **Window compare below the maps.** A bottom panel holds one benign window (x) next to one attack
  window (y): heatmap, kinematic strips, physics consistency and provenance per side, each stepped on
  its own. Pick windows with the tick rows. When several attacks are shown, choose which one with the
  attack select. "Both windows at the playhead" jumps each side to the window under the current clock.
- **Several attacks against one benign.** The "Comparing" card lists the other raw-VeReMi attack traces
  of the same type from the same run that overlap the benign vehicle for at least 30 steps
  (`Dataset.attacksOverlapping`). Tick them to add them to the y map and the Both map, each in its own
  colour and labelled `v{sender}`. The shared window becomes the benign trace's time range, clipped to
  the span of the selected attacks (`Dataset.benignMultiAttackView`). In the reference sample (seed 42) one benign
  vehicle overlaps at most 3 attack traces (GridSybil: 5 benign vehicles overlap 2 or more). Raising
  `forged_pairs_per_cell` in the exporter would give more.
- **Raw-data version:** `notebooks/benign_vs_attack_maps.ipynb` builds the same three maps directly from
  `data/VeReMi-Dataset` for any run, plus a claimed-vs-implied speed plot.

## Data — `public/data/`

A stratified sample (~5 MB) produced by `scripts/export_simulation_sample.py` (read-only on `data/`). It is **not
committed**: `simulation/public/` is gitignored, so generate it once after cloning (the encoder tab's sample comes from
`npm run sim:data`, see below):

```bash
cd ..   # repo root
.venv/bin/python scripts/export_simulation_sample.py --out simulation/public/data --seed 42
.venv/bin/python -m pytest scripts/tests
```

- `manifest.json` — feature order, `norm_mean/std` (copied from `config.json`, never recomputed), class
  palette, benign `road_bbox`, and one record per identity with provenance + byte offsets.
- `steps.f32` — row-major float32 `(total_steps, 13)`; `times.f64` — absolute simulation seconds.

**Stitching contract.** Windows are `X[start:start+20]` slices at stride 10 of an identity's full
sequence (deltas were computed over the full sequence before slicing). The exporter reconstructs the
sequence as `concat(w[k][0:10] for k < K−1) + w[K−1]`, asserts pairwise overlaps
`w[k][10:20] == w[k+1][0:10]` bit-for-bit, re-slices and asserts equality with the source, and anchors
time at each window's recorded `start_time` (asserting each `end_time`). Up to 9 tail rows after the last
window are unrecoverable by construction. `dt == 0` occurs only at an identity's very first step.

**Sampling.** 35 identities per class × time-group cell (10 cells). Attackers are drawn by *physical
sender* (all surviving pseudonyms, ≤ 8 per sender) so the Sybil-sender view has complete groups. Two
longest identities per class are tagged `stress`. For up to 6 sampled attackers per (family, group)
cell, the raw dataset is additionally scanned once per scenario subfolder for a fabricated broadcast
pseudonym (see "v3" above); not every attempt finds one.

**`defect_demo`.** The two `GridSybil_0709 … _1` identities are the known windowing splice (653 vehicles
merged under the sentinel `senderPseudo == 1`; see
`docs/research-notes/data_understanding/gridsybil-windowing-defect.md`). Their first 40 windows are
exported on purpose: filter tag = `defect_demo` and step through them to watch the "vehicle" hop between
653 cars at ~20,000 m/s. This is what the reconstruction objective was trained on in v1.

## Layout

```
scripts/export_simulation_sample.py   SampleConfig · WindowStore · IdentityStitcher ·
                                      PseudonymResolver · StratifiedSampler · BroadcastForger ·
                                      SampleWriter · Exporter
scripts/tests/test_export_simulation_sample.py
src/core/      manifest (+ data_source/paired_identity_id) · dataset (+ pairedAttackers) ·
               identity (+ windowSlice, dataSource, pairedId) · scenario · window_view · kinematics ·
               playback · viewport (+ zoomBy/panBy/reset) (+ tests)
src/render/    layer (CanvasSurface, static/dynamic split) · map_renderer (+ setScene label override) ·
               strip_renderer · heatmap_renderer
src/ui/        controls (scenario + 5 modes) · transport · consistency_panel ·
               provenance (+ data-source line, switch-source button) · window_scrubber ·
               window_compare_panel · theme · styles.css
src/main.ts    App: dataset → playback clock → renderers/UI
```

## Encoder input tab (T = 64, benign + GridSybil trial)

A second tab, **"Encoder input (T = 64)"**, shows exactly what the TimesNet encoder reads: 64 × 13 windows + mask
from `src/data/encoder_input/benign_gridsybil/T64/`. Code lives in `src/encoder/` (core logic + tests in
`src/encoder/core/`, UI in `ui/` and `render/`), loaded on first open; the legacy tab is unchanged.

- **Data:** `npm run sim:data` (= `scripts/export_encoder_sample.py`, read-only on `src/data/`) writes a 7.7 MB sample to
  `public/data/encoder/` (`manifest.json`, `x.f32` = normalised real rows only, `predictions/index.json`).
  `public/` is gitignored, so run the exporter once after cloning. Sample rule (seed 0): per split × scenario ×
  class, whole sender vehicles with all their windows (4,549 windows), plus rule-picked presenter windows badged
  "illustrative".
- **Select:** split (train / pretrain_val / val / test), scenario group, run, class; **Single window** or **Group**
  (a link's crops, a sender vehicle — privileged for pseudonym-1 ghosts, a receiver, a seeded batch of 32).
- **Panels:** caveats banner; 64 × 13 tensor heatmap with padding hatched and the mask column; mask & length (with
  the length-only AUC 0.531 note); spatial view (claimed vs receiver positions + range, metres); feature strips (real
  rows only); provenance (ids → raw VeReMi trace, split integrity over the full dataset); detection (empty until a
  model writes `predictions/`); full-dataset per-split table (from metadata, not the sample).
- **Toggles:** normalised (what the model sees) / raw units; labels on/off (labels are evaluation only).
- **Links:** the URL hash (`#encoder?split=test&win=<id>…`) reopens the same view — use "copy link" for slides.
- **Predictions (later):** add `public/data/encoder/predictions/<model>.json` (`{window_id: {score, pred}}`) and list
  it in `predictions/index.json` with checkpoint hash, seed, training splits and the val threshold; the detection
  panel then shows scores and warns on windows from the model's training split.

## Detection tab (model-all, label-free)

A third tab, **"Detection (model-all)"**, shows what the pretrained `model-all` encoder (`--dataset all`, T = 24) does on
test windows without any fine-tuning: a label-free kNN anomaly score (plan task S1.4.4). Code lives in
`src/detection/` (core + tests in `core/`, histogram canvas in `render/`, DOM in `ui/`), loaded on first open; it reuses
the encoder tab's helpers (`dom`, `StaticCanvas`, `SpatialMap`, colours, `.enc-*` styles).

- **Data:** `npm run sim:detect` (repo root, = `scripts/export_detection_sample.py`) writes `public/data/detection/`
  (`manifest.json` schema 1 + `x.f32` = little-endian float32 real rows in **raw units**, `n × 13` per window,
  concatenated in `windows` order, `offset` = first row). Without it the tab says to run that command. The loader
  checks that offsets are contiguous and cover `x.f32` exactly, so a stale or truncated bundle fails loudly.
- **Method (from the manifest, not computed here):** score = 1 − mean cosine similarity of the window's embedding to
  its K nearest neighbours in a bank of unlabeled train windows; θ = a label-free train-score quantile (no val split
  exists, D8′, so θ is not tuned). Scope: test split, group 1416 only (0709 waits on F14).
- **Panels:** caveats + model / method / scope cards; metrics per class for the scored sample and the full 1416 test
  split (AUROC vs benign, length-only baseline AUROC, flag rate at θ, n, mean rows — values verbatim, `—` = null);
  score histogram per class on one axis with the θ line and the selected window's marker; window list (filter by
  true class and flagged / not flagged, sort by score); selected-window detail (score, `rank_pct`, flag, true class
  badged "evaluation only", claimed vs receiver positions in metres, the n × 13 feature rows as a shaded table).
- **Labels** are evaluation only: the exporter reads them after scoring; the page uses them for filtering and metrics.
- **Links:** `#detection?cls=…&flag=…&sort=…&win=…` reopens the same view.

### Layout (simplified)

The tab (**"Detect attacks"**) opens in **Test yourself** mode and reads top to bottom:

1. **The window card** with three steps — *Look* (the trajectory: where the sender says it is vs where the receiver
   is) → **Let the model decide** (“ATTACK” / “normal” and an unusualness bar with the alarm threshold) → **Show the
   answer** (the true class and, in plain words, *attack caught / false alarm / correctly left alone / attack missed*).
   Score, threshold, rank, scenario and window id sit under *Technical details*; the raw 13-feature rows under
   *Raw message values*.
2. **How well does it work?** — one small table per attack: share caught, the model's AUROC, the length-only rule's
   AUROC, and whether the model beats that rule; plus the false-alarm rate on normal traffic.
3. Folded: **Score chart** (per-class histograms, threshold line) and **How it works** (four-line method, caveats,
   the exact exporter numbers verbatim, checkpoint and threshold rule).

**Test yourself** lists windows in a fixed shuffled order as `Window #n` with only their length — no score, alarm,
class, scenario or id — and the URL holds only `win=q<n>` (the sample order and the id would give the class away).
**🎲 Random window** picks an unseen one; the sidebar counts how often the model was right (this session only).
**Explore** shows every step at once and adds class / alarm filters.
