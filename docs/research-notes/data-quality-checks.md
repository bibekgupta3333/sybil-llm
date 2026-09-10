# Data Quality Checks (Task 1.7)

Four independent, reproducible checks on the VeReMi-Extension pipeline: duplicate messages,
non-monotonic `sendTime`, inter-message gaps (`dt`), and coordinate-frame sanity. Each has a clear
PASS/FLAGGED verdict with a counted anomaly rate. Reproduced as a runnable QA cell block at the end of
`notebooks/eda_veremi.ipynb`.

**Scope of the raw-file checks (1 & 2):** sampled directly from `data/VeReMi-Dataset/` (13 GB, read-only,
not scanned in full) — 32 trace files spanning all 8 family×group folders, 21,306 type-3 (BSM receipt)
messages across 793 per-vehicle sender sequences. **Scope of the derived-feature checks (3 & 4):** a random
sample of 50,000 of the 285,926 prepared windows in `data/prepared_data/X_windows.npy`, cross-referenced
against `window_metadata.parquet` for the `is_attacker` split.

## Check 1 — Duplicate messages: **PASS**

Method: for each sampled trace file, key every **type-3** record (an actual BSM receipt — `sendTime`,
`sender`, `messageID`) and count repeats of that key within the file.

- Result: **0 duplicates** across 21,306 type-3 messages in 32 files.
- Caveat that mattered during this check: raw traces also contain **type-2** records (`type: 2`), which are
  a vehicle's *self-observation* of its own state and carry no `sendTime`/`sender`/`messageID` — they must be
  excluded from the duplicate key, or every type-2 row in a file collapses onto the same `(None, None, None)`
  key and reads as a false-positive duplicate. (An earlier pass before excluding type-2 rows reported 819
  "duplicates" out of 6,116 lines — all type-2 artifacts of the check itself, not real data.)

## Check 2 — Non-monotonic `sendTime`: **PASS**

Method: for each sampled file, group type-3 records by `sender` and confirm `sendTime` strictly increases
step-to-step within each sender's message sequence.

- Result: **0 out-of-order steps** across 793 sender sequences.
- VeReMi's BSM logs are internally consistent in time ordering; no reordering/replay artifact appears at the
  raw-message level in the sample. (Replay-style attacks, per Task 1.2, manipulate *content* — positions,
  speeds — not the receipt ordering.)

## Check 3 — Inter-message gaps (`dt`): **PASS**

Method: read the `dt` feature column directly from `X_windows.npy` (50,000-window sample, 1,000,000 steps) —
this column is stored **raw** (seconds), not normalized, confirmed by comparing its sample mean/std (587.7 /
387.8 for `pos_x`, as a cross-check) against `config.json`'s `norm_mean`/`norm_std` (591.1 / 421.4 — matches
within sampling noise). Beacon interval and gap threshold are derived from the data itself, not assumed.

- Typical inter-message interval: **median 1.0 s**, mean 0.80 s, p99 1.0 s, max 156.0 s (VeReMi's default ~1 Hz
  BSM broadcast rate).
- Gap threshold = 3× median = 3.0 s. **2,119 / 1,000,000 steps (0.21%)** exceed it — consistent with ordinary
  broadcast jitter and occasional real network gaps, not a systemic defect.
- **`dt ≤ 0`: 4,371 steps, all exactly `dt == 0`, all at timestep index 0 of every window with zero exceptions**
  (verified per-column: indices 1–19 have zero occurrences). This is not a data gap — a window's first step has
  no "previous step" inside that window to difference against, so `dt` is a deterministic sentinel there, not a
  raw-data timing defect. Downstream modeling should be aware that column-0 `dt` carries no signal.

## Check 4 — Coordinate-frame sanity: **PASS (with an expected attacker-only exception)**

Method: read `pos_x`/`pos_y` directly from the same 50,000-window sample (raw meters, per the check-3
finding), split by `is_attacker` from `window_metadata.parquet`, and cross-check `pos[2]` (z) on 2,286 raw
messages across 3 sampled trace files.

- **Benign windows**: `pos_x ∈ [24.7, 1396.1]` m, `pos_y ∈ [24.3, 1359.6]` m — bounded, positive, consistent
  with a local-frame SUMO planar map at Luxembourg-city scale (the sampled sub-region is on the order of 1.4 km
  × 1.4 km; the full LuST map extent is larger but this range shows no runaway values).
- **z-coordinate**: `0` in all 2,286 checked messages — confirms a 2-D planar simulation, no 3-D artifact.
- **Attacker windows** (GridSybil, DoS*Sybil, DataReplaySybil): `pos_x ∈ [-8868.4, 6569.0]` m,
  `pos_y ∈ [-9625.3, 9593.4]` m — an order of magnitude beyond the benign extent, including negative
  coordinates that fall off any plausible road network.
- **This is the expected attack signature, not a data-quality defect.** GridSybil fabricates ghost positions
  on an arbitrary grid unconstrained by road geometry; the other Sybil families forge or replay positions with
  similar disregard for physical plausibility. The extreme, out-of-bounds coordinates *are* the discriminative
  signal Task 1.2's attack-taxonomy write-up should point to, and a strong candidate feature for the
  physical-plausibility columns proposed in Phase 2 (task 2.3).

## Summary

| Check | Verdict | Anomaly count |
|---|---|---|
| 1. Duplicate messages | ✅ PASS | 0 / 21,306 type-3 messages |
| 2. Non-monotonic sendTime | ✅ PASS | 0 / 793 sender sequences |
| 3. Inter-message gaps (dt) | ✅ PASS | 2,119 / 1,000,000 steps >3×median (0.21%, expected jitter); 4,371 `dt==0` all explained as the column-0 sentinel |
| 4. Coordinate-frame sanity | ✅ PASS | 0 anomalies for benign; attacker windows extend far beyond the benign extent **by design** (attack signature, not a defect) |

No raw-data quality defects were found in the sampled files or the prepared windows. The one artifact worth
carrying forward into feature engineering (Phase 2) is the `dt == 0` sentinel at every window's first timestep
— models should not be misled into treating it as a real zero-gap observation.
