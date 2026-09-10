# VeReMi-Extension Dataset Structure

Task 1.1 write-up. Everything below was verified directly against `data/VeReMi-Dataset/` (raw, gitignored,
13 GB) and cross-checked against the parsing logic in `notebooks/eda_veremi.ipynb`. Where something is
ambiguous or unverified, it is stated as such rather than guessed.

## 1. Directory layout

```
data/VeReMi-Dataset/
├── DataReplaySybil_0709/        ← {Family}Sybil_{Group}
│   ├── VeReMi_25200_28800_2022-9-13_21:7:46/     ← one simulation RUN (7:00–8:00am)
│   │   ├── traceJSON-9-1092-A0-25200-7.json      ← one file per RECEIVING vehicle
│   │   ├── traceJSON-21615-21613-A17-28800-8.json
│   │   ├── ...  (~1,000–2,200 files per run)
│   │   └── traceGroundTruthJSON-7.json           ← ONE file, all vehicles' true positions
│   └── VeReMi_28800_32400_2022-9-13_21:7:46/     ← next hour (8:00–9:00am)
├── DataReplaySybil_1416/
│   ├── VeReMi_50400_54000_.../                    ← 14:00–15:00
│   ├── VeReMi_54000_57600_.../                    ← 15:00–16:00
│   └── VeReMi_57600_61200_.../                    ← 16:00–17:00
├── DoSDisruptiveSybil_0709/   DoSDisruptiveSybil_1416/
├── DoSRandomSybil_0709/       DoSRandomSybil_1416/
└── GridSybil_0709/            GridSybil_1416/
```

8 top-level folders = 4 attack families × 2 time-of-day groups. **`_0709`** runs cover 07:00–09:00
(two 1-hour chunks: `25200_28800`, `28800_32400` — seconds since simulator midnight, `25200 s = 7:00`).
**`_1416`** runs cover 14:00–17:00 (three 1-hour chunks: `50400_54000`, `54000_57600`, `57600_61200`).
The `_1416` label spans one more hour than its name suggests (14→17, not 14→16) — recorded as observed,
not resolved further; not worth chasing since `group` is only ever used as a categorical split key (0709
vs 1416), never parsed for its literal hour range.

## 2. The two file types

### `traceJSON-{nodeId}-{pseudoId}-A{attackType}-{startTime}-{simNum}.json`

One file per vehicle **as a receiver** — it logs that vehicle's own GPS readings (`type: 2`, no `sender`
field) interleaved with Basic Safety Messages (BSMs) it *received* from every neighbor it heard
(`type: 3`, with a `sender`/`senderPseudo` identifying who broadcast it). Verified on a real file: node
`21615`'s own trace file contains messages from 23 distinct `sender` vehicles, broadcasting under 33
distinct `senderPseudo` values — i.e. more pseudonyms than physical senders, which is the Sybil signature
itself (one attacker, several fake identities visible to a listener).

Filename fields, decoded via `parse_trace_filename()` in the notebook and confirmed against real files:

| Field | Meaning | Verified example |
|---|---|---|
| `nodeId` | the vehicle's real SUMO id (matches `sender` in ground truth) | `21615` |
| `pseudoId` | a pseudonym associated with this node at file-creation time | `21613` |
| `A{attackType}` | attacker-type code, **0 = benign** | `A16` |
| `startTime` | run start second (matches the parent folder name) | `28800` |
| `simNum` | run/simulation id (matches the `-{simNum}.json` suffix on the GT file in the same folder) | `8` |

Attack-type codes (from `ATTACK_MAP` in the notebook, cross-checked against folder names):

| Code | Family | int label (config.json) |
|---|---|---|
| `A0` | Benign | 0 |
| `A17` | DataReplaySybil | 1 |
| `A19` | DoSDisruptiveSybil | 2 |
| `A18` | DoSRandomSybil | 3 |
| `A16` | GridSybil | 4 |

(Note the deliberate mismatch between attack-code order and int-label order — the int labels come from
alphabetically sorting `attack_label` strings in the notebook, not from the numeric attack codes.)

### `traceGroundTruthJSON-{simNum}.json`

One file per run, all vehicles combined, every line `type: 4`. This is the **true** position/kinematics
record for every vehicle — attackers' fabricated broadcast identities (the `senderPseudo` values seen in
*other* vehicles' `traceJSON` receive-logs) do **not** appear here. GT records only ever contain a
vehicle's own `sender` id.

**Important, verified anomaly:** in the sampled `GridSybil_0709` run, GT `senderPseudo` is almost-always
node-specific *except* for the sentinel value `1`, which 653 distinct GridSybil attacker `sender` ids
(all `A16`) share in that one subfolder — none of them collide with a different attack label, but the
current pipeline's window-building step groups by `senderPseudo` (not by `sender`/`node_id`), so all 653
GridSybil vehicles that reported under GT-pseudonym `1` would be merged into a single sender-group before
sliding-window construction. This looks like a labeling artifact specific to how VeReMi's GridSybil attacker
model writes its own ground-truth pseudonym field (not observed in a `DataReplaySybil` folder checked for
comparison — there, 2,221 senderPseudo values were all node-unique). **Flagging this explicitly for Task
1.5 (leakage audit) and Task 1.7 (data-quality checks) — it was outside this task's scope to fix, but it
should be verified and (if confirmed) either patched in the windowing groupby key (group by `sender`, not
`senderPseudo`) or documented as an accepted, bounded artifact before citing GridSybil window counts.**

## 3. How an attacker label attaches to a message

Confirmed directly in the notebook (cell comment, verified 100% overlap): **`GT.sender == filename's
`nodeId``**, scoped to `(family, group, subfolder)` — i.e. the same physical node id can be benign in one
run and an attacker in another, so the label lookup key is always the four-tuple
`(family, group, subfolder, node_id)`, not `node_id` alone.

## 4. Worked example — one message → one row → one window

**Raw file:** `data/VeReMi-Dataset/DataReplaySybil_0709/VeReMi_25200_28800_2022-9-13_21:7:46/traceJSON-9-1092-A0-25200-7.json`
— node 9, pseudo 1092, `A0` (benign).

**Ground truth line for that sender** (`traceGroundTruthJSON-7.json`, `sender: 9`, `senderPseudo: 1092`):
```json
{"type": 4, "sendTime": 25202.602763370374, "sender": 9, "senderPseudo": 1092,
 "pos": [265.6337750163457, 45.99272548997644, 0.0],
 "spd": [-0.050542505663603604, 0.4891161266190862, 0.0], ...}
```

**Pipeline (`notebooks/eda_veremi.ipynb`):**
1. `load_ground_truth()` flattens `pos`/`spd`/`acl`/`hed` into `pos_x, pos_y, spd_x, spd_y, ...` columns → `df_gt`.
2. The label merge (§3 above) attaches `attack_code=0`, `attack_label="Benign"`, `is_attacker=False` to every row with `sender=9` in this subfolder.
3. Rows are sorted by `(family, group, subfolder, senderPseudo, sendTime)`; `dt`, `dpos_x/y`, `dspd_x/y` are computed as within-sender diffs (first row of a sequence gets `0.0`).
4. A sliding window of `T=20` steps, `stride=10`, is cut from this sender's sequence. The **first** window (`window_start_idx=0`) for `senderPseudo=1092` starts at `sendTime=25202.602763` — this is exactly row 0 of `data/prepared_data/window_metadata.parquet`.
5. That window's raw feature matrix is `X_windows.npy[0]`, shape `(20, 13)`. Verified: `X[0,0] = [265.63, 45.99, -0.0505, 0.489, ...]` matches the GT line above; `X[0,1]` shows `dt=1.0` (next message one second later) and `dpos_x=0.0599, dpos_y=1.709` (position delta), confirming the delta-feature computation.
6. Label: `y_multiclass[0] = label_to_int["Benign"]`, `y_binary[0] = 0`.

## 5. Cookbook — locating any scenario's data

To find "GridSybil attack, morning scenario (7–9am), the 8:00–9:00 run":
```
data/VeReMi-Dataset/GridSybil_0709/VeReMi_28800_32400_<date>_<time>/
```
Inside that folder: every `traceJSON-*-A16-*.json` file is an attacker's own receive-log; every
`traceJSON-*-A0-*.json` is benign; `traceGroundTruthJSON-*.json` is the one true-position file for the
whole run. To find which windows in the prepared dataset came from this run, filter
`window_metadata.parquet` on `family == "GridSybil"`, `group == "0709"`,
`subfolder == "VeReMi_28800_32400_..."`.

## Open items for other tasks

- **1.5 / 1.7:** verify and resolve the GridSybil `senderPseudo`-collision artifact (§2) before citing
  per-class window counts as final.
- **1.3:** this doc covers `pos/spd/acl/hed` at the structural level; full units/noise-variant semantics
  are Task 1.3's job (the `*_noise` fields exist alongside every main field but are unused by the current
  pipeline — see `X_windows.npy` columns, which are all non-`_noise`).
