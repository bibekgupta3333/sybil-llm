# Cross-Scenario Leak Audit (WBS S1.1.1, finding F1)

**Status: leak confirmed on the v1 split.** 89.7% of test identities (2,283 / 2,546 benign;
3,291 / 3,670 overall) have a near-copy of the same vehicle in train. Every number computed on the
v1 `idx_{train,val,test}.npy` split, including everything in `models/results/`, is optimistic and
must not be quoted as a result. The fix is the vehicle-grouped split of S1.1.5 (decision D8).

## What F1 is

The four Sybil scenarios of one group (`DataReplaySybil`, `DoSDisruptiveSybil`, `DoSRandomSybil`,
`GridSybil` × `_0709` / `_1416`) are four runs over **the same SUMO traffic**: the same vehicles drive
the same routes in the same time windows; only the attack differs. So a vehicle that appears in
`DoSRandomSybil_1416/VeReMi_50400_54000_*` appears again, with the same `senderPseudo` and a
trajectory within ~0.2 m, in the other three scenario folders of that group.

The v1 split (`split-protocol.md`) grouped by `sender_uid` = family + group + subfolder +
pseudonym. That is sender-disjoint *within* one scenario folder, but it treats the four copies of
one vehicle as four unrelated identities, so they are split independently and ~90% of test
vehicles have a copy in train.

## Method

* **Key** (v1): `(group, time_window, senderPseudo)`, with `time_window` = `VeReMi_<start>_<end>`
  parsed from `subfolder` (the date suffix differs between scenario folders, the window does not).
  v1 metadata has no physical vehicle id, so the pseudonym stands in for it (the S1.1.5 split uses
  the physical id instead).
* **Identity**: `sender_uid` (one per scenario folder × pseudonym). An eval identity *has a copy*
  when its key occurs on any train row.
* **Exclusion**: `GridSybil_0709` (66,045 windows) is dropped from both sides. Its
  `senderPseudo == 1` is a sentinel shared by hundreds of physical vehicles (splice defect,
  `gridsybil-windowing-defect.md`), so its pseudonym keys are not vehicle keys. GridSybil numbers
  below are therefore GridSybil_1416 only.
* **Same class / same binary**: whether the train copy carries the identical class name, or only
  the same benign-vs-attack side.
* **Position check**: for 300 seeded (seed 42) eval identities with a copy, the identity's earliest
  window is paired with the train window of the same key whose `start_time` is closest; the pair
  distance is the median over the 20 timesteps of the `(pos_x, pos_y)` gap (v1 `X_windows.npy`
  columns 0–1, metres, unnormalised).

Code: `src/split.py::LeakAudit` (unit-tested on a hand-built table in
`src/tests/test_splits.py`). Read-only against `data/prepared_data/`; writes nothing.

## Results (v1 split, GridSybil_0709 excluded)

| Split | Class | Identities | With train copy | Share | Copy same class | Copy same binary | Median copy distance (p90) |
|---|---|---:|---:|---:|---:|---:|---|
| test | Benign | 2,546 | 2,283 | **89.7%** | 2,283 | 2,283 | 0.21 m (0.42 m) |
| test | DataReplaySybil | 339 | 310 | 91.4% | 0 | 310 | 59.2 m (70.7 m) |
| test | DoSDisruptiveSybil | 348 | 304 | 87.4% | 0 | 304 | 0.22 m (61.4 m) |
| test | DoSRandomSybil | 341 | 303 | 88.9% | 0 | 303 | 0.22 m (61.8 m) |
| test | GridSybil (1416) | 96 | 91 | 94.8% | 0 | 91 | 21.4 m (80.9 m) |
| test | **All** | 3,670 | 3,291 | **89.7%** | 2,283 | 3,291 | 0.24 m (41.0 m) |
| val | Benign | 2,000 | 1,813 | 90.7% | 1,813 | 1,813 | 0.21 m (0.49 m) |
| val | DataReplaySybil | 249 | 219 | 88.0% | 0 | 219 | 57.8 m (68.9 m) |
| val | DoSDisruptiveSybil | 290 | 269 | 92.8% | 0 | 269 | 0.21 m (62.1 m) |
| val | DoSRandomSybil | 273 | 246 | 90.1% | 0 | 246 | 0.20 m (61.8 m) |
| val | GridSybil (1416) | 75 | 75 | 100.0% | 0 | 75 | 24.8 m (83.8 m) |
| val | **All** | 2,887 | 2,622 | **90.8%** | 1,813 | 2,622 | 0.23 m (52.9 m) |

Benign copies: max distance 1.37 m (test) / 1.20 m (val) over the 300 pairs; median start-time gap
0.04 s. (The F1 row in `stage1-ssl-wbs.md` quotes max 1.0 m from an earlier, different 300-pair
sample; the median 0.21 m reproduces.)

### What the attack rows mean

* **Attackers are the same vehicles in every scenario.** Every attack identity with a copy has a
  train copy that is *also an attacker* — of a different family (same binary label, never the same
  class). The binary benign/attack task therefore leaks for attackers as much as for benign
  vehicles; the 5-class task sees a copy with a *different* attack label.
* For DoS attackers the median copy is ~0.2 m away, but p90 ~61 m: a minority of paired windows
  have positions that differ between the two families (not investigated further here).
  DataReplay (~59 m) and GridSybil (~21 m) copies differ more because the replayed / ghost positions
  differ between scenarios — but the *identity* (same vehicle, window, pseudonym) still leaks.

## Consequences for v1 results

* All v1 numbers (`models/results/`, `models/roadfm_lite_final.pt` evaluations, few-shot,
  zero-shot and cross-scenario transfer within a group) were measured on a test set where ~90% of
  identities have a ~0.2 m near-copy in train. They overstate generalisation and are invalid until
  re-run on the new split (RULE 3; `src/agent.md` §4 "treat all v1 numbers as invalid").
* Cross-group transfer (0709 ↔ 1416) is not affected by F1 itself (different traffic), but it used
  the same per-scenario identity grouping and will be re-run on the new split anyway.

## How S1.1.5 fixes it

`src/split.py::VehicleSplitter` assigns whole groups `f"{group}:{time_window}:{sender}"` —
`sender` = the true physical vehicle id from the trace filename (grouping key only, never an input)
— pooled over the 4 scenarios, to exactly one of train / val / test (70 / 15 / 15% of groups,
seed 42), then carves `pretrain_val` (10% of train groups) out of train. It asserts with
`SplitGuard` zero group overlap across all four splits and across 0709/1416, and that every row is
assigned exactly once. Running this audit on the new split must give 0 copies in every row (the
unit test `test_contract_key_and_group_split_has_no_leak` checks exactly that on synthetic data).
**This is a RULE 3 split change** (decision D8): it is recorded in `docs/plan/research-plan.md`
when the `idx_*_rx.npy` files are produced.

## Reproduce

> **Note (2026-10-06):** the code that produced these numbers (`src/split.py`, `python -m src.prepare audit`)
> was removed with the rest of the S1.1 implementation at the user's request. The numbers below stand as
> recorded; the commands are kept for when the audit is re-implemented.

From the repo root (read-only; ~1 s):

```bash
.venv/bin/python -m src.prepare audit          # CLI (S1.1.1 subcommand)
```

```bash
.venv/bin/python -c "
from src.split import LeakAudit
print(LeakAudit.for_v1('data/prepared_data').run(['test', 'val']).to_string(index=False))
"
```

(The copy-distance numbers in the table above came from a one-off position check during the audit; that
helper was removed in the 2026-10-06 code simplification because nothing else used it.)
