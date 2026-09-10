# Split Protocol — Leakage Audit (Plan Task 1.5)

**Status: all audit checks pass.** This document describes how the committed splits in
`data/prepared_data/` were built, and records the output of `scripts/audit_splits.py`, which
re-verifies the leakage guarantees independently of the notebook that produced them.

## Why a dedicated grouping key

Windows are built with a **sliding window of T=20 timesteps at stride 10** per sender
(`notebooks/eda_veremi.ipynb`, "Data Preparation Methodology"), so consecutive windows from the
same sender overlap by 10 timesteps. A random window-level split would put near-duplicate windows
on both sides of train/test — the model would partly memorize rather than generalize. Every split
in this project is therefore assigned **before windowing, at the sender level**.

The grouping key is **`sender_uid = family + "_" + group + "_" + subfolder + "_" + senderPseudo`**
— e.g. `DataReplaySybil_0709_VeReMi_25200_28800_2022-9-13_21:7:46_10216212`. This scopes a sender
pseudonym to the exact scenario file it came from, which matters because VeReMi/SUMO can reuse
`senderPseudo` integers across independent scenario runs — `sender_uid` treats those as distinct
identities, which is the conservative and correct choice for splitting.

## Two independent partitions, same window pool

The 285,926 windows are partitioned two different ways for two different purposes; both partitions
cover every window, and both are built by grouping on senders first:

1. **`idx_train` / `idx_val` / `idx_test`** (188,647 / 43,491 / 53,788 — 66.0% / 15.2% / 18.8%) — the
   main experiment split, used for all of Phases 3–6 except cross-scenario transfer. Built with two
   chained `sklearn.model_selection.GroupShuffleSplit` calls (`random_state=42`) on `sender_uid`:
   first 80/20 into train+val vs. test, then 80/20 of train+val into train vs. val. Every sender
   appears in exactly one of the three splits.
2. **`idx_group_0709` / `idx_group_1416`** (195,551 / 90,375 — 68.4% / 31.6%) — a **scenario-holdout**
   partition by the `group` field alone (SUMO time-of-day segment: 0709 = 07:00–09:00,
   1416 = 14:00–16:00), used for the Phase 5 cross-scenario transfer protocol (train on one group,
   test on the other). Because `group` is baked into `sender_uid`, this partition is automatically
   sender-disjoint too — verified below, not merely assumed.

These two partitions are **independent axes over the same windows**, not nested: a given window's
train/val/test membership and its 0709/1416 membership are orthogonal facts about it.

## Normalization

Per-feature mean/std (`norm_mean.npy`, `norm_std.npy`) are computed from `X[idx_train]` flattened to
`(N_train × T, D)` — **train split only**, never touching val or test. This satisfies plan task 2.2's
acceptance criterion and is re-verified below by recomputing the stats from scratch and diffing
against the committed `.npy` files.

## Audit script and its findings

`scripts/audit_splits.py` is read-only against `data/`, exits non-zero on any failure, and checks
five things. Full output from the current committed data:

```
==============================================================================
1. TRAIN/VAL/TEST — index-level partition
==============================================================================
[PASS] train/val index-disjoint
[PASS] train/test index-disjoint
[PASS] val/test index-disjoint
[PASS] train+val+test covers all windows — union=285,926 vs n_windows=285,926
    sizes: train=188,647 val=43,491 test=53,788

==============================================================================
2. TRAIN/VAL/TEST — sender_uid leakage (the core Task 1.5 check)
==============================================================================
[PASS] train/val sender_uid overlap — 0 shared senders
[PASS] train/test sender_uid overlap — 0 shared senders
[PASS] val/test sender_uid overlap — 0 shared senders
    unique senders: train=16,124 val=4,031 test=5,039

==============================================================================
3. GROUP 0709 / 1416 — scenario-holdout partition
==============================================================================
[PASS] group_0709/1416 index-disjoint
[PASS] group_0709+group_1416 covers all windows — union=285,926 vs n_windows=285,926
[PASS] group_0709/1416 sender_uid overlap — 0 shared sender_uid (expected 0 — group is baked into sender_uid)

==============================================================================
4. GROUP 0709 / 1416 — raw senderPseudo overlap (informational)
==============================================================================
[INFO] raw senderPseudo overlap: 0 of 11,777 distinct ids (0.0%) appear in both groups — not a
leakage failure by construction (sender_uid disambiguates by group+subfolder), but confirms
senderPseudo integers are recycled by SUMO across separate scenario runs, so senderPseudo alone
must never be used as the grouping key for any future split.

==============================================================================
5. NORMALIZATION — recomputed from train split only
==============================================================================
[PASS] norm_mean.npy matches train-only recomputation
[PASS] norm_std.npy matches train-only recomputation

==============================================================================
RESULT: ALL CHECKS PASSED
==============================================================================
```

**Note on check 4:** the raw `senderPseudo` integer happens to be 0% overlapping between the 0709
and 1416 groups in this dataset (11,777 distinct ids, none shared) — i.e. VeReMi did not, in
practice, recycle pseudonym integers across these particular scenario runs. That is a property of
this specific dataset build, not a guarantee; the `sender_uid` scheme is what makes the split
correct regardless, and any new data ingestion must not assume `senderPseudo` alone is safe to
group on.

## Verdict for Task 1.5

**PASS.** The committed splits satisfy the acceptance criterion: overlapping windows from one
sender never straddle a split boundary (verified at the `sender_uid` level, both for
train/val/test and for the 0709/1416 scenario holdout), and normalization statistics are
demonstrably train-only. Re-run `python scripts/audit_splits.py` after any change to
`data/prepared_data/` to re-verify — it is fast (a few seconds) and exits non-zero on any leak.

## What this does not cover

- **Attack-family leakage across cross-scenario transfer**: 0709 and 1416 are disjoint by
  construction, but this says nothing about whether the *attack behavior* looks different enough
  between the two time-of-day segments to make transfer meaningful — that is an empirical Phase 5
  question, not a leakage question.
- **Message-level duplication within a sender's own windows** (the stride-10 overlap): this is
  expected and intentional (it is why sender-level splitting exists), not a defect to fix.
