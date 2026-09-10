# GridSybil_0709 windowing defect — spliced-vehicle windows

**Status: confirmed, quantified, unresolved.** Surfaced during Phase 1 parallel investigation (tasks 1.1 and 1.3
independently flagged pieces of this; verified and quantified directly against `data/prepared_data/window_metadata.parquet`
and the raw ground truth on 2026-09-10). This is a real defect in the windowing pipeline, not a documentation gap —
it belongs in Phase 2 (task 2.6, `scripts/prepare_data.py`) as a fix, not just a note.

## The bug

`notebooks/eda_veremi.ipynb` builds sliding windows by grouping the ground-truth dataframe on
`['family', 'group', 'subfolder', 'senderPseudo']`, sorting each group by `sendTime`, computing `dt`/`dpos`/`dspd`
as diffs *within* the group, then slicing 20-step windows from the sorted sequence. This assumes `senderPseudo`
uniquely identifies one vehicle's message sequence within a subfolder.

That assumption is verified false for one specific run. In the ground truth file for
`GridSybil_0709/VeReMi_28800_32400_.../traceGroundTruthJSON-8.json`:

```
type-4 records scanned:            415,223
records with senderPseudo == 1:     71,767
distinct raw `sender` (vehicle) ids under senderPseudo == 1:  653
```

`senderPseudo == 1` is a sentinel/default value shared by 653 different physical vehicles in this run — not a real
pseudonym. The windowing code's `groupby(...senderPseudo...)` therefore merges 71,767 ground-truth messages from 653
unrelated vehicles into one "sequence," sorts them purely by `sendTime`, and slices 20-step windows out of the
result. Consecutive rows in such a window can belong to different physical vehicles, so:

- `dt`, `dpos_x/y`, `dspd_x/y` (computed as diffs within the group) are frequently **garbage** — a jump between two
  unrelated vehicles' positions/speeds at nearly the same timestamp, not a real kinematic transition.
- The window's label (`attack_label = grp['attack_label'].iloc[0]`) is taken from whichever vehicle happens to sort
  first — in this run all 653 vehicles are GridSybil attackers so the label is *accidentally* correct, but the
  mechanism generalizes badly and would mislabel a mixed group.

Label *attachment* itself is fine — `attack_code`/`attack_label`/`is_attacker` are looked up correctly via
`GT.sender == filename.node_id` (verified 100% overlap, see task 1.1). The defect is specifically in the **windowing
group key**, which should include `sender` (or `node_id`), not rely on `senderPseudo` alone.

## Scope — precisely quantified

```
Total windows in data/prepared_data/:  285,926
Affected windows (senderPseudo==1):     14,764   (5.16% of all windows)
Affected family/group:                  GridSybil, group 0709 ONLY — both subfolders
                                         (VeReMi_25200_28800: 7,589 windows; VeReMi_28800_32400: 7,175 windows)
GridSybil_1416 windows:                 NOT affected — largest sender-groups there are 100-190 windows,
                                         consistent with single-vehicle trajectory lengths, not a merged mass
```

No other family/group combination shows this pattern (checked: the next-largest sender-groups by window count are
all in the 100-190 range across every family/group, i.e. ordinary long individual trajectories).

## What this means right now

- **Task 1.4 (class balance)** numbers for GridSybil/0709 include 14,764 corrupted windows. The window *count* is
  real, but a meaningful fraction of GridSybil_0709's kinematic features (`dt`, `dpos_x/y`, `dspd_x/y` especially)
  are not physically meaningful.
- **Task 1.2 (attack taxonomy)** — the GridSybil example trajectory in `attack_trajectory_examples.png` was drawn
  from a per-vehicle raw trace file directly, not from a spliced window, so that finding is unaffected.
- **Task 1.5 (leakage audit)** — unaffected for its actual purpose (sender-level disjointness across splits still
  holds, since `sender_uid` is built from the same `senderPseudo` field consistently across train/val/test). But it
  means ~5% of the *training and evaluation data itself* — wherever this `sender_uid` group lands — is spliced
  garbage, independent of which split it's in.
- **Task 1.7 (data quality)** — the committed checks (duplicates, monotonic `sendTime` *within* a sampled per-vehicle
  raw file, `dt` gap histogram, coordinate bounds) do not catch this, because each check operated either on raw
  per-vehicle files (where this problem doesn't exist — it's an artifact of the windowing groupby) or on aggregate
  `dt` statistics that don't isolate one corrupted `sender_uid` group's internal discontinuity. This is a legitimate
  gap in the 1.7 checklist, not an error in what was checked — it needs a fifth check: **per-window position
  continuity** (e.g. max per-step displacement vs. plausible top speed × dt) that would catch exactly this.

## Fix required (blocks Phase 2 task 2.6)

`scripts/prepare_data.py` (task 2.6, not yet written) must group by a key that uniquely identifies one physical
vehicle's message sequence — e.g. `(family, group, subfolder, sender)` using the raw `sender`/`node_id` (which the
label lookup already proves is 100% reliable), not `(..., senderPseudo)`. Whether to also regenerate the currently
committed `data/prepared_data/X_windows.npy` before further pretraining experiments is a call for whoever owns
Phase 2 — flagging here rather than deciding unilaterally, since `data/` writes require confirmation (CLAUDE.md
rule). At minimum, any v1 result trained on the current windows should not be cited as clean going forward without
this caveat attached.

## Suggested new task 1.7b (or fold into 2.6's acceptance criteria)

Add a per-window continuity check: for every window, flag if any consecutive-step displacement implies a speed
exceeding a generous upper bound (e.g. 60 m/s, well above any plausible urban vehicle speed) — this would have
caught the GridSybil_0709 splice directly and should be added to the reproducible pipeline's QA gate before
`X_windows.npy` is ever regenerated.
