# VeReMi Message Field Reference

Task 1.3 of `docs/plan/research-plan.md`. Verified empirically against raw trace files under
`data/VeReMi-Dataset/{Family}_{Group}/VeReMi_<start>_<end>_<timestamp>/` — specifically
`GridSybil_0709/VeReMi_28800_32400_2025-11-15_13:57:9/` (a per-vehicle `traceJSON-*.json` and the
scenario's `traceGroundTruthJSON-*.json`) — by loading JSON lines directly with Python, cross-referencing
records by `messageID`, and computing observed value ranges. No file under `data/` was modified.

## Message types

| `type` | Meaning (verified) | Present fields |
|---|---|---|
| `2` | The vehicle's own GPS/state reading (its belief about itself) — no `sender`/`senderPseudo`/`messageID` | `rcvTime`, `pos(_noise)`, `spd(_noise)`, `acl(_noise)`, `hed(_noise)` |
| `3` | A received Basic Safety Message (BSM) broadcast from another vehicle — carries sender identity | all of the above **plus** `sendTime`, `sender`, `senderPseudo`, `messageID` |
| `4` | A ground-truth log entry (in `traceGroundTruthJSON-*.json`) — the scenario-wide message ledger, one entry per BSM sent by any vehicle | same shape as type `3` |

`traceGroundTruthJSON-*.json` is **not** a separately-simulated "true state" file — for a given
`messageID`, its `pos`/`pos_noise`/etc. are byte-identical to the value carried in the per-vehicle
trace that received it (verified: diff = `[0.0, 0.0]` on a matched record). It is best understood as
the scenario's central message log used to derive ground-truth attacker labels, not an independent
noise-free reference.

## Field-by-field reference

| Field | Unit / type | Coordinate frame / observed range | Noise-variant relationship | In `feature_cols`? |
|---|---|---|---|---|
| `sendTime` | seconds (SUMO simulation clock) | scenario-relative; matches the folder's `<start>_<end>` window (e.g. `28800`–`32400`) | n/a | derived into `dt` (inter-message gap) |
| `rcvTime` | seconds | equals `sendTime` in the sampled records (0.0 diff) — no simulated propagation delay observed in this scenario | n/a | not used directly; `sendTime` drives ordering |
| `sender` | integer vehicle ID | the underlying SUMO vehicle instance — stable per physical vehicle for the scenario | n/a | not a model input; used only for split-group assignment and attack-signature analysis |
| `senderPseudo` | integer pseudonym | **the actual identity as broadcast** — this is what a receiver / detector sees, not `sender` | n/a | drives per-sender windowing (`config.json`'s implicit sender key is `senderPseudo`) |
| `messageID` | integer | unique per broadcast message | n/a | used only to join trace ↔ ground truth |
| `pos` = `[pos_x, pos_y, pos_z]` | meters, planar (SUMO/LuST local projection, not lat/lon) | `pos_z` is `0.0` in every sampled record (2D simulation); `pos_x`/`pos_y` range roughly −3.6k…+2.8k m across the full scenario (~6–7 km span) — consistent with the LuST (Luxembourg SUMO Traffic) map extent, not a lat/lon-derived frame | `pos_noise` is a small-magnitude 2-vector (observed ~4–6, i.e. a few meters) riding alongside `pos` in every message — see noise discussion below | `pos_x`, `pos_y` ✅ |
| `spd` = `[spd_x, spd_y, spd_z]` | m/s | `spd_z` = 0; `spd_x`/`spd_y` observed in roughly ±10 m/s (urban driving speeds) | `spd_noise` present alongside, same shape | `spd_x`, `spd_y` ✅ |
| `acl` = `[acl_x, acl_y, acl_z]` | m/s² | `acl_z` = 0 | `acl_noise` present alongside | `acl_x`, `acl_y` ✅ |
| `hed` = `[hed_x, hed_y, hed_z]` | unit heading vector | **verified unit vector**: `hed_x² + hed_y² = 1.0` in every sampled record; `hed_z` = 0 | `hed_noise` present but its values (observed ~10–50) are far too large to be an additive perturbation of a unit vector — see below | `hed_x`, `hed_y` ✅ |
| `dt` | seconds (derived, not a raw field) | inter-message time gap for a given sender's window | n/a | ✅ (derived) |
| `dpos_x`, `dpos_y` | meters (derived) | per-step position delta | n/a | ✅ (derived) |
| `dspd_x`, `dspd_y` | m/s (derived) | per-step velocity delta | n/a | ✅ (derived) |
| `pos_noise`, `spd_noise`, `acl_noise`, `hed_noise` | same units as their base field | present in every message, main and noise fields always co-occur | see below | ❌ not in `feature_cols` |

## What the `*_noise` fields actually are (verified, with one caveat)

For a matched `messageID` between a vehicle's trace and the scenario ground-truth log, `pos` and
`pos_noise` are **identical** in both files — the ground-truth log is not a denoised reference. Given
VeReMi's published methodology (position/speed/acceleration perturbed by a configurable GPS-like noise
model before broadcast), the most consistent reading is:

- `pos`, `spd`, `acl` are the **already-noised, broadcast values** — i.e., what the sending vehicle
  actually puts on the air and what every consumer (other vehicles, our model) sees.
- `pos_noise`, `spd_noise`, `acl_noise` are a **companion diagnostic channel from the simulator**
  recording the noise magnitude applied at that step (their scale — a few meters for position, small
  fractions of a m/s for speed — matches a plausible GPS/IMU noise budget, not the raw signal itself).
- `hed_noise`'s much larger values (~10–50 in the observed units, incompatible with perturbing a unit
  vector directly) suggest it is likely an **angular** quantity (e.g. degrees of heading error) rather
  than a Cartesian noise vector matching `hed`'s `[x, y]` shape. **This is inferred, not directly
  confirmed** — the VeReMi format documentation should be checked against the original paper before this
  is stated as fact in the thesis; flagging it here as an open item rather than asserting it.

## Decision: main fields, not noise variants

`data/prepared_data/config.json`'s 13 `feature_cols` (`pos_x, pos_y, spd_x, spd_y, acl_x, acl_y, hed_x,
hed_y, dt, dpos_x, dpos_y, dspd_x, dspd_y`) use the **main** kinematic fields exclusively — no `*_noise`
column is included.

**Rationale:** the main fields are what a real receiver (RSU, neighboring vehicle, or our detector)
actually observes on the wire — they already carry the simulator's injected sensor noise baked in, so
training on them matches the deployed detection setting. The `*_noise` channels are a simulator-internal
diagnostic that a real-world detector would never have access to (you don't get told "here is exactly how
much noise was added to this GPS fix"); using them as model inputs would be a form of information leakage
from privileged simulator state. They are kept in reserve, per the plan's "Known risks" note on simulator
regularity (§ Phase 1), as a potential **robustness/ablation check** — e.g. adding extra synthetic noise on
top of the main fields, calibrated against the `*_noise` channel's observed magnitude, to test whether the
model over-relies on the simulator's characteristic smoothness.

## Bonus finding: this *is* the Sybil attack signature, directly observable in `senderPseudo`

Grouping the ground-truth log by `sender` and collecting the distinct `senderPseudo` values used, on the
`GridSybil_0709` scenario sampled: **1,557 senders used exactly one pseudonym** (benign-looking single
identity) while **476 senders broadcast under 6–7 distinct `senderPseudo` values simultaneously**
(example: `sender 21633` → pseudonyms `{1, 10216332, 20216332, 30216332, 40216332, 50216332, 60216332}`).
This is the ground-truth Sybil mechanism made concrete: one physical vehicle (`sender`) fabricates multiple
broadcast identities (`senderPseudo`). Windowing by `senderPseudo` (as the pipeline does) therefore
produces multiple *distinct pseudonym-sequences* for a single attacking vehicle — worth stating explicitly
in the attack-taxonomy write-up (task 1.2) and double-checking during the leakage audit (task 1.5): a
`sender`-level split (not just `senderPseudo`-level) may be the more conservative leakage boundary, since
several pseudonym-windows trace back to the same underlying `sender`.
