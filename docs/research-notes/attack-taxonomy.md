# VeReMi Sybil Attack Taxonomy (Task 1.2)

Characterizes the 4 Sybil attack families in VeReMi-Extension against benign behavior, so that "what
signal should the model find?" has a concrete, per-family answer. Figure:
`results/figures/eda/attack_trajectory_examples.png`.

**Verification method.** Attacker-type codes were confirmed directly from the local dataset (not just
literature) by listing filenames in one run folder per family and counting the `A<code>` suffix
(`traceJSON-<a>-<b>-A<code>-<c>-<runId>.json`):

| Folder | Benign count | Attacker code | Attacker count |
|---|---|---|---|
| `GridSybil_0709` | 1,553 (`A0`) | `A16` | 666 |
| `DataReplaySybil_0709` | 1,293 (`A0`) | `A17` | 554 |
| `DoSRandomSybil_0709` | 1,554 (`A0`) | `A18` | 667 |
| `DoSDisruptiveSybil_0709` | 1,293 (`A0`) | `A19` | 554 |

Every `traceJSON-*` file is one vehicle's **received-message log** — it contains that vehicle's own GPS
readings (`type: 2`) plus every Basic Safety Message it *received* from neighbors (`type: 3`, carrying
`sender`, `senderPseudo`, `pos`, `spd`, `acl`, `hed`). The Sybil signature shows up when one physical
`sender` ID broadcasts under many distinct `senderPseudo` values — this was verified directly in each
family's data below, not assumed from the filename code alone.

---

## GridSybil (A16)

**Mechanism — verified from data.** One physical attacker (`sender 27735` in the sampled file) broadcast
under **6 simultaneous pseudonyms** (`10277353, 20277353, 30277353, 40277353, 50277353, 60277353` —
literally the base ID with a prefix digit, an artifact of the simulator's pseudonym-generation scheme, not
attacker behavior). At the same instant (`rcvTime ≈ 30519.16`), three of those pseudonyms reported positions
`(116.5, 889.6)`, `(116.6, 885.9)`, `(109.8, 889.5)` — a handful of meters apart, forming a small fixed
cluster/offset pattern around one location rather than one vehicle's continuous path. This matches the
literature description (`docs/proposal/proposal-introduction-guide.md`): fixed spatial offsets from a
single true position, broadcast concurrently, producing an apparent "grid" of ghost vehicles.

**What the trajectory shows (Figure, panel 2).** Instead of one smooth path, the panel shows several
short, near-parallel tracks a few meters apart, each belonging to a different `senderPseudo` — visually
distinct from the benign panel's single continuous curve. Reported speed for this family's attacker
messages was noticeably lower than the same-folder benign vehicle (mean 2.31 m/s vs. 3.34 m/s in the
sampled files) — consistent with `docs/proposal/proposal-introduction-guide.md`'s note that Grid-Sybil
positions are quasi-static/off-pattern rather than road-following.

**What the model should learn.** This is a **spatial multiplicity** signal: several identities occupying
near-identical space at the same timestamp. A single-window, single-identity encoder (RoadFM-Lite's current
per-window input) cannot see this directly — it is a *cross-sender* pattern. Within one window, the
learnable proxy is that a Grid-Sybil identity's `pos` sequence is nearly static (low `dpos` magnitude)
while claiming plausible `spd`/`acl` — i.e., **position and velocity become mutually inconsistent**, which
is exactly the "physical-plausibility feature" planned in Phase 2 (task 2.3: speed–position consistency
`‖Δpos/Δt − spd‖`). Note this is **not** the same as the planned "position-offset" SSL corruption (task
2.4), which perturbs `pos` while keeping `spd/acl` from the original trace — Grid-Sybil's *real* signature
is closer to a **near-static offset cluster**, so the synthetic corruption should be checked against this
real pattern before assuming it transfers.

---

## DataReplaySybil (A17)

**Mechanism — verified from data.** Sampled attacker `sender 4833` broadcast under 89 pseudonyms. One
pseudonym (`5048332`) reported the **exact same position** `(269.9, 114.5)` across three consecutive
messages at `sendTime = 26921.45, 26922.45, 26923.45` (1-second spacing) while its own reported speed
field still claimed ~14.9 m/s — a stalled/repeated position paired with a non-zero speed claim. This is a
direct, verified replay signature: stale position data re-broadcast (or interpolated from a stale base)
while other fields continue to update independently, producing an internal inconsistency rather than a
smoothly evolving trace.

**What the trajectory shows (Figure, panel 3).** Multiple short tracks are visible, several of which
contain flat segments (repeated or barely-moving points) where a continuously-moving benign vehicle would
show a smooth curve. This is the **hardest family to distinguish visually** at a glance — most points still
fall on/near plausible road-like paths (mean speed 12.0 m/s in the sample, close to benign's 8.0 m/s in the
same folder) — because the underlying data originates from real (stolen or reused) trajectory segments, not
fabricated positions.

**What the model should learn.** The discriminative signal is **temporal discontinuity within a claimed
identity**: repeated/near-duplicate `pos` values, or `dt`/`dpos` sequences that reset or repeat mid-window,
rather than any single out-of-range value. This maps directly onto the planned **"replay" SSL corruption**
(task 2.4) and gives it real support — but the synthetic corruption should also model the "position frozen
while speed keeps updating" inconsistency observed here, not just literal segment replay, since that
combination is what made this pattern detectable in the raw data.

---

## DoSRandomSybil (A18)

**Mechanism — verified from data.** Sampled attacker `sender 20313` broadcast under 100 pseudonyms at a
much higher rate than benign in the same file (2,905 messages for this attacker vs. 587 for a benign
vehicle in the same folder — roughly 5× the message volume; per-file mean `dt` 0.052s vs. benign 0.051s in
this pair, but aggregate volume is the clearer discriminator here). Individual pseudonyms' positions jump
incoherently step to step — e.g. pseudonym `10203133` reported `(495.5, 1208.7) → (1389.2, 579.7) → (570.8,
786.9)` across three ~50-second-spaced messages, a physically impossible teleport-like jump (~1000+ m in
50s implies >20 m/s in a straight line that reverses direction each step) with reported mean speed 26.1 m/s.
This directly confirms the "random/erratic position, elevated message rate" description.

**What the trajectory shows (Figure, panel 4).** The plotted path is disjoint, jagged jumps across the
whole map extent rather than a locally coherent route — visually the most obviously non-physical family.

**What the model should learn.** Position deltas (`dpos`) that are inconsistent with `spd × dt` at nearly
every step, and elevated message-arrival frequency, are the clearest per-window signals. This lines up
closely with the planned **"speed-scale" and "position-offset" SSL corruptions**, and is likely the easiest
family for RoadFM-Lite to detect — the v1 pilot's precision (0.992) with recall gap (0.824) suggests the
model may already be catching flagrant cases like this while missing subtler ones (DataReplay is the more
likely source of missed recall; per-family recall breakdown is still pending, task 1.4/6.5).

---

## DoSDisruptiveSybil (A19)

**Mechanism — verified from data.** Sampled attacker `sender 7155` broadcast under 77 pseudonyms, mostly
**one message per pseudonym** (77 messages / 77 pseudonyms in the sample) rather than sustained per-identity
tracks — e.g. pseudonyms `1071552, 2071552, 3071552` each appear exactly once, at scattered positions
`(1204.9, 982.9)`, `(820.5, 760.4)`, `(831.4, 768.3)` roughly half a second apart. Combined with elevated
overall message volume (606 messages this file vs. 487 benign in the same folder) and irregular timing
(`dt` mean 0.284s but max 12.47s — much burstier than benign's steadier 0.193s mean), this matches the
literature's "off-road/scattered positions + high message rate + irregular kinematics" description, but the
verified signature here is specifically **single-shot, disposable identities** rather than sustained fake
tracks — closer to a flooding/disruption pattern than a spatial-forgery pattern.

**What the trajectory shows (Figure, panel 5).** Isolated points scattered across the map rather than
connected tracks — because most pseudonyms only ever send one message, there is barely a "trajectory" to
plot for any single identity, which is itself the signal (a real vehicle should have accumulating history).

**What the model should learn.** Within a fixed 20-step window built per `senderPseudo` (as the current
`data/prepared_data/` pipeline does), a DoS-Disruptive identity is disproportionately likely to produce a
**short, sparse window** (few real messages, possibly padded) rather than a full 20-step trace — this
interacts with the windowing pipeline's `min_seq_len` cutoff (task 2.1) and should be checked: are
disruptive-attacker windows systematically under-represented after the `min_seq_len=20` filter, biasing the
class balance further? (Feeds directly into task 1.4's per-class-per-group count table.)

---

## Cross-family summary

| Family | Core mechanism (verified) | Primary window-level signal | Closest planned SSL corruption |
|---|---|---|---|
| GridSybil (A16) | Several pseudonyms cluster near one static point | pos↔speed inconsistency (near-static pos, nonzero claimed speed) | position-offset (partial match — real pattern is clustered/static, not a single offset) |
| DataReplaySybil (A17) | Position freezes/repeats while other fields keep updating | repeated/near-duplicate `pos` mid-window | replay (good match, but should include the "frozen pos + moving speed" combo) |
| DoSRandomSybil (A18) | Incoherent position jumps, elevated message rate | large `dpos` inconsistent with `spd·dt`; high message frequency | speed-scale / position-offset (good match) |
| DoSDisruptiveSybil (A19) | Mostly single-shot disposable pseudonyms, bursty timing | short/sparse per-identity windows, irregular `dt` | none of the four map cleanly — flags a possible 5th corruption type or a `dt`-irregularity operator |

**Implication for Phase 2 (flagged for task 2.4 design notes).** Three of four families translate well onto
the planned corruption suite; **DoSDisruptiveSybil's real signature is about window sparsity and timing
irregularity**, which none of {replay, shuffle, speed-scale, position-offset} directly targets. Consider
adding a "temporal irregularity / dropout" corruption (irregular `dt`, dropped steps) to the TCP pretext
task, or explicitly note in the thesis that this family is expected to be harder to catch through the
current SSL design.

## What is literature vs. verified here

- **Verified directly from `data/VeReMi-Dataset/` in this session:** attacker code mapping (A16–A19),
  multi-pseudonym-per-sender structure for all four families, the specific numeric examples quoted above
  (positions, timestamps, message counts, speeds).
- **From `docs/proposal/proposal-introduction-guide.md`** (general VeReMi-Extension characterization,
  not independently re-derived here): the "off-road grid," "stolen trajectory," "random," and
  "scattered/disruptive" framing used to name each mechanism, and the reported ratios/percentages per
  attack type in the full dataset.
