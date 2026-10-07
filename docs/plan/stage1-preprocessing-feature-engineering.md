# Stage 1 — preprocessing and feature engineering

> Status: measured on the full local data, 2026-10-06. Audience: thesis chapter and advisor review.
> Related: [stage1-ssl-wbs.md](stage1-ssl-wbs.md) (decisions D1–D9), [clarification.md](clarification.md)
> (adopted plan), [split protocol](../research-notes/data_understanding/split-protocol.md),
> [cross-scenario leak](../research-notes/data_understanding/cross-scenario-leak.md).

## Purpose

This document describes how raw VeReMi-Extension receiver logs become the tensors the Stage 1 TimesNet encoder
reads. It covers what one sample is, which 13 features describe each message, how variable-length message
sequences are cut, bucketed and masked, how the data is split, and what the encoder must do with the result.
No attack labels are used to build the inputs. Labels are stored next to them for evaluation only.

## Data as recorded

- Source: `data/VeReMi-Dataset/` (13 GB, read-only): 23,032 receiver trace files, 20,364,658 received messages.
- Beacons arrive at 1 Hz. Positions are SUMO metres. There is no road map in the data.
- Each trace file is one receiving vehicle's log. Each entry is a message it heard: the sender's claimed
  position, velocity and acceleration, plus the receiver's own GPS state when the message arrived.
- Labels (benign or attack type) come from the file names and the ground truth. They are never inputs.

## The sample unit: links

A **link** is what one receiving car heard from one sender pseudonym, in time order (paired with the sender id
for bookkeeping). All windows are cut from links.

**Every received copy is kept.** One broadcast heard by five cars gives five links, one per receiver. This is
intentional:

- The model is receiver-centric (D2). Receiver position and velocity, range, bearing and inter-arrival times
  differ from one receiver to the next, so each copy is a different observation.
- A deployed detector runs on one car and sees only its own copies. Training on every copy matches that.
- Keeping only one copy per message (the primary listener) shrinks the range distribution (benign range p50
  36 m / p99 348 m vs. 81 m / 384 m with all copies, F12) and throws away observations.

The split is by sender vehicle (see below), so the copies of one broadcast never land in different splits.

## The 13 features

| # | Name | Meaning | Unit / transform |
|---|---|---|---|
| 1–2 | `claimed_pos_x`, `claimed_pos_y` | position the sender claims in its message | m |
| 3–4 | `rx_pos_x`, `rx_pos_y` | receiver's own GPS position at reception | m |
| 5–6 | `claimed_vel_x`, `claimed_vel_y` | velocity the sender claims | m/s |
| 7–8 | `rx_vel_x`, `rx_vel_y` | receiver's own velocity at reception | m/s |
| 9–10 | `claimed_acl_x`, `claimed_acl_y` | acceleration the sender claims | m/s² |
| 11 | `range` | distance between claimed sender position and receiver | m |
| 12 | `bearing` | direction from receiver to claimed sender position | rad, one wrapped angle (not sin/cos) |
| 13 | `log_dtau` | time since the previous message on the same link | `log(max(Δτ, 1e-3))`, Δτ in s; null for a link's first message |

All features are then standardised (next section).

**Never an input:** time of day (it alone separates 0709 from 1416), the true sender id, the `*_noise` fields,
and any ground-truth or label field.

## Normalisation

- Mean and standard deviation per feature are computed **from the train split only**, and only over real rows
  (mask = 1). Missing `log_dtau` values are ignored in the statistics.
- After standardisation, padding rows are set to exactly 0, and a missing `log_dtau` is set to 0 (the train mean).
- The same train statistics are applied to pretrain_val, val and test.

## The window technique: crop, length buckets, mask

> **Current setting (2026-10-06, user choice): `buckets = (64,)`** — every window is padded to 64 rows, so the
> encoder always sees T = 64 (376,427 windows, 71.0% padding rows, output `<split>/b64/` only, 1.8 GB). The
> bucketed variant below (26.3% padding) is one setting away. The mask rules apply even more strongly at 71% padding.

Links have very different lengths (benign median 14 messages, GridSybil median 11, p90 ≈ 45). The encoder input
uses a maximum length of **T = 64**:

- **Crop.** Every link is cut into consecutive, non-overlapping pieces of at most 64 messages
   (`crop_stride = 64`). Every message ends up in exactly one window. Nothing is dropped.
- **Length buckets {8, 16, 32, 64}.** A piece of n messages goes to the smallest bucket ≥ n and is padded at the
   end only up to that bucket size.
- **Mask.** Each window carries a mask: 1 for real rows, 0 for padding. The mask is always a prefix of ones.

Example: a 150-message link and a 5-message link.

```
150-message link  m1 ................................................ m150
crop at 64    ->  [ m1  .. m64  ]  [ m65 .. m128 ]  [ m129 .. m150 ]
                    64 messages      64 messages      22 messages
bucket        ->     b64              b64              b32
padding       ->     none             none             10 rows
mask          ->  1111...1111 (64) 1111...1111 (64) 1...1 (22) 0...0 (10)

5-message link    m1 m2 m3 m4 m5
crop          ->  [ m1 .. m5 ]              (shorter than 64: one window)
bucket        ->     b08
padding       ->     3 rows
mask          ->  1 1 1 1 1 0 0 0
```

Each shard holds one split and one bucket, so a batch always has one length. The batch's T varies between
batches (8, 16, 32 or 64).

### Why this technique

- **A fixed T without padding loses most data.** Keeping only links with at least T messages keeps 5% of links and
  21% of messages at T = 64. T = 24 is the largest T that keeps at least half the messages.
- **Padding everything to 64 is mostly padding.** 71.0% of rows would be padding, and TimesNet's FFT and
  convolutions would read those zeros as data.
- **Bucketing** (Khomenko et al., 2016, arXiv:1708.05604) keeps padding low (26.3% here) and lets each batch run
  at its own T. FFT and convolutions work at any T.
- **Masks make padding invisible to training.** Masked pooling and masked losses are standard in time-series
  foundation models (MOMENT, TimesFM, Chronos). The official TimesNet classification head masks only the output,
  not the FFT, so our encoder adds masking itself (see "What the encoder must do").
- **Cropping long links** bounds sequence length. Random crops of long links can also serve as contrastive views
  (TS2Vec, AAAI 2022).

### Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Resampling or interpolation to a fixed length | Invents motion that was never broadcast; breaks Δτ |
| Backfill (repeat the last message) | Looks exactly like a frozen-position attack |
| Packing several links into one row | TimesNet's FFT and convolutions would mix the links |
| Fixed T, drop short links | Loses 79% of messages at T = 64 |
| Pad all windows to 64 | 71.0% padding |

## Split by vehicle (RULE 3)

- The split is by **sender vehicle**, taken from `index.json`: 5,752 sender vehicles, none in two splits.
- This replaces the v1 split, which leaked across the four scenarios (F1; the scenarios reuse the same traffic).
- Splits: `train`, `pretrain_val` (label-free checkpoint selection, D7), `val`, `test`.
- Checked: 0 vehicles in two splits.

## Outputs and file layout

```
src/data/
├── prepared_data/                      2.9 GB, one prepared JSON per raw trace file
│   ├── <mirrors the VeReMi tree>       every received copy, 13 features, messages grouped into links
│   └── index.json                      labels (from file names), vehicle split, normalisation, counts
└── encoder_input/
    └── T64/                            1.0 GB, 48 shards
        ├── metadata.json               settings, normalisation stats, counts, checks
        ├── train/
        │   ├── b08/  part-00000.json  part-00000_info.json ...
        │   ├── b16/  ...
        │   ├── b32/  ...
        │   └── b64/  ...
        ├── pretrain_val/  (b08 b16 b32 b64)
        ├── val/           (b08 b16 b32 b64)
        └── test/          (b08 b16 b32 b64)
```

Shard schema (what the encoder reads; **no labels**):

```jsonc
[
  { "id": "<window id>",
    "x": [[13 floats], ...],      // bucket rows × 13 features, padding rows = 0
    "mask": [1, 1, ..., 0, 0] }   // bucket entries, 1 = real, 0 = padding
]
```

Side file `part-XXXXX_info.json` (evaluation only): per window `id`, label, `n_messages`, `bucket` and the
link/vehicle ids. The encoder never opens it during pretraining.

Source for `T64`: the `GridSybil_0709` and `GridSybil_1416` prepared files, benign and GridSybil senders.
271 links with unknown labels (4,123 messages) are dropped; 354,197 links and 6,980,356 messages are kept.

## Numbers

Windows per split and bucket (Benign / GridSybil):

| Split | b08 | b16 | b32 | b64 | Total |
|---|---|---|---|---|---|
| train | 26,599 / 62,113 | 19,198 / 31,549 | 22,998 / 32,640 | 15,487 / 25,848 | 236,432 |
| pretrain_val | 2,588 / 8,295 | 1,981 / 4,303 | 2,224 / 3,953 | 1,724 / 3,338 | 28,406 |
| val | 6,452 / 14,804 | 4,697 / 7,871 | 5,384 / 8,109 | 3,437 / 6,867 | 57,621 |
| test | 6,113 / 14,153 | 4,765 / 7,372 | 5,469 / 6,523 | 3,698 / 5,875 | 53,968 |
| **Total** | 141,117 | 81,736 | 87,300 | 66,274 | **376,427** |

Benign share: 35.3% of windows (132,814 benign, 243,613 GridSybil).

Padding per bucket:

| Bucket | b08 | b16 | b32 | b64 | Overall | If all padded to 64 |
|---|---|---|---|---|---|---|
| Padding rows | 50.7% | 23.7% | 28.0% | 19.5% | **26.3%** | 71.0% |

Checks passed: shapes per bucket; mask is a prefix of ones with sum = `n_messages`; padding rows exactly 0; no
NaN; window ids unique; 0 vehicles in two splits; 376,427 windows with a total mask sum of 6,980,356 (= messages
kept).

## What the encoder must do with the mask

- **Pooling:** z = masked mean of H over real rows only.
- **Losses:** masked reconstruction, physics heads P1–P3 and contrastive loss are computed on real rows only.
- **TimesBlock padding:** compute the period padding from the batch's own length T, not from a fixed 64.
- **Optional:** choose the FFT periods from real rows only, so padding does not create false periods.

## Limitations and open choices

- **Very short windows (1–3 messages):** 18,509 benign (13.9%) and 49,242 GridSybil (20.2%). The class rates
  differ, so window length could become a shortcut, and TimesNet's FFT finds little in fewer than 8 messages.
  Option: a floor (`min_messages`, e.g. 4). Not yet decided.
- **DataReplay and DoS are not captured per link.** These attacks rotate pseudonyms every 1–2 messages (69–88%
  of their links are single messages), so a per-link window sees almost nothing (F11). A receiver time window
  (all senders one car hears in 10–20 s) is the documented later option; it is outside the current Stage 1
  scope.
- **Ghost pseudonym 1:** GridSybil ghosts share pseudonym 1 in every GridSybil run; these messages are grouped
  by sender id (`split_by_sender`). Whether to keep this or drop them is open (F13).
- **Class imbalance:** about 35% benign / 65% GridSybil. Stage 1 does not use labels, but probes and metrics must
  account for it.
- **Outliers:** normalisation is mean/std, which is sensitive to extreme values (e.g. long Δτ gaps or far
  ranges). Whether to clip or switch to robust scaling is open.

## How to reproduce

Run from the repo root, in this order:

- `src/pipeline/input_representation.ipynb` — raw logs → `src/data/prepared_data/` (≈ 66 s).
- then `src/eda/eda_window_size.ipynb` — read-only EDA of link lengths, the T sweep and class balance (writes nothing).
- then `src/pipeline/benign_gridsybil/encoder_input_T64.ipynb` — `prepared_data` → `src/data/encoder_input/benign_gridsybil/T64/` with checks (≈ 1.4 min).

None of them write under `data/`.
