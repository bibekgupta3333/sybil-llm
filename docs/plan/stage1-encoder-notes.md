# Encoder study notes — TimesNet, benign + GridSybil trial (T = 64)

Written 2026-10-07. Study notes for the encoder built (not trained) in
[`src/model/benign_gridsybil/timesnet_encoder_T64.ipynb`](../../src/model/benign_gridsybil/timesnet_encoder_T64.ipynb).
Every number below comes from that notebook's code and outputs; nothing has been trained, so there are **no model
results** here. Input data: [`stage1-preprocessing-feature-engineering.md`](stage1-preprocessing-feature-engineering.md).
Plan: [`clarification.md`](clarification.md) and the WBS [`stage1-ssl-wbs.md`](stage1-ssl-wbs.md).

**In one paragraph.** The encoder reads one window, 64 messages × 13 features plus a mask that marks which rows are
real. A token embedding turns each message into a 128-number vector. Four TimesBlocks then find each window's
strongest rhythms (periods) with an FFT over its real rows. Each block folds the sequence into a 2-D grid per period,
runs small 2-D convolutions and mixes the results. Out come `H` (one vector per message, 64 × 128) and `z` (one vector
per window, 128, the average over real rows only). The mask is applied at every stage, so padding never changes the
output. The model has 2,301,312 parameters. Stage 1 will train it without attack labels.

## Contents

- [The encoder from the ground up](#the-encoder-from-the-ground-up)
- [Masking inside the model, and the checks that prove it works](#masking-inside-the-model-and-the-checks-that-prove-it-works)
- [How Stage 1 will use the encoder, what is built and what is not](#how-stage-1-will-use-the-encoder-what-is-built-and-what-is-not)


## The encoder from the ground up

### What an encoder is, and why we pretrain one

An encoder is a function that turns raw input into vectors that are easier to use downstream. Ours reads one
window of messages a receiver heard on one link and returns two things:

- **H**, shape (B, 64, d): one d-dimensional vector per position (per message). Later a small decoder uses H to
  reconstruct messages we hide.
- **z**, shape (B, d): one vector for the whole window, the mean of H over the real messages only. Later the
  physics heads and the contrastive loss read z, and in the end z is the window's representation for probes and
  few-shot detection.

Here d = `d_model` = 128. We **pretrain** the encoder without attack labels (Stage 1) because labels are scarce.
The idea is that an encoder that has learned what normal, physically plausible motion looks like should need
only a few labelled examples to separate Sybil traffic later. The notebook
(`src/model/benign_gridsybil/timesnet_encoder_T64.ipynb`) builds and inspects this encoder; **it trains nothing**
(no optimizer, no backward pass) and never opens the `_info.json` label files.

### The input: 64 rows × 13 features + a mask

Each message is a row of 13 numbers, in this column order:

| # | features | meaning |
|---|---|---|
| 0–1 | `claimed_pos_x/y` | position the sender claims |
| 2–3 | `rx_pos_x/y` | receiver's own position |
| 4–5 | `claimed_vel_x/y` | velocity the sender claims |
| 6–7 | `rx_vel_x/y` | receiver's own velocity |
| 8–9 | `claimed_acl_x/y` | acceleration the sender claims |
| 10 | `range` | receiver–sender distance |
| 11 | `bearing` | one wrapped angle (not sin/cos, D3) |
| 12 | `log_dtau` | log of the time gap to the previous message |

There is **no time of day** (it alone separates the 0709 and 1416 groups). Values are z-scores using
train-split statistics. A window holds at most 64 messages; shorter windows are padded with all-zero rows up to
64, and a **mask** of 64 flags says which rows are real (1) and which are padding (0). The mask is always a run of
1s followed by 0s.

A real training window from `train/b64/part-00000.json` with 12 real messages (values rounded):

```
row  0: [ 0.34 -0.78  1.07 -0.51 -0.02 -0.02  0.44 -1.23 -0.17 -0.04  0.39 -1.36  0.00]
row  1: [ 0.31 -0.79  1.08 -0.55  0.08  0.01  0.45 -1.22  0.40  0.11  0.43 -1.38  0.16]
row  2: [ 0.90 -0.53  1.10 -0.61 -0.15 -0.06  0.33 -0.88 -0.83 -0.19 -0.40 -1.02  1.62]
 ...
row 11: [ 1.06 -0.49  0.63 -0.88  0.81  0.27 -1.36 -0.44  4.42  1.19  0.27  0.33  0.16]
row 12: [ 0     0     0     0     0     0     0     0     0     0     0     0     0   ]   <- padding
 ...    (rows 12..63 are all zero)
mask:   1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 ... 0
```

This window is typical: **about 71% of all rows are padding**. In the notebook's seeded batch of 256 windows,
real rows range from 1 to 64 with a median of 14, and the padding share is 70.5%. So the encoder must look only
at real rows; otherwise it can learn "where the zeros start" (the window length) instead of motion. Section B
covers this in detail.

### TimesNet intuition: rhythms, folding, 2-D convolution

TimesNet (Wu et al., ICLR 2023) starts from one observation: many sequences have **rhythms**, and a rhythm is
easier to see in 2-D than in 1-D.

**Find rhythms with an FFT.** A Fourier transform says how strongly each frequency f is present. Frequency f
means "repeats f times in the window", so its period is `n // f` steps. Our version runs the FFT over each
window's **n real rows only**, averages the amplitude over channels, ignores f = 0 (the mean) and f = 1 (once per
window) via `min_freq = 2`, and keeps the **top 3** frequencies, separately for every window.

Worked example (window 0 of the batch, 28 real rows): the FFT has frequencies 0…14. The three strongest allowed
ones give periods **[14, 7, 5]** (28 // 2 = 14, 28 // 4 = 7, 28 // 5 = 5) with amplitudes [2.02, 1.56, 1.42]. A
window with fewer than 4 real rows has no allowed frequency (n // 2 < 2) and gets period 1; a window with 4 real
rows gets [2, 1, 1]. In the notebook's period diagnostic, 23.7% of period slots fall back to 1.

**Fold.** For one period p, the 64-step sequence is zero-padded to a multiple of p and cut into rows of length p,
stacked into a grid of `cycles × p`. With p = 8:

```
time:   0  1  2  3  4  5  6  7  8  9 ... 63

grid (8 cycles × 8):      column = position in the cycle
          col0 col1 ... col7
cycle 0:   0    1   ...   7
cycle 1:   8    9   ...  15
cycle 2:  16   17   ...  23
  ...
cycle 7:  56   57   ...  63
```

Neighbours **along a row** are consecutive messages (short-term change). Neighbours **down a column** are the
same moment in the previous cycle (change from one cycle to the next). With p = 14, 64 steps are padded to 70 and
folded into 5 × 14.

**2-D convolution.** An Inception block runs three 2-D convolutions side by side (kernels 1×1, 3×3, 5×5, "same"
padding) and averages them. A 3×3 kernel at one cell sees its two time neighbours *and* the cells one period
earlier and later, so it compares both directions at once.

**Unfold and mix.** The grid is reshaped back to 64 steps (extra padding cut off). This is done for each of the 3
periods, giving 3 versions of the sequence. They are mixed with `softmax(amplitudes)` per window, so a stronger
rhythm gets more weight. Then the block's input is added back (**residual**), and the result is multiplied by the
mask so padding rows stay exactly 0. Note: a fallback period-1 slot has amplitude 0, but softmax still gives it a
non-zero weight (exp(0) = 1 before normalising), so it still contributes.

Inside one TimesBlock, the convolutional path is

```
Inception(d=128 -> d_ff=64) -> GELU -> Inception(64 -> 128)
```

The period search runs again in every block, on that block's input (128 channels instead of 13), so periods can
change between blocks. For window 0 they were [14, 7, 5] in block 0 and [14, 7, 9] in blocks 1–3. Across the
batch, 16.4% / 18.8% / 23.8% of windows in blocks 1 / 2 / 3 have periods different from block 0.

### Token embedding and LayerNorm

Before the blocks, `TokenEmbedding` lifts 13 features to d = 128 per row:

- a **Conv1d over time** (13 → 128, kernel 3, no bias), so each token already sees its previous and next message.
  It pads with **zeros, not circularly**. The THUML reference wraps around, which would let the last rows of the
  window (padding) leak into the first real row.
- plus a **fixed sine/cosine position code** (a buffer, not learned, so it adds no parameters). There is no
  time-of-day or other time-feature embedding (D3).
- then dropout (0.1, the only dropout in the encoder) and × mask.

After every TimesBlock comes one **LayerNorm** (normalises each row's 128 values to mean 0 and variance 1, then
learns a scale and shift), **shared by all four blocks** as in the reference model, followed by × mask again.

### One batch through every layer

B = 256, T = 64, d = 128, d_ff = 64, k = 3.

| Stage | Shape | Notes |
|---|---|---|
| `x`, `mask` from the loader | (B, 64, 13), (B, 64) bool | padding rows already 0 |
| `fill_padding` | (B, 64, 13) | padding forced to 0 whatever the file held |
| Conv1d (on transposed input) | (B, 13, 64) → (B, 128, 64) → (B, 64, 128) | kernel 3, zero padding |
| + positions, dropout, × mask | (B, 64, 128) | embedding output |
| **TimesBlock**: period finder | periods (B, 3) long, weights (B, 3) | per window, real rows only |
| fold, one period p | (b, 64, 128) → (b, 128, ⌈64/p⌉, p) | b = windows sharing that period |
| Inception 128 → 64, GELU | (b, 64, ⌈64/p⌉, p) | |
| Inception 64 → 128 | (b, 128, ⌈64/p⌉, p) | |
| unfold, cut to 64 | (b, 64, 128) | |
| stack the 3 slots | (B, 64, 128, 3) | |
| softmax mix + residual, × mask | (B, 64, 128) | |
| shared LayerNorm, × mask | (B, 64, 128) | |
| (repeat for blocks 2–4) | (B, 64, 128) | |
| **H** | (B, 64, 128) | exactly 0 at padding rows (checked) |
| **z** = masked mean of H | (B, 128) | average over real rows only |

### Where the 2.30M parameters live, and why d_ff = 64

| Module | Parameters | How |
|---|---|---|
| embedding | 4,992 | 13 · 128 · 3, no bias |
| each TimesBlock | 574,016 | two Inception layers, see below |
| 4 blocks | 2,296,064 | |
| LayerNorm | 256 | 128 scales + 128 shifts |
| **encoder total** | **2,301,312** | F9 target 2.30M, gap +0.057% |

Per block: kernels 1×1 + 3×3 + 5×5 have 1 + 9 + 25 = 35 weights per channel pair. The first Inception has
128 · 64 · 35 = 286,720 weights + 3 · 64 = 192 biases; the second has 64 · 128 · 35 = 286,720 + 3 · 128 = 384
biases. Sum: 574,016.

Why the bottleneck **d_ff = 64**: almost every parameter sits in these 2-D convolutions, and their size grows
with `d × d_ff`. WBS finding F9 counted that with d_ff = d and the reference's 6 kernels, d = 128 / 256 / 512
gives 37.5M / 150M / 600M parameters — far too many for roughly 18k vehicle identities. With d_ff fixed at 64
and 3 kernels the encoder is 2.30M / 4.60M / 9.20M, so it grows only linearly with d across the planned sweep.

The notebook also defines (untrained) heads on top: reconstruction 18,189, physics 24,963, projection 33,024, for
2,377,488 in total. They are dropped after pretraining; only the encoder is kept (Section C).

---

## Masking inside the model, and the checks that prove it works

### Why the model needs a mask at all

Every window is stored as 64 rows × 13 features, but most windows are short. Across the whole encoder input,
**71.0% of all rows are padding** (70.5% in the notebook's demo batch of 256). The median window has only
12–14 real messages (14 in the demo batch). The shortest has 1 and the longest 64. The `mask` (B, 64) records
which rows are real: `True` for a real message, `False` for padding. The real rows always come first, followed
by the padding.

Padding rows are zeros. If the model treats them as data, everything it computes starts to depend on **how
long the window is**, not on how the vehicle moved. The simplest case is pooling. Take one feature with three
real values and two padding rows:

```
values : 2   4   6   0   0
mask   : 1   1   1   0   0

plain mean  = (2+4+6+0+0) / 5 = 2.4    <- pulled toward 0 by padding
masked mean = (2+4+6) / 3     = 4.0    <- the true average of the real rows
```

With 64 rows the plain mean is `(n / 64) × true mean` for a window with n real rows, so it scales with
length. The EDA measured the effect on real data. For the `log_dtau` feature, a per-window mean over the real
rows has an AUC of 0.492 (benign vs GridSybil), which is no signal. An unmasked mean drops it to 0.442 because
it now encodes length. Length alone separates the classes at **AUC 0.531**. A model that learns "where the
padding starts" would therefore look slightly useful while having learned nothing about motion.

### Every place the mask is applied

| Where | What the code does | Why |
|---|---|---|
| Input | `MaskedOps.fill_padding(x, mask)` sets padding rows to 0 | Whatever the file put in the padding rows is erased, so the input contents there can never matter (check 1). |
| Token embedding | `out * mask` after conv + positions + dropout | The sine/cosine position code is non-zero at every position, and the kernel-3 convolution at the first padding row sees the last real row. Without this step the padding rows would hold values. |
| Each TimesBlock | `(mix of k results + x) * mask` at the end of `forward` | The 2-D convolutions and the residual spill values into padding cells. |
| After the shared LayerNorm | `self.norm(block(h, mask)) * keep` | LayerNorm maps an all-zero row to its bias β, which is not zero, so the rows must be zeroed again. |
| Period search | `PeriodFinder` runs the FFT on `h[rows, :n]`, the first n real rows only | The chosen rhythms should describe the motion, not the edge where the padding starts (see below). |
| Pooling | `z = MaskedOps.masked_mean(H, mask)`: the sum over real rows divided by the real-row count (clamped ≥ 1) | Avoids the 2.4 vs 4.0 problem above. |
| Reconstruction | `ReconstructionMasker` gives padding a score of 2.0 so it sorts last and is never hidden, and `masked_mse` counts only the hidden rows | There is nothing to predict in padding, and visible rows would make the task trivial. |

The physics heads and the projection head read `z`, which is already masked, so they need no extra step.
Shapes are unchanged throughout: `x` (B, 64, 13) → embedding (B, 64, 128) → 4 blocks (B, 64, 128) = `H`, with
`H` exactly 0 at padding rows (printed `True` in the notebook), and `z` (B, 128).

### The FFT runs on real rows only

TimesNet picks its periods from an FFT. The first version ran the FFT over all 64 rows, padding included.
That was replaced, for two reasons.

1. **Leakage into neighbouring frequencies.** A window with 12 real rows followed by 52 zeros looks, to the
   FFT, like "signal × a rectangle that switches off at row 12". Multiplying by a rectangle spreads every true
   frequency into its neighbours (spectral leakage). The sharp drop to zero also adds its own components.
   The top-3 peaks then partly describe the padding edge.
2. **A measured difference.** The synthetic test used sine waves with a known period (4 or 8) plus noise,
   at real lengths n ∈ {16, 24, 32, 40, 64}, followed by padding. Real-rows-only found the true period as the
   top choice **10/10** times. The zero-padded FFT managed **9/10**.

How the code works: windows with the same real length n are grouped. The code runs `rfft` over their first n
rows, averages the amplitude over the channels, allows only frequencies f ≥ 2 (`min_freq`), takes the top 3,
and sets **period = n // f**. A period can therefore never exceed n // 2. An example from the batch: a window
with 28 real rows gets periods [14, 7, 5] (f = 2, 4, 5), and a window with 4 real rows gets [2, 1, 1].

The fold still runs over the full 64 rows. For the 28-row window with p = 14, the grid is ⌈64/14⌉ = 5 cycles
× 14, zero-padded to 70 rows. Cycles 0–1 hold the real data and cycles 2–4 are zeros.

**Periods are chosen per window.** The reference implementation averages the spectrum over the whole batch,
so every window shares the same periods. A window's output would then depend on which other windows happen to
be in its batch. Here each window's periods come from its own rows only (WBS S1.2.2), which is what check 2
tests.

### How this differs from the THUML reference

| Reference TimesNet (THUML Time-Series-Library) | This encoder | Reason |
|---|---|---|
| Token Conv1d pads **circularly** | **Zero** padding | With circular padding the first real row would see the last row, which is padding. |
| FFT over the full sequence, amplitude averaged over the **batch**, one shared top-k | FFT over **each window's real rows**, own top-k | No padding in the spectrum, and no dependence on the batch. |
| Only f = 0 excluded | f = 0 and f = 1 excluded (`min_freq = 2`) | f = 1 means "once per window", which is not a rhythm. |
| Embedding = token + position + **time-feature** embedding | Token + position only | D3 forbids time of day as an input, because it separates 0709 from 1416. |
| No mask inside the blocks | Output **× mask** after every block and after the LayerNorm | Keeps padding exactly 0 at every depth. |
| Classifier multiplies by the padding mask, then **flattens** (T·d) | **Masked mean** → z (d) | The plan calls for an average-pooled z, and a mean over real rows does not scale with length. |

The parts that are kept: Inception kernels 1×1/3×3/5×5 (`Inception_Block_V1`, MIT licence), the d → d_ff → d
bottleneck with GELU, softmax mixing of the k results by amplitude, the residual, the shared LayerNorm, and
the kernel-3 token convolution without bias.

### The checks, and what each one proves

All checks run on CPU in eval mode (dropout off), float32, on the seeded demo batch of 256 windows, with the
**untrained** model.

| # | Check | Result | What it proves |
|---|---|---|---|
| 1 | Fill padding rows with random values (×10), compare H at real rows and z | max \|ΔH\| = 0, max \|Δz\| = 0 | The values stored in padding have no effect at all. (Their *position* still has an effect; see limitations.) |
| 2 | Shortest, median and longest window run alone vs inside the batch | max \|Δz\| = 7.2e-7 | Per-window periods work: batch neighbours do not change a window's output, apart from float rounding. |
| 3 | NaN / Inf in H, z, recon, proj and the 3 physics logits | none (7 tensors) | The forward pass is numerically sound, including 1-row windows. |
| 4 | Build the model twice with seed 0 | max \|Δz\| = 0 | Fully reproducible weights and outputs (RULE 4). |
| 5 | Masked mean of [2, 4, 6, pad, pad] | 4.0 (plain 2.4) | Pooling ignores padding. |
| 6 | Same forward on CPU and Apple GPU (MPS) | CPU ≈ 1.46 s, MPS ≈ 0.32 s per 256 windows; \|Δz\| = 1.7e-6 | The GPU gives the same answer about 4.6× faster. |
| 7 | Correlation of ‖z‖ with the real-row count | Spearman −0.24 (Pearson −0.17) | Measured only, no pass/fail. Length is still visible to an untrained encoder. |

The period diagnostic (S1.2.6) adds one more number. **23.7% of all period slots** in the batch fall back to
period 1, because those windows are too short to have a rhythm. Between 16% and 24% of windows pick different
periods in blocks 1–3 than in block 0.

### Known limitations

- **Edge effect.** Padding is zero, but it is still *there*. The kernel-3 embedding convolution, the 2-D
  kernels in the fold and the "same" zero padding all let the last real rows see zeros where a longer window
  would have data. A window's end position therefore leaves a trace in H, which check 1 cannot catch because
  it only changes padding *values*.
- **Very short windows.** A window with n real rows has frequencies 0 … n // 2. Requiring f ≥ 2 leaves:
  - n < 4: no usable frequency, so all 3 slots get period 1. The fold is then a single column, and the 2-D
    kernels act along time only.
  - n = 4–5: one usable period. n = 6–7: two.
  - n ≥ 8: all three.

  The fallback slots carry amplitude 0, and the softmax still gives them weight. For example,
  softmax([0.169, 0, 0]) ≈ [0.37, 0.32, 0.32], so short windows lean heavily on plain time convolution.
- **Untrained length correlation.** Spearman −0.24 between ‖z‖ and length means the masking removes the
  *mechanical* leak (padding values, plain mean) but not every route to length. The real test comes later: a
  frozen-encoder probe must clearly beat the length-only AUC 0.531.
- **Batch invariance is approximate.** The match is to about 1e-6, not exact. Different batch sizes and the
  grouping of windows by n and by period change the order of floating-point sums. In training mode, dropout
  adds randomness on top. Both effects are harmless, but "identical" would be the wrong word.

---

## How Stage 1 will use the encoder, what is built and what is not

### Three heads on top of the encoder

The encoder on its own only turns a window into numbers: `H` (B, 64, 128), one vector per message, and `z`
(B, 128), one vector per window. Stage 1 teaches it **without attack labels** by giving it three jobs. Each job has
its own small network ("head") on top of the encoder. After pretraining the heads are thrown away and only the
encoder is kept.

| Head | Reads | Layers | Output | Parameters | What it will learn |
|---|---|---|---|---|---|
| `ReconstructionHead` | `H` (B, 64, 128) | Linear 128→128, GELU, Linear 128→13, applied to every row | `recon` (B, 64, 13) | 18,189 | rebuild the features of messages we hid |
| `PhysicsHeads` (3 MLPs) | `z` (B, 128) | per head: Linear 128→64, GELU, Linear 64→1 | 3 logits, each (B,): `p1_speed_jump`, `p2_speed_scaled`, `p3_sharp_turn` | 24,963 (3 × 8,321) | "did *we* inject this kind of violation into the window?" |
| `ProjectionHead` | `z` (B, 128) | Linear 128→128, ReLU, Linear 128→128, then scaled to length 1 | `proj` (B, 128) | 33,024 | map two views of the same window close together |

Totals measured in the notebook: encoder 2,301,312 + heads 76,176 = **2,377,488** parameters. The full model class
is `RoadFMLite`; `forward(x, mask)` returns a dict with `H`, `z`, `recon`, `physics`, `proj`.

The physics heads follow decision D4. Their target is a flag we create ourselves: with probability 0.5 a window gets
a violation injected (P1 speed spike / position jump, P2 speed scaled without matching positions, P3 impossible
sharp turn), and the head must say whether it did. It never sees an attack label. The injectors are **not built
yet**, so these heads currently have nothing to learn from.

### Hiding messages for reconstruction

`ReconstructionMasker` hides **exactly 25% of each window's real rows** (`round(0.25 × real rows)`, at least 1).
Padding rows are never picked, because there is nothing there to predict. The choice is random but uses its own
seeded generator, so the same seed hides the same rows. Output: `x_hidden_input` (B, 64, 13), a copy of `x` with
hidden rows set to 0, and `hidden` (B, 64) bool, always a subset of `mask`.

From the notebook run (seed 0):

```
window 0: real rows 28 -> hidden 7 at positions [2, 3, 12, 13, 18, 19, 27]
window 1: real rows 15 -> hidden 4 at positions [1, 6, 7, 13]
```

All positions are below the window's length, so only real rows were hidden. Setting hidden rows to 0 is a
placeholder: S1.3.A2 replaces it with a learned mask token, so TimesNet's FFT does not see artificial zero blocks.

### The two loss functions defined

- **Masked reconstruction loss** (`masked_reconstruction_loss`): mean squared error between `recon` and the
  original `x`, summed over the 13 features and averaged over **hidden rows only**. Visible rows and padding never
  count. (S1.3.A3 will later restrict it to sender position, velocity, acceleration and log-Δτ and log it per feature.)
- **NT-Xent** (`nt_xent`, the SimCLR loss). Stack the two views' projections into 2B vectors. Each vector's
  positive is the other view of the same window; the other 2B − 2 vectors are negatives. Similarities are cosines
  divided by τ = 0.1, and the loss is cross-entropy for "pick your partner".

Small example with B = 2 (windows a, b, two views each → a1, b1, a2, b2): for anchor a1 the right answer is a2;
b1 and b2 are the wrong answers. A random guesser picks among 2B − 1 = 3 candidates, so its loss is about ln 3.

**Why the printed losses mean nothing.** The notebook prints reconstruction 2.0967 and NT-Xent 3.0049 (chance
ln(511) = 6.2364 for B = 256). The weights are random. The two "views" are only the same batch with two different
25% maskings, a stand-in for the real augmentations; the two views are nearly identical inputs, so even an
untrained network can pair them, which is why NT-Xent sits below chance. Neither number is a result.

### Plan vs built

| Plan item (WBS) | Status |
|---|---|
| TimesNet encoder, masking, masked mean pooling (S1.2.x) | built, checked, **not trained** |
| Three heads + `RoadFMLite` | built, shape-checked only |
| Random masking, exactly 25% of real rows (S1.3.A1, part) | built (zero-fill placeholder) |
| Block masking: one block of 20–30%, chosen 50/50 per sample (S1.3.A1) | not built |
| Learned mask token (S1.3.A2) | not built |
| Masked reconstruction MSE / NT-Xent τ = 0.1 | defined, evaluated once on random weights |
| Violation injectors P1–P3, p = 0.5 (S1.3.P1) | not built |
| Physically consistent augmentations: common-mode shift, time stretch, crop by subsampling (S1.3.C1) | not built |
| Hard negatives: same 50 m grid cell, same group, other pseudonym, ≥ 10 min apart (D6) | not built |
| Joint loss λ1·L_rec + λ2·L_nce + λ3·L_P1 + λ4·L_P2 + λ5·L_P3, λ1 = 1, λ3–λ5 = 0.3, λ2 ∈ {0.1, 0.3, 1}, normalised (D7, S1.3.J1) | not built |
| Training loop | not built |
| Label-free checkpoint selection: masked MSE, alignment/uniformity, effective rank, head losses (S1.3.J2) | not built |

The torch environment is still unpinned (RULE 4), so no training run would be citable yet.

### What the plan text still needs because of padding

The WBS was written for windows with no padding. With 71% padding rows and a median of about 14 real rows, three
wordings must change:

- **Random masking: 25% of the real rows**, not 25% of T. At T = 64, 25% of T is 16 rows, more than a typical
  window holds. The built masker already does this.
- **Block masking: 20–30% of the real rows**, not "L = round(r·T)", which means 13–19 steps of T = 64 and would
  cover a whole 14-row window.
- **Every loss on real rows only**: reconstruction on hidden real rows, pooling over real rows, and augmentations
  and injectors must leave padding untouched.

### Questions the professor may ask

**Why TimesNet?** It is the adopted plan. TimesNet finds a window's dominant rhythms with an FFT, folds the
sequence into a 2-D grid per rhythm and uses 2-D convolutions, so it sees neighbouring messages and "the same moment
one cycle earlier" together. Vehicle motion is mostly not periodic, which is exactly why a Transformer encoder
under the same pretraining is kept as an ablation (S1.4.5).

**Why T = 64?** T = 64 is the plan's length (D1, 128 as a sensitivity). Long links are cut into pieces of at most
64 messages, so every message is used once. Padding every window to 64 was the user's choice (one fixed length;
compute cost accepted); length buckets (8/16/32/64, 26.3% padding) are the documented alternative.

**Why a mask instead of no padding?** Dropping links shorter than 64 keeps only 21% of messages. Resampling
invents motion and breaks Δτ; repeating the last message looks like a frozen-position attack. Padding plus a mask
keeps every message and makes the padding invisible: filling padding with random garbage changed `H` at real rows
and `z` by exactly 0.

**How are labels kept out?** The split is by sender vehicle. Window ids (whose file names carry the attack code) are
bookkeeping only and never an input. Physics heads learn our own injection flag. Contrastive positives are two views
of the same window. Checkpoints will be chosen with label-free measures. Labels are used only for evaluation.

**What proves it works?** So far only that the plumbing is correct: padding ignored (Δ = 0), a window alone vs. in
the batch agrees to 7.2e-7, no NaN/Inf, same seed → identical output, CPU vs MPS agree to 1.7e-6. Whether the
representation is *useful* is not shown yet. That needs training and then frozen-encoder probes (S1.4.1) that
beat both TimesNet trained from scratch and the length-only baseline (AUC 0.531). Untrained, ‖z‖ already correlates
with the number of real rows (Spearman −0.24), so the length check matters.

### Glossary

| Term | Meaning here |
|---|---|
| Encoder | The network that turns a window (B, 64, 13) into `H` and `z`; kept after pretraining |
| H | Per-message output, (B, 64, 128); zero at padding rows |
| z | Per-window output, (B, 128): masked mean of `H` over real rows |
| Mask | (B, 64) bool, True = real message, False = padding |
| Padding | Filler rows that bring a short window up to 64; 71% of all rows |
| Period | Length of a repeating rhythm, in messages; per window, top 3 from the FFT (period = real rows // frequency) |
| FFT | Fast Fourier transform: splits a sequence into frequencies; here over each window's real rows only |
| Inception | A block of parallel 2-D convolutions with different kernel sizes (1×1, 3×3, 5×5), averaged |
| TimesBlock | One TimesNet layer: FFT → fold into 2-D per period → Inception → unfold → weighted mix → residual |
| LayerNorm | Rescales each row's 128 numbers to mean 0, variance 1, with learned scale and shift |
| Projection head | Small MLP from `z` to a unit-length 128-vector used only by the contrastive loss |
| NT-Xent | SimCLR's contrastive loss: softmax over cosine similarities / τ, "pick your partner view" |
| Ablation | A run with one part removed or swapped (e.g. Transformer instead of TimesNet) to measure that part's value |
