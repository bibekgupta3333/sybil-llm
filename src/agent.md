# agent.md — rules for all work in `src/`

**Scope.** `src/` holds **all new code for the adopted TimesNet plan**: Stage 1 (self-supervised TimesNet
pretraining, now) and Stage 2 (few-shot fine-tuning + memory bank, deferred). Older v1 code (`scripts/`,
`models/transformer_model.ipynb`) stays where it is; do not extend it for the new plan.

**Read first, in this order:**
1. Root `CLAUDE.md` — project rules 1–7 (they apply here unchanged; this file adds `src/`-specific rules).
2. `docs/plan/clarification.md` — the adopted plan (Part 1) and why it differs from the original (Part 2).
3. `docs/plan/stage1-ssl-wbs.md` — the task list: S1.0–S1.4 now, S2.0–S2.3 deferred, with acceptance criteria.
4. This file.

**Status (2026-10-06):** decisions D1–D9 adopted; nothing in `src/` is implemented yet. Next tasks with no
`data/` write: S1.1.1 (leak audit) and S1.2.x (TimesNet encoder). S1.1.2 needs the user's OK to write
`data/prepared_receiver/`. Progress pages: `stage1-plan.html` (per task) and `tracker.html` (whole thesis).

**Code paradigm (user's choice):** about **90% object-oriented, 10% small pure functions**, Google Python Style,
readability first. OOP is preferred for readability and for managing a growing research codebase.

**Contents:** 1. Quick facts · 2. Folder layout and architecture · 3. Coding standards ·
4. Research guardrails in code · 5. Agent workflow

---

## 1. Quick facts (memorise; source: `docs/plan/clarification.md`, `docs/plan/stage1-ssl-wbs.md`)

| Item | Value |
|---|---|
| Beacon rate | **1 Hz** (Δτ median 1.000 s; `rcvTime == sendTime` in every type-3 record) |
| Sequence unit | network-heard pseudonym sequence, de-duplicated by `messageID` (D1) |
| T | **64** primary, **128** sensitivity; exactly T messages, no padding, equal windows per vehicle |
| d_in | **13** (list below) |
| Encoder | TimesNet, 4 TimesBlocks, d ∈ {128, 256, 512}, **d_ff = 64**, **3** Inception kernels, **top-k = 3** periods **per sample**, FFT bins f ≥ 2, dropout 0.1, no time-of-day embedding; outputs `(H: B×T×d, z: B×d)` via GAP over T |
| Param budget | 2.30M / 4.60M / 9.20M at d = 128 / 256 / 512 (±0.1%); Transformer ablation ≈ 1.19M at d = 128 |
| Masking | per sample, 50/50: random **exactly 25% of T**, or one block of round(r·T), r ~ U(0.20, 0.30) (13–19 steps at T = 64); learned mask token, no zero-fill |
| Reconstruction | 2-layer MLP on H[:, masked]; masked MSE on sender pos (window-relative), velocity, acceleration, log-Δτ; must beat linear interpolation on block masks |
| P1–P3 | injected violations, **p = 0.5** per window: P1 speed spike / position jump, P2 speed scaled without positions, P3 impossible sharp turn; BCE MLP heads on z |
| Augmentations | common-mode shift 3–5 m (sender + receiver), time stretch s ∈ [0.95, 1.05] (v·s, a·s², Δτ/s), crop 80–100% by subsampling. **Never** per-step GPS jitter or bare speed scaling (they mimic Sybil artifacts, F10) |
| InfoNCE | **τ = 0.1**, **batch 256**, projection d → d → 128 (discarded); hard negatives: same 50 m cell, same group, different pseudonym, \|Δt\| ≥ 10 min, **β = 0.5** |
| Joint loss | **λ1 = 1**, **λ3 = λ4 = λ5 = 0.3**, **λ2 ∈ {0.1, 0.3, 1}** + one uncertainty-weighting run; losses normalised (EMA); ≥ 3 seeds |
| Stage 2 (deferred) | train **A16** GridSybil, **A18** DoSRandom, **A19** DoSDisruptive + benign; **A17** DataReplay held out (zero-shot); **n ∈ {10, 20, 30, 50}**, ≥ 5 support sets per n; LR = S1 LR / 10; **K ∈ {5, 10, 20}**, θ on val |
| Splits | grouped by `(time window, physical vehicle)` pooled over the 4 scenarios; pretrain-val = 10% of train identities |

**The 13 adopted features** (whitelist, in this order — `settings.FEATURE_NAMES`):
`claimed_pos_x, claimed_pos_y, rx_pos_x, rx_pos_y, claimed_vel_x, claimed_vel_y, rx_vel_x, rx_vel_y,
claimed_acl_x, claimed_acl_y, range, bearing, log_dtau` — "rx" = the receiver's own type-2 fix at rcvTime (D2).
**Bearing = one value** (decided 2026-10-06): `atan2(dy, dx)` in radians, wrapped to (−π, π] via
`tools.wrap_angle`; never split into sin/cos — the input stays **13 features**.

---

## 2. Folder layout (flat — one file per topic)

**Simple on purpose.** `src/` is one flat folder: one file per topic, named after what it does, read top to bottom.
No sub-packages. Run everything from the repo root (`python -m src.prepare ...`; `from src.split import ...`).

```
src/
├── agent.md          # this file
├── config.json       # all settings ("input" and "split" sections)
├── settings.py       # InputConfig, SplitConfig (.from_file), FEATURE_NAMES (13), ATTACK_NAMES
├── tools.py          # small shared tools: range_bearing / wrap_angle / safe_log, leak checks
│                     #   (assert_label_free, assert_disjoint_groups), SafeWriter, Provenance
├── raw_logs.py       # read VeReMi trace files: RunFinder, TraceReader -> RunLogs
├── sequences.py      # messages -> pseudonym sequences -> 13 features -> exact-T windows;
│                     #         SequenceBuilder, FeatureMaker, Samples, DatasetWriter, Manifest
├── split.py          # VehicleSplitter / VehicleSplit (vehicle-grouped split), LeakAudit (F1)
├── checks.py         # ShortcutProbe (length / time of day at chance?), RetentionReport
├── prepare.py        # THE command: build -> split -> manifest -> probe (DataPreparation)
├── tests/            # conftest.py (synthetic raw data) + one test_<file>.py per file above
├── notebook/         # s1_checks.ipynb — plots only, imports from src
└── data/             # prepared S1.1 data (gitignored): X_T64, meta, labels, idx_*_rx, config.json
```

**Status (2026-10-06):** input representation lives in **one notebook, `src/pipeline/input_representation.ipynb`**
(user's choice). It writes JSON only: a mirror of `data/VeReMi-Dataset` under `src/data/prepared_data/` (one
prepared file per raw trace file, every received copy kept) plus `index.json`. Don't invent other layouts. The `.py` layout below is the plan
for later code (encoder, pretraining); it is not in use for input representation.

**Current layout (2026-10-07, user choice — code stays inside the notebooks):**

```
src/
  agent.md
  pipeline/
    input_representation.ipynb        all scenarios: raw -> prepared_data
    benign_gridsybil/
      encoder_input_T64.ipynb         first trial: benign + GridSybil links -> 64 x 13 windows + mask
  eda/
    eda_window_size.ipynb             all scenarios: link lengths, choice of T (read-only)
    benign_gridsybil/
      eda_encoder_input_T64.ipynb     first trial: EDA of its encoder input (read-only)
  data/                               gitignored
    prepared_data/                    all scenarios
    encoder_input/benign_gridsybil/T64/
```

Scope: benign + GridSybil is the first trial of the whole pipeline; the all-attack version gets sibling folders
(e.g. `pipeline/all_attacks/`, `data/encoder_input/all_attacks/`) once it works.

Notebooks find the repo root by walking up to `CLAUDE.md`, so they run from any folder.

**Files still to come** (add them flat, same style): `encoders.py` (S1.2 TimesNet + Transformer ablation),
`pretext.py` (masking, P1–P3 injectors, augmentations), `objectives.py` (losses, heads), `training.py`
(pretrainer, label-free checkpoint selection), `evaluation.py` (S1.4 probes), `stage2.py` (deferred).

**File discipline**
- Add to the file for that step before creating a new one. A new file only for a new step or when a file passes
  ~600 lines; say why. Never add folders inside `src/`.
- One test file per source file (`sequences.py` → `tests/test_sequences.py`); shared fixtures only in
  `tests/conftest.py`.
- Settings go in `config.json` (one section per component); a new knob = a new dataclass field with a default.
- Large outputs only in `src/data/` (gitignored) or `data/` (RULE 2, ask first) — never elsewhere in `src/`.

### WBS task → file → class

| WBS | File | Class / entry point |
|---|---|---|
| S1.1.1 | `split.py`, `prepare.py audit` | `LeakAudit` |
| S1.1.2 | `raw_logs.py`, `prepare.py build` | `RunFinder`, `TraceReader` |
| S1.1.3 | `sequences.py` | `FeatureMaker` |
| S1.1.4 | `sequences.py`, `config.json` | `SequenceBuilder` (`cut()`) |
| S1.1.5 | `split.py`, `prepare.py split` | `VehicleSplitter`, `VehicleSplit` |
| S1.1.6 | `checks.py`, `prepare.py probe`, `notebook/s1_checks.ipynb` | `ShortcutProbe`, `RetentionReport` |
| S1.1.7 | `sequences.py`, `prepare.py manifest` | `Manifest` |
| S1.0.4 | `requirements-train.txt` (repo root), `tools.py` | `Provenance` |
| S1.2.x | `encoders.py` (to write) | `TimesNetEncoder`, `TransformerEncoder` → `(H, z)` |
| S1.3.x | `pretext.py`, `objectives.py`, `training.py` (to write) | masking, P1–P3 injectors, augmentations, losses, `Stage1Pretrainer` |
| S1.4.x | `evaluation.py` (to write) | frozen-encoder probes, transfer, anomaly score |
| S2.x | `stage2.py` (to write, deferred) | few-shot fine-tuning, memory bank |

Shapes: x `(B, T, 13)`, T = 64; H `(B, T, d)`; z `(B, d)` = average of H over T.

### How the pieces connect

```
raw_logs.py -> sequences.py -> split.py -> checks.py          (prepare.py runs them in this order)
                     |
                     +-> data (X, meta, labels, idx_*) -> encoders.py -> pretext.py / objectives.py -> training.py
settings.py, tools.py <- used by every file (and import nothing from the step files)
```

- Collaborators are passed into `__init__`; settings objects are passed whole (no loose kwargs, no globals).
- Randomness only through a seeded `np.random.Generator` (or torch generator) created from the settings' seed.
- Inheritance at most one level below a base class; vary behaviour with settings, not subclasses.
- `prepare.py` (and future CLIs) only parse arguments, build objects and call them.

---

## 3. Coding standards (OOP 90% / functions 10%)

User's rule, verbatim intent: *"OOP 90% and functions 10%, such that it is readable."* Google Python
Style Guide. Readability over cleverness. House-style reference: the S1.1 files in `src/` (`sequences.py`, `split.py`)
and, for older idioms, `scripts/prepare_data.py` / `scripts/export_simulation_sample.py`
(`WindowStore`, `IdentityStitcher`, `StitchError`, `Exporter`) — match their idioms.

### 3.1 When a class is required vs when a function is allowed

**Class — required** for anything that:
- holds config or state (seeds, RNGs, paths, stats, buffers, open handles);
- is a pipeline stage, data source, model component, SSL objective/pretext task, evaluator,
  metrics recorder, checkpoint/run writer, or CLI entry orchestrator;
- has **more than one related operation** on the same data (e.g. `fit` + `transform`, `build` + `audit`).

**Module-level function — allowed only if ALL hold:** pure (no I/O, no globals, no RNG unless the
generator is passed in), stateless, ≤ ~20 lines, a math/geometry/conversion/formatting helper
(`range_bearing(dx, dy)`, `kmh_to_mps(v)`, `format_pct(x)`). These live in `src/tools.py`, are fully
typed, docstringed, and unit-tested. Private one-liners inside a module (`_steps_within(...)`) are fine
if they meet the same bar. No CLI `main()` logic beyond parse-args → build objects → `run()`.

```python
# GOOD — pure helper in src/tools.py
def range_bearing(dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (range_m, bearing_rad) for planar offsets of any shape."""
    return np.hypot(dx, dy), np.arctan2(dy, dx)

# BAD — "function" that hides state, I/O and several jobs
def build_and_save(path, seed=42):
    rng = np.random.default_rng(seed); x = np.load(path); ...; np.save("out.npy", x)
```

```python
# GOOD — stateful stage as a class with injected config
class WindowMasker:
    def __init__(self, config: MaskConfig, rng: np.random.Generator) -> None:
        self._config, self._rng = config, rng
    def mask(self, windows: np.ndarray) -> MaskedBatch: ...

# BAD — module-global state + free functions
_RNG = np.random.default_rng(0)
MASK_RATIO = 0.3
def mask(w): ...
```

### 3.2 Class conventions
- **Configs:** `@dataclasses.dataclass(frozen=True, slots=True)`; validate in `__post_init__` and raise
  `ValueError` with the offending value. Document fields under `Attributes:` (as `PipelineConfig` does).
  Configs are serialisable (`dataclasses.asdict`) and written next to results (RULE 4).
  ```python
  @dataclasses.dataclass(frozen=True, slots=True)
  class MaskConfig:
      """Attributes: ratio: fraction of timesteps masked, in (0, 1)."""
      ratio: float = 0.25
      def __post_init__(self) -> None:
          if not 0.0 < self.ratio < 1.0:
              raise ValueError(f"ratio must be in (0, 1), got {self.ratio}")
  ```
- **Interfaces:** `abc.ABC` + `@abc.abstractmethod` for families with shared code (objectives, encoders,
  evaluators); `typing.Protocol` for duck-typed seams (e.g. a `WindowSource` with `load() -> np.ndarray`).
- **Dependency injection:** collaborators (config, RNG/seed, data source, logger-free writers) are passed
  to `__init__`; classes never construct their own data paths or read env/global config.
- **Single responsibility;** few public methods, verb-named (`build`, `fit`, `encode`, `evaluate`,
  `write`); helpers are `_private`. Value objects get `__repr__`/`__eq__` from `@dataclass`.
- **No global mutable state, no singletons,** no module-level RNGs. Module constants are UPPER/`_UPPER`
  and immutable.
- **Inheritance depth ≤ 2** below an ABC / `nn.Module`; prefer composition.
- **`nn.Module` subclasses:** build sub-modules in `__init__`, keep `forward` short (≲15 lines) and
  delegating; no data loading, logging, or metric computation inside `forward`.
- **Custom exceptions** subclass a builtin (`class StitchError(ValueError)`), defined per module.
- **No god classes:** a class past ~300 lines or ~7 public methods → split it.

### 3.3 Google style essentials (enforced)
- **No step numbering in code:** never write "Step 1", "step 2", "(1) … (2) …" or similar in docstrings,
  comments, names or log messages. Describe what the code does instead (user preference, 2026-10-06).
- **Module docstring** first: one-line purpose, then the plan task it implements
  (`"Implements WBS S1.3.A1 (…)."`), what it reads/writes, and a `Usage:` line for runnable modules.
- **Docstrings:** Google style on every public module/class/method/function, with `Args:`, `Returns:`,
  `Raises:`. Document array/tensor shapes and dtypes inline: `windows: (B, T, 13) float32`,
  `Returns: logits (B, n_classes)`. Name feature order by reference to `config.json`, never re-list it.
- **Typing:** `from __future__ import annotations` in every module; full hints on all signatures
  (incl. `-> None`); `collections.abc` for `Sequence`/`Mapping`/`Iterator`; `X | None`, not `Optional`.
- **Naming:** `CapWords` classes, `snake_case` functions/vars/modules, `UPPER_SNAKE` constants,
  `_leading_underscore` private. No single-letter names except loop indices and math (`B, T, F`).
- **Imports:** three groups separated by a blank line — stdlib / third-party / local (`from src import tools`,
  `from src.split import VehicleSplitter`); import modules, not `*`; no `sys.path` hacks inside `src/` (tests use the package).
- **Logging:** `_LOG = logging.getLogger(__name__)`; `%`-style lazy args (`_LOG.info("wrote %d", n)`);
  never `print` outside notebooks. Configure handlers only in the entry-point `main()`.
- **Errors:** explicit exception types with messages naming the bad value/path; never bare `except:` or
  `except Exception: pass`; prefer raising over `assert` for data validation (asserts may be stripped).
- **Formatting:** line length **100** (some v1 lines overflow — don't copy that); f-strings for
  messages; `pathlib.Path` for all paths (no `os.path`, no string concatenation).
- **Determinism:** randomness only through an injected `np.random.Generator` / `torch.Generator` or an
  explicit `seed` config field; never `np.random.seed`/global torch seeding inside library code.

### 3.4 Testing
- `pytest`, under `src/tests/`, **one file per module**: `src/encoders.py` →
  `src/tests/test_encoders.py`, `src/settings.py` + `src/tools.py` → `src/tests/test_tools.py`. Inside a file, one
  `Test<ClassName>` class per production class. Shared synthetic fixtures live only in `src/tests/conftest.py`.
- **Synthetic fixtures only** — build tiny arrays in `tmp_path` (see `src/tests/conftest.py`).
  Unit tests never read `data/`, `models/*.pt`, or the network.
- Test what the WBS acceptance criteria state: output **shapes** (`(B, T, 13)` in → expected out),
  **determinism** (same seed → identical arrays/tensors; different seed → different), **invariants**
  (no NaN, masks hit the configured ratio, no label/`idx_val`/`idx_test` access in Stage 1 code paths,
  split disjointness by vehicle), and config validation (`pytest.raises(ValueError)`).
- Torch tests start with `torch = pytest.importorskip("torch")` so the EDA venv (no torch) still runs
  the rest; never install torch to make a test pass (RULE 4 — ask first).
- Run from the repo root: `.venv/bin/python -m pytest src/tests -q` (torch tests run only in the
  training env once S1.0.4 pins it). Keep unit tests < a few seconds each; mark slow ones
  `@pytest.mark.slow`.

### 3.5 Pre-finish readability checklist (run before reporting done)
1. Every new stateful thing is a class; every free function is pure, ≤ ~20 lines, in `src/tools.py`
   (or `_private`), typed and tested.
2. Module docstring names the WBS task id; every public API has Args/Returns/Raises and tensor shapes.
3. Configs are frozen dataclasses with `__post_init__` validation and are persisted with results + seed.
4. No `print`, bare `except`, globals/singletons, `os.path`, or lines > 100 chars.
5. No reads of `y_*`, attack codes, `idx_val`, or `idx_test` in Stage 1 pretraining code; no writes to
   `data/` without asking.
6. New tests pass (`.venv/bin/python -m pytest src/tests -q`); torch tests skip cleanly without torch.
7. A reviewer can name each class's single job from its name + docstring first line; anything over
   ~300 lines has been split.

---

## 4. Research guardrails in code

### Data access (RULE 2)
- Open everything under `data/` read-only. Never write, move or delete there without the user's explicit OK
  for that exact path (approved so far: none; `data/prepared_receiver/` is pending S1.1.2).
- Route every persistent array/table write through one writer class (`src/tools.py::SafeWriter`)
  that raises if the output exists unless `overwrite=True`, and refuses paths outside an allow-list.
- Large outputs go only to `src/data/` (gitignored) or an approved `data/` folder; never elsewhere in `src/`.

### Label-free Stage 1 (advisor constraint)
- Stage 1 classes (parser, features, encoder, masking, injectors, augment, sampler, losses, selector) take no
  label argument. Gate their data entry with `src/tools.py::assert_label_free`, which raises on `y_*` arrays, attack codes
  (`attackType`, A16–A19), `idx_val`, `idx_test` (or their `_rx` successors).
- Pretext tasks read train-split identities only (pretrain-val is a slice of train). In transfer runs, read
  only the source group's train identities.
- Probes (S1.4) are the only Stage 1 code that reads labels: fit on **train** labels only, score val/test.
- Checkpoint selection (S1.3.J2) is label-free and must assert it never opened labels or `idx_val`.
- Never select "benign-only" data, class-weight, or resample by label in pretraining or the memory bank.

### Never-inputs
- The feature extractor builds inputs from the 13-name whitelist above and raises on anything else.
- Never as inputs: `sender` / physical id, `*_noise` fields, ground-truth `pos`/`spd`, time of day (sin/cos or
  raw), run id, sequence length. Time of day may appear only as a *probe target* (S1.1.6).

### Leakage (RULE 3)
- Build splits by `(time window, physical vehicle)` across all 4 scenarios; the physical id is a grouping key
  only and is dropped before tensors are built.
- Assert at split build time: zero group overlap across train/val/test and across 0709/1416; persist the
  assertion result in the split manifest.
- Sample at identity level: one window per pseudonym per batch (S1.3.C3); the sampler never reads `sender`.
- Fit normalisation stats on train only; save them with the manifest; apply to val/test unchanged.
- Any split change: flag it **loudly** in the reply and record it in `docs/plan/research-plan.md` in the same
  turn. S1.1.5 is such a change.
- Treat all v1 numbers in `models/results/` as invalid (F1, F2): never quote them as results.

### Shortcut checks
- Cut every sample to exactly T with equal windows per vehicle (length-matched sampling; length AUC ≈ 0.5).
- S1.1.6 probes on length, time of day and run id must sit at chance; a failed probe blocks S1.3.

### Reproducibility (RULE 4)
- Every run writes a run directory with `config.json` (all hyperparameters, T, d, feature list, split manifest
  hash), `seed`, git commit hash (+ dirty flag), `pip_freeze.txt`, torch version + device, and `metrics.json`
  — the `models/roadfm_lite_config.json` + `results_summary.json` pattern, one directory per run.
- Seed `random`, `numpy`, `torch` (and MPS/CUDA) from the settings' seed (one seeded generator per component); enable deterministic algorithms where
  the backend supports them; log any op that is not.
- The torch env is unpinned until S1.0.4 lands `requirements-train.txt`. Flag this in every training reply; no
  run is citable before it.
- No `pip install` / `brew install` / new dependency without the user's OK.

### Integrity (RULE 1)
- Never tune on test. Stage 1 settings are chosen label-free; K, θ and n on val; test is read **once** per
  final evaluation (log the read).
- Report what the model produces, including nulls, failed probes and per-family zeros.
- Fix genuine bugs openly (state the bug and its effect on earlier numbers); never post-process results.

### Notebooks (RULE 5)
- First cell of every `src/notebook/*.ipynb`: `assert Path("CLAUDE.md").is_file() and Path("data").is_dir(), "run from repo root"`.
- Import all logic from `src`; notebooks hold only loading, calling and plotting. No class or model code.

### Git and scope (RULES 6, 7)
- Never commit, push, `reset --hard` or `clean` unless asked. Staging and `git diff` are fine.
- Out-of-scope ideas (OSM map matching, FL simulation, full 19-type download, new baselines) get one line,
  "out of scope: …", and no code.

---

## 5. Agent workflow (per task)

1. **Pick one WBS id** (e.g. S1.2.2) from `docs/plan/stage1-ssl-wbs.md`; check its `Dep` column is done.
   If it needs a `data/` write or a download, ask first and stop.
2. **Read its acceptance criterion** and the plan text it cites (`clarification.md` § and decision D#).
   Note the deliverable's `src/` path (see "WBS task → src path" above).
3. **Design the interface first**: classes, dataclass configs, public methods, shapes, Google docstrings.
   OOP-first (~90% classes, ~10% small pure functions). Show it in the reply before large implementations.
4. **Write tests** in `src/tests/` that encode the acceptance criterion (shapes, exact ratios, determinism,
   guard raises on labels, zero split overlap) plus a tiny synthetic fixture. No real-data reads in unit tests.
5. **Implement** to the interface. Settings live in `src/config.json`; entry points are flat files like `src/prepare.py`.
6. **Run the tests** (`python -m pytest src/tests -q` from the repo root) and any acceptance script. Fix until
   green; never weaken a test to pass.
7. **Record evidence**: test output, measured numbers, run directory path. Numbers come from the run, not the
   plan.
8. **Update status in the same turn**: the task row/notes in `docs/plan/stage1-ssl-wbs.md` and the
   "Stage 1 plan" section of `docs/plan/research-plan.md` (decisions, split changes, nulls).
9. **Remind the user** to tick the task in `stage1-plan.html` and `tracker.html`; flag RULE 4 if training ran.

**Definition of done:** acceptance criterion met with evidence quoted in the reply · tests in `src/tests/`
pass · no label / `idx_val` / `idx_test` access outside S1.4 probes · no unapproved writes under `data/` or
large files under `src/` · run dir (config, seed, commit, pip freeze, metrics) present for anything trained ·
WBS + research-plan updated · user reminded to tick both trackers · nothing committed.
