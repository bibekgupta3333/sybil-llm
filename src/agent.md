# agent.md — rules for all work in `src/`

**Scope.** `src/` holds **all new code for the adopted TimesNet plan**: self-supervised TimesNet
pretraining (now) and few-shot fine-tuning + memory bank (deferred). Older v1 code (`scripts/`,
`models/transformer_model.ipynb`) stays where it is; do not extend it for the new plan.

**Read first, in this order:**
1. Root `CLAUDE.md` — project rules 1–7 (they apply here unchanged; this file adds `src/`-specific rules).
2. `docs/plan/clarification.md` — the adopted plan (Part 1) and why it differs from the original (Part 2).
3. `docs/plan/stage1-ssl-wbs.md` — the task list: S1.0–S1.4 now, S2.0–S2.3 deferred, with acceptance criteria.
4. This file.

**Status (2026-10-08):** input representation and the benign + GridSybil encoder input are built (notebooks);
the self-supervised pretraining package `src/model/benign_gridsybil/timesnet/` is built (TimesNet only, d = 128,
d_ff = 64, D12) and `--check` passes; nothing is trained. The current state is in the root `agent.md` §1–§6.
Progress pages: `stage1-plan.html` (per task) and `tracker.html` (whole thesis).

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
| T | **64** primary, **128** sensitivity; links cropped at 64, every window padded to 64 rows + mask (1 = real) |
| d_in | **13** (list below) |
| Encoder | TimesNet, 4 TimesBlocks, **d = 128, d_ff = 64** (the plan's bottleneck, D12; d_ff = 128 and d = 256 / 512 are ablations), **3** Inception kernels, **top-k = 3** periods **per sample**, FFT bins f ≥ 2, dropout 0.1, no time-of-day embedding; outputs `(H: B×T×d, z: B×d)` via GAP over T |
| Param budget | encoder exactly 2,301,312 at d = 128, d_ff = 64 (D12); heads 101,136 (P1–P3 hidden width d); 4.60M with d_ff = 128 (ablation); Transformer ablation removed from the code (D11) |
| Masking | per sample, 50/50: random **exactly 25% of T**, or one block of round(r·T), r ~ U(0.20, 0.30) (13–19 steps at T = 64); learned mask token, no zero-fill |
| Reconstruction | 2-layer MLP on H[:, masked]; masked MSE on sender pos (window-relative), velocity, acceleration, log-Δτ; must beat linear interpolation on block masks |
| P1–P3 | injected violations, **p = 0.5** per window: P1 speed spike / position jump, P2 speed scaled without positions, P3 impossible sharp turn; BCE MLP heads on z |
| Augmentations | common-mode shift 3–5 m (sender + receiver), time stretch s ∈ [0.95, 1.05] (v·s, a·s², Δτ/s), crop 80–100% by subsampling. **Never** per-step GPS jitter or bare speed scaling (they mimic Sybil artifacts, F10) |
| InfoNCE | **τ = 0.1**, **batch 256**, projection d → d → 128 (discarded); hard negatives: same 50 m cell, same group, different pseudonym, \|Δt\| ≥ 10 min, **β = 0.5** |
| Joint loss | **λ1 = 1**, **λ3 = λ4 = λ5 = 0.3**, **λ2 ∈ {0.1, 0.3, 1}** + one uncertainty-weighting run; losses normalised (EMA); ≥ 3 seeds |
| Few-shot fine-tuning (deferred) | train **A16** GridSybil, **A18** DoSRandom, **A19** DoSDisruptive + benign; **A17** DataReplay held out (zero-shot); **n ∈ {10, 20, 30, 50}**, ≥ 5 support sets per n; LR = pretraining LR / 10; **K ∈ {5, 10, 20}**, θ on val |
| Splits | grouped by `(time window, physical vehicle)` pooled over the 4 scenarios (D8); benign + GridSybil encoder input: **90 / 10 train / test by sender vehicle** (D8′); pretraining check set = 10% of train vehicles; no val split |

**The 13 adopted features** (whitelist, in this order — `timesnet/data.py::FEATURES`, and `features` in each
`metadata.json`):
`claimed_pos_x, claimed_pos_y, rx_pos_x, rx_pos_y, claimed_vel_x, claimed_vel_y, rx_vel_x, rx_vel_y,
claimed_acl_x, claimed_acl_y, range, bearing, log_dtau` — "rx" = the receiver's own type-2 fix at rcvTime (D2).
**Bearing = one value** (decided 2026-10-06): `atan2(dy, dx)` in radians, wrapped to (−π, π] by
`wrap_angle` in `pipeline/input_representation.ipynb`; never split into sin/cos — the input stays **13 features**.

---

## 2. Folder layout

Run everything from the repo root (`python -m src.model.benign_gridsybil.timesnet.train ...`). Data code stays inside
notebooks as classes (user choice, 2026-10-07); model code is a package of small modules next to the notebook that explains it.

```
src/
  agent.md · README.md
  pipeline/                           notebooks that write data (classes inside the notebook)
    input_representation.ipynb        all scenarios: raw -> data/prepared_data/
    benign_gridsybil/
      encoder_input_T64.ipynb         first trial: benign + GridSybil links -> 64 x 13 windows + mask
    all/
      encoder_input_T24.ipynb         all attacks, T = 24, compact gzipped JSON (not for training until F14)
  eda/                                read-only analysis notebooks
    eda_window_size.ipynb             all scenarios: link lengths, choice of T
    benign_gridsybil/
      eda_encoder_input_T64.ipynb     first trial: EDA of its encoder input
  model/benign_gridsybil/
    encoder_T64.ipynb                 the encoder explained and checked (imports timesnet/encoder.py)
    timesnet/                         self-supervised pretraining package (TimesNet only)
      config.py data.py encoder.py heads.py views.py losses.py monitors.py train.py
      pretrain_monitor.ipynb          read-only plots of a run
  tests/                              pytest for the timesnet package (synthetic tensors, CPU)
  data/                               gitignored
    prepared_data/                    all scenarios
    encoder_input/benign_gridsybil/T64/ · encoder_input/all/T24/
  runs/pretraining/benign_gridsybil/T64/<run_id>/   gitignored run folders
```

Scope: benign + GridSybil is the first trial of the whole pipeline; the all-attack version uses sibling `all/`
folders (`pipeline/all/`, `data/encoder_input/all/`). Naming: `<subset>/` folder (`benign_gridsybil`, `all`) +
`_T<n>` file suffix. Tests for `scripts/` (legacy v1 tools, exporters, `hf_hub.py`) live in `scripts/tests/`.

Notebooks find the repo root by walking up to `CLAUDE.md`, so they run from any folder.

**Files still to come:** S1.4 frozen-encoder probes and few-shot fine-tuning (S2.x, deferred) go in new modules
next to the package they use; say why in the reply.

**File discipline**
- Add to the module for that job before creating a new one. A new module only for a new job or when a file passes
  ~600 lines; say why.
- Settings go in a frozen dataclass (`timesnet/config.py::PretrainConfig`); a new knob = a new field with a default.
- Large outputs only in `src/data/`, `src/runs/` (both gitignored) or `data/` (RULE 2, ask first).

### WBS task → file → class

| WBS | File | Class / entry point |
|---|---|---|
| S1.1.x | `pipeline/input_representation.ipynb`, `pipeline/benign_gridsybil/encoder_input_T64.ipynb` | notebook code (links, features, crop + pad + mask, vehicle split, normalisation) |
| S1.0.4 | `requirements-train.txt` (repo root), `.venv-train` | pinned torch env; `train.py` writes `env.json` per run |
| S1.2.x | `timesnet/encoder.py` | `TimesNetEncoder` → `(H, z)` (Transformer ablation removed, D11) |
| S1.3.x | `timesnet/views.py`, `heads.py`, `losses.py`, `data.py`, `train.py`, `monitors.py` | `ReconMasker`, `Augmenter`, `ViolationInjector`, `PretrainingModel`, `Objective`, `LabelFirewall`, `Trainer`, `Evaluator` |
| S1.4.x | to write | frozen-encoder probes, transfer, anomaly score |
| S2.x | to write (deferred) | few-shot fine-tuning, memory bank |

Shapes: x `(B, T, 13)` + mask `(B, T)`, T = 64; H `(B, T, d)`; z `(B, d)` = masked average of H over real rows.

### How the pieces connect

```
input_representation.ipynb -> encoder_input_T64.ipynb -> src/data/encoder_input/benign_gridsybil/T64/
  -> timesnet/data.py (LabelFirewall, FeatureShards) -> views.py -> encoder.py + heads.py -> losses.py
  -> train.py (Trainer, label-free check-set selection) -> src/runs/pretraining/... -> pretrain_monitor.ipynb
```

- Collaborators are passed into `__init__`; settings objects are passed whole (no loose kwargs, no globals).
- Randomness only through a seeded `np.random.Generator` (or torch generator) created from the settings' seed.
- Inheritance at most one level below a base class; vary behaviour with settings, not subclasses.
- `train.py` (and future CLIs) only parse arguments, build objects and call them.

---

## 3. Coding standards (OOP 90% / functions 10%)

User's rule, verbatim intent: *"OOP 90% and functions 10%, such that it is readable."* Google Python
Style Guide. Readability over cleverness. House-style reference: the pretraining package `src/model/benign_gridsybil/timesnet/`
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
(`range_bearing(dx, dy)`, `kmh_to_mps(v)`, `format_pct(x)`). They live next to the class that uses them (a
`_private` helper in the module, or a function cell in the notebook), fully typed and docstringed. Private one-liners inside a module (`_steps_within(...)`) are fine
if they meet the same bar. No CLI `main()` logic beyond parse-args → build objects → `run()`.

```python
# GOOD — pure helper next to the code that uses it
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
  `ValueError` with the offending value. Document fields under `Attributes:` (as `PretrainConfig` in `timesnet/config.py` and `PipelineConfig` in `scripts/prepare_data.py` do).
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
  `Returns: logits (B, n_classes)`. Name feature order by reference to `FEATURES` / `metadata.json`, never re-list it.
- **Typing:** `from __future__ import annotations` in every module; full hints on all signatures
  (incl. `-> None`); `collections.abc` for `Sequence`/`Mapping`/`Iterator`; `X | None`, not `Optional`.
- **Naming:** `CapWords` classes, `snake_case` functions/vars/modules, `UPPER_SNAKE` constants,
  `_leading_underscore` private. No single-letter names except loop indices and math (`B, T, F`).
- **Imports:** three groups separated by a blank line — stdlib / third-party / local
  (`from src.model.benign_gridsybil.timesnet import encoder`); import modules, not `*`; no `sys.path` hacks inside `src/` (tests use the package).
- **Logging:** `_LOG = logging.getLogger(__name__)`; `%`-style lazy args (`_LOG.info("wrote %d", n)`);
  never `print` outside notebooks. Configure handlers only in the entry-point `main()`.
- **Errors:** explicit exception types with messages naming the bad value/path; never bare `except:` or
  `except Exception: pass`; prefer raising over `assert` for data validation (asserts may be stripped).
- **Formatting:** **black**, line length **120** (`pyproject.toml`; run `npm run format` / `npm run format:check`, or format-on-save via `.vscode/settings.json`); f-strings for
  messages; `pathlib.Path` for all paths (no `os.path`, no string concatenation).
- **Determinism:** randomness only through an injected `np.random.Generator` / `torch.Generator` or an
  explicit `seed` config field; never `np.random.seed`/global torch seeding inside library code.

### 3.4 Testing
- `pytest`; config in `pyproject.toml` (`testpaths = scripts/tests, src/tests`). Tests for the pretraining package go in
  `src/tests/`, **one file per module** (`timesnet/encoder.py` → `src/tests/test_encoder.py`, `data.py` →
  `test_data.py`, `losses.py` → `test_losses.py`, `views.py` → `test_views.py`); shared synthetic fixtures live in
  `src/tests/conftest.py`. Tests for `scripts/` live in `scripts/tests/`.
- **Synthetic fixtures only** — tiny tensors / arrays built in the test or in `tmp_path`. Unit tests never read
  `data/`, `src/data/`, `models/*.pt`, run folders or the network.
- Test what the WBS acceptance criteria state: output **shapes** (`(B, T, 13)` in → expected out),
  **determinism** (same seed → identical arrays/tensors; different seed → different), **invariants**
  (no NaN, padding ignored exactly, masks hit the configured ratio, no label / `_info` access in pretraining code
  paths, split disjointness by vehicle), and config validation (`pytest.raises(ValueError)`).
- Run from the repo root: `npm test` (= `.venv-train/bin/python -m pytest`, both suites; the training env has torch
  and pytest). Keep unit tests < a few seconds each; mark slow ones `@pytest.mark.slow`. Never install a package to
  make a test pass without the user's OK (RULE 4).

### 3.5 Pre-finish readability checklist (run before reporting done)
1. Every new stateful thing is a class; every free function is pure, ≤ ~20 lines, `_private` or next to its
   user, typed and tested.
2. Module docstring names the WBS task id; every public API has Args/Returns/Raises and tensor shapes.
3. Configs are frozen dataclasses with `__post_init__` validation and are persisted with results + seed.
4. No `print` outside notebooks, bare `except`, globals/singletons, `os.path`; `npm run format:check` passes
   (black, line length 120).
5. No reads of `y_*`, attack codes, `idx_val`, or `idx_test` in pretraining code; no writes to
   `data/` without asking.
6. New tests pass (`npm test`).
7. A reviewer can name each class's single job from its name + docstring first line; anything over
   ~300 lines has been split.

---

## 4. Research guardrails in code

### Data access (RULE 2)
- Open everything under `data/` read-only. Never write, move or delete there without the user's explicit OK
  for that exact path (approved so far: none).
- Writers check their output folder (refuse to overwrite unless asked) and never write outside `src/data/` or
  `src/runs/`.
- Large outputs go only to `src/data/` (gitignored) or an approved `data/` folder; never elsewhere in `src/`.

### Label-free pretraining (advisor constraint)
- Pretraining classes (parser, features, encoder, masking, injectors, augment, sampler, losses, selector) take no
  label argument. Gate their data entry with `timesnet/data.py::LabelFirewall`, which refuses `_info` label files and `val` / `test`;
  the rule covers `y_*` arrays, attack codes (`attackType`, A16–A19), `idx_val`, `idx_test` (or their `_rx` successors).
- Pretext tasks read train-split identities only (the check set is a slice of train vehicles). In transfer runs, read
  only the source group's train identities.
- Probes (S1.4) are the only pretraining code that reads labels: fit on **train** labels only, score val/test.
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
- The torch env is pinned in `requirements-train.txt` (`.venv-train`, 2026-10-07); every run saves `config.json` and
  `env.json` (commit + dirty flag, Python / torch / numpy versions, device). A run whose env differs from the pin is not citable.
- No `pip install` / `brew install` / new dependency without the user's OK.

### Integrity (RULE 1)
- Never tune on test. Pretraining settings are chosen label-free; K, θ and n on val; test is read **once** per
  final evaluation (log the read).
- Report what the model produces, including nulls, failed probes and per-family zeros.
- Fix genuine bugs openly (state the bug and its effect on earlier numbers); never post-process results.

### Notebooks (RULE 5)
- Notebooks in `src/` find the repo root by walking up to `CLAUDE.md`; a new notebook without that does this in its first cell: `assert Path("CLAUDE.md").is_file() and Path("data").is_dir(), "run from repo root"`.
- Data and EDA code lives in the notebooks as classes (user choice); model code lives in the `timesnet/` package and
  the model notebooks import it instead of copying it.

### Git and scope (RULES 6, 7)
- Never commit, push, `reset --hard` or `clean` unless asked. Staging and `git diff` are fine.
- Out-of-scope ideas (OSM map matching, FL simulation, full 19-type download, new baselines) get one line,
  "out of scope: …", and no code.

---

## 5. Agent workflow (per task)

1. **Pick one WBS id** (e.g. S1.2.2) from `docs/plan/stage1-ssl-wbs.md`; check its `Dep` column is done.
   If it needs a `data/` write or a download, ask first and stop.
2. **Read its acceptance criterion** and the plan text it cites (`clarification.md` § and decision D#).
   Note the deliverable's `src/` path (see "WBS task → file → class" above).
3. **Design the interface first**: classes, dataclass configs, public methods, shapes, Google docstrings.
   OOP-first (~90% classes, ~10% small pure functions). Show it in the reply before large implementations.
4. **Write tests** in `src/tests/` that encode the acceptance criterion (shapes, exact ratios, determinism,
   guard raises on labels, zero split overlap) plus a tiny synthetic fixture. No real-data reads in unit tests.
5. **Implement** to the interface. Settings live in a frozen config dataclass; entry points are modules like `timesnet/train.py`.
6. **Run the tests** (`npm test` from the repo root) and any acceptance script. Fix until
   green; never weaken a test to pass.
7. **Record evidence**: test output, measured numbers, run directory path. Numbers come from the run, not the
   plan.
8. **Update status in the same turn**: the task row/notes in `docs/plan/stage1-ssl-wbs.md` and the
   "Pretraining plan" section of `docs/plan/research-plan.md` (decisions, split changes, nulls).
9. **Remind the user** to tick the task in `stage1-plan.html` and `tracker.html`; flag RULE 4 if training ran.

**Definition of done:** acceptance criterion met with evidence quoted in the reply · tests in `src/tests/`
pass · no label / `idx_val` / `idx_test` access outside S1.4 probes · no unapproved writes under `data/` or
large files under `src/` · run dir (config, seed, commit, pip freeze, metrics) present for anything trained ·
WBS + research-plan updated · user reminded to tick both trackers · nothing committed.
