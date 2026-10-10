"""Settings of a self-supervised pretraining run.

`PretrainConfig` holds every knob (data, encoder sizes, objectives, optimisation, evaluation, resources) and is
saved to `config.json` in the run folder. Presets: `joint` (the plan, D7) and `recon_only` (sanity baseline).
Every other module takes it as `settings`. `DATASETS` lists the two training sets (`--dataset grid | all`): where
the encoder input lives, its window length and where the run folders go.
"""

from __future__ import annotations

import dataclasses
import math

F14_NOTE = (
    "F14 open: evaluation on all/T24 test waits on the F14 decision (GridSybil_0709 near-copies across the "
    "vehicle split); pretraining reads train only"
)


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """One training set: encoder input folder, window length, parent of its run folders, default run name."""

    name: str
    data_dir: str
    seq_len: int
    runs_dir: str
    note: str = ""

    @property
    def run_id(self) -> str:
        """Fixed output name of the full run (`model-grid`, `model-all`)."""
        return f"model-{self.name}"


DATASETS: dict[str, DatasetSpec] = {
    "grid": DatasetSpec(
        "grid",
        "src/data/encoder_input/benign_gridsybil/T64",
        64,
        "src/runs/pretraining/benign_gridsybil/T64",
    ),
    "all": DatasetSpec(
        "all",
        "src/data/encoder_input/all/T24",
        24,
        "src/runs/pretraining/all/T24",
        F14_NOTE,
    ),
}


@dataclasses.dataclass
class PretrainConfig:
    """Every knob of the run; saved to config.json.

    Width d = 128 with inner width d_ff = 64 (the professor's plan; decision D12, 2026-10-08): encoder 2,301,312
    parameters. d_ff = 128 or d = 256 / 512 are ablations for later (d_ff = d at d = 512 is 73.4M parameters and
    ~16x the compute per step).
    """

    # data
    dataset: str = "grid"  # CLI --dataset: grid (benign + GridSybil, T = 64) or all (all attacks, T = 24)
    data_dir: str = "src/data/encoder_input/benign_gridsybil/T64"
    runs_dir: str = "src/runs/pretraining/benign_gridsybil/T64"  # CLI --runs-dir (e.g. a mounted volume)
    shards_per_split: int | None = None  # None = all shards; smoke uses 1
    workers: int = 8
    holdout_frac: float = 0.10  # train / test layout: share of train sender vehicles kept as the check set
    # TimesNet sizes (same as src/model/benign_gridsybil/encoder_T64.ipynb)
    seq_len: int = 64
    n_features: int = 13
    d_model: int = 128
    d_ff: int = 64  # inner width of each TimesBlock (bottleneck, D12 = the plan)
    n_blocks: int = 4
    top_k: int = 3
    n_kernels: int = 3
    dropout: float = 0.1
    min_freq: int = 2
    proj_dim: int = 128
    # objectives
    use_recon: bool = True
    use_nce: bool = True
    use_physics: bool = True
    lambda_recon: float = 1.0
    lambda_nce: float = 0.3
    lambda_p1: float = 0.3
    lambda_p2: float = 0.3
    lambda_p3: float = 0.3
    recon_ratio: float = 0.25  # random masking: exactly this share of real rows
    block_ratio: tuple[float, float] = (0.20, 0.30)
    temperature: float = 0.1
    inject_prob: float = 0.5
    min_rows_aux: int = 4  # P and NT-Xent losses only for windows with >= this many real rows
    # data scaling (EDA fix: GridSybil outliers up to z = 30 inflated the train std of these features)
    robust_features: tuple[str, ...] = ("claimed_pos_x", "claimed_pos_y", "range")  # median / IQR + soft tail
    soft_clip: float = 5.0  # beyond |z| = 5 values are compressed logarithmically (order kept, still visible)
    # contrastive false negatives (EDA fix): windows of the same broadcasting pseudonym in the same run are
    # copies or crops of the same broadcasts (and pseudonym 1 is shared by ghosts), so they are never negatives
    mask_same_broadcast: bool = True
    shift_m: tuple[float, float] = (3.0, 5.0)
    stretch: tuple[float, float] = (0.95, 1.05)
    crop: tuple[float, float] = (0.80, 1.00)
    jump_m: tuple[float, float] = (50.0, 150.0)
    speed_up: tuple[float, float] = (2.0, 3.0)
    speed_down: tuple[float, float] = (0.2, 0.5)
    turn_rad: tuple[float, float] = (math.pi / 2, math.pi)
    min_speed_mps: float = 2.0  # P2 / P3 need a moving sender
    norm_warmup_steps: int = 200  # loss scales are running means until here, then frozen
    # optimisation
    batch_size: int = 256  # the plan's value; 128 halves the windows per step and roughly doubles the steps
    # speed: each batch holds windows of (almost) the same length, so a TimesBlock runs a few FFT / conv groups
    # instead of ~50; batches are still random (shuffled within each length, batch order shuffled every epoch),
    # and NT-Xent negatives then cannot be told apart by length
    length_bucketed: bool = True
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_clip: float = 1.0
    max_epochs: int = 20
    max_steps: int | None = None
    max_hours: float = 8.0
    patience: int = 5
    # evaluation / logging
    eval_windows: int = 8192
    eval_batch_size: int = 512
    log_every: int = 50
    seed: int = 0
    # resources
    device: str = "auto"
    cpu_threads: int = 0  # torch CPU threads (0 = all cores, the torch default)
    prefetch_batches: int = 3  # batches prepared by a background thread while the GPU trains (0 = off)
    mem_gb: float = 22.0
    resource_guard: bool = True  # False (CLI --no-resource-guard): no MPS memory cap and no memory-budget stop

    @classmethod
    def preset(cls, name: str) -> "PretrainConfig":
        """Named presets: `joint` (default plan, D7) and `recon_only` (sanity baseline)."""
        if name == "joint":
            return cls()
        if name == "recon_only":
            return cls(use_nce=False, use_physics=False)
        raise ValueError(f"unknown preset {name!r}")

    def use_dataset(self, name: str) -> "PretrainConfig":
        """Points the data, window length and run folder parent at one of `DATASETS` (returns self)."""
        if name not in DATASETS:
            raise ValueError(f"unknown dataset {name!r}; choose one of {sorted(DATASETS)}")
        spec = DATASETS[name]
        self.dataset, self.data_dir, self.seq_len, self.runs_dir = spec.name, spec.data_dir, spec.seq_len, spec.runs_dir
        return self

    @classmethod
    def from_dict(cls, values: dict) -> "PretrainConfig":
        """Rebuilds settings from a saved config.json (lists back to tuples; unknown keys from older runs dropped)."""
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items() if k in known})

    def validate(self) -> None:
        """Fails early on settings that cannot work."""
        if not (self.use_recon or self.use_nce or self.use_physics):
            raise ValueError("at least one objective must be on")
        if self.mem_gb < 4:
            raise ValueError("mem_gb must leave room for the data (~1 GB) and the model")
        if self.dataset not in DATASETS:
            raise ValueError(f"unknown dataset {self.dataset!r}")
        if self.batch_size < 2:
            raise ValueError("NT-Xent needs batch_size >= 2")
