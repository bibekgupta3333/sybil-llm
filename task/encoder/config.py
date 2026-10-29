"""Settings of one baseline run (one-class autoencoder or supervised encoder + head).

`EncoderTaskConfig` holds every knob and is written to `config.json` in the run folder.
"""

from __future__ import annotations

import dataclasses

DATA_DIR = "src/data/encoder_input/benign_gridsybil/T64"
RUNS_DIR = "task/encoder/runs"
N_FEATURES = 13
SEQ_LEN = 64
MODES = ("ae", "supervised")


@dataclasses.dataclass(frozen=True)
class EncoderTaskConfig:
    """Every setting of a run.

    Attributes:
        mode: "ae" (one-class reconstruction autoencoder, benign train only) or "supervised" (encoder + head, BCE).
        seed: model seed (weights, batch order). The fit / validation vehicle split always uses `split_seed`.
        data_dir: T64 encoder input (read only).
        runs_dir: parent of the run folders.
        cache_dir: decoded shards as .npy (built once from the JSON shards, then memory-mapped).
        val_frac: share of train sender vehicles held out as validation.
        split_seed: seed of that vehicle split (0, fixed for every run).
        mask_channel: append the mask as a 14th input channel (padding rows are zero in the 13 features).
        latent_dim: size of z.
        channels: encoder conv widths (13 -> c0 stride 1 -> c1 stride 2 -> c2 stride 2).
        head_hidden: hidden width of the supervised head.
        lr, weight_decay, batch_size, max_epochs, patience: Adam + early stopping on validation loss.
        plateau_factor, plateau_patience: ReduceLROnPlateau on validation loss.
        maha_ridge: ridge added to the z covariance diagonal before inversion.
        device: "auto" (mps > cuda > cpu), or an explicit torch device.
        eval_batch: batch size for scoring.
        max_windows: cap per split for quick checks (0 = all).
    """

    mode: str = "ae"
    seed: int = 0
    data_dir: str = DATA_DIR
    runs_dir: str = RUNS_DIR
    cache_dir: str = f"{RUNS_DIR}/cache"
    val_frac: float = 0.1
    split_seed: int = 0
    mask_channel: bool = True
    latent_dim: int = 16
    channels: tuple[int, int, int] = (32, 64, 64)
    head_hidden: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 5
    plateau_factor: float = 0.5
    plateau_patience: int = 3
    maha_ridge: float = 1e-6
    device: str = "auto"
    eval_batch: int = 4096
    max_windows: int = 0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

    @property
    def run_id(self) -> str:
        """Run folder name, e.g. `ae-seed0`, `supervised-seed2` (`-check` suffix when windows are capped)."""
        suffix = "-check" if self.max_windows else ""
        return f"{self.mode}-seed{self.seed}{suffix}"

    def to_dict(self) -> dict:
        """JSON-ready settings."""
        return dataclasses.asdict(self)
