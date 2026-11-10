"""Settings of one autoencoder run; written to `config.json` in the run folder."""

from __future__ import annotations

import dataclasses

DATA_DIR = "src/data/encoder_input/benign_gridsybil/T48"
RUNS_DIR = "task/autoencoder/runs"
SEQ_LEN = 48
N_FEATURES = 13


@dataclasses.dataclass(frozen=True)
class AutoencoderConfig:
    """Every setting of a run.

    Attributes:
        data_dir: T48 encoder input (read only).
        runs_dir: parent of the run folders.
        seq_len, n_features: window shape (48 x 13).
        latent_dim: size of z.
        lr, weight_decay: Adam.
        batch_size: training batch size.
        max_epochs: upper bound on epochs.
        patience: early stopping on validation loss (epochs without improvement).
        plateau_factor, plateau_patience: ReduceLROnPlateau on validation loss.
        min_delta: relative validation-loss decrease needed to count as an improvement (resets early stopping);
            0 = any decrease (the setting of the `seed0` run).
        val_frac: share of train sender vehicles held out as validation (split seed = `seed`, as in the `seed0`
            run); 0 reads the on-disk `val` split of `data_dir` instead.
        seed: weights and batch order (the train / val / test split is fixed on disk).
        device: "auto" (mps > cuda > cpu) or an explicit torch device.
        eval_batch: batch size for validation and scoring.
        workers: processes that decode the JSON shards.
        log_feature_error: also write `feature_history.jsonl` (per-feature train / val masked MSE per epoch) into a
            run folder of its own (`seed<k>_featlog`); off by default.
    """

    data_dir: str = DATA_DIR
    runs_dir: str = RUNS_DIR
    seq_len: int = SEQ_LEN
    n_features: int = N_FEATURES
    latent_dim: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 300
    patience: int = 10
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    min_delta: float = 0.0
    seed: int = 0
    val_frac: float = 0.1
    device: str = "auto"
    eval_batch: int = 4096
    workers: int = 8
    log_feature_error: bool = False

    @property
    def run_id(self) -> str:
        """Run folder name, e.g. `seed0` (`seed0_featlog` with per-feature logging)."""
        return f"seed{self.seed}" + ("_featlog" if self.log_feature_error else "")

    def to_dict(self) -> dict:
        """JSON-ready settings."""
        return dataclasses.asdict(self)
