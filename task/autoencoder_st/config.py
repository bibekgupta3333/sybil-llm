"""Settings of one supervised autoencoder run; written to `config.json` in the run folder."""

from __future__ import annotations

import dataclasses

DATA_DIR = "src/data/encoder_input/benign_gridsybil/T48_tvt"
RUNS_DIR = "task/autoencoder_st/runs"
SEQ_LEN = 48
N_FEATURES = 13
AUTOENCODER_PARAMS = 71_741
HEAD_PARAMS = 577
TOTAL_PARAMS = AUTOENCODER_PARAMS + HEAD_PARAMS


@dataclasses.dataclass(frozen=True)
class SupervisedAEConfig:
    """Every setting of a run.

    Attributes:
        data_dir: T48_tvt encoder input (read only; train / val / test fixed on disk by sender vehicle).
        runs_dir: parent of the run folders.
        seq_len, n_features: window shape (48 x 13).
        latent_dim: size of z.
        head_hidden, head_dropout: classifier head on z (Linear -> ReLU -> Dropout -> Linear -> logit).
        recon_weight: lambda of the masked reconstruction MSE in loss = BCE + lambda * MSE.
        lr, weight_decay: Adam.
        batch_size: training batch size.
        max_epochs: upper bound on epochs.
        grad_clip: max global L2 gradient norm (the pre-clip norm is logged).
        patience: early stopping on validation AUROC (epochs without improvement).
        min_delta: absolute AUROC increase needed to count as an improvement.
        plateau_factor, plateau_patience: ReduceLROnPlateau on validation AUROC (mode max).
        target_fpr: benign false positive rate on val that fixes the second alarm threshold.
        seed: weights, dropout and batch order.
        device: "auto" (mps > cuda > cpu) or an explicit torch device.
        eval_batch: batch size for validation and scoring.
        workers: processes that decode the JSON shards.
    """

    data_dir: str = DATA_DIR
    runs_dir: str = RUNS_DIR
    seq_len: int = SEQ_LEN
    n_features: int = N_FEATURES
    latent_dim: int = 16
    head_hidden: int = 32
    head_dropout: float = 0.2
    recon_weight: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 100
    grad_clip: float = 1.0
    patience: int = 10
    min_delta: float = 1e-4
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    target_fpr: float = 0.05
    seed: int = 0
    device: str = "auto"
    eval_batch: int = 4096
    workers: int = 8

    @property
    def run_id(self) -> str:
        """Run folder name, e.g. `seed0`."""
        return f"seed{self.seed}"

    def to_dict(self) -> dict:
        """JSON-ready settings."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SupervisedAEConfig":
        """Inverse of `to_dict`."""
        return cls(**d)
