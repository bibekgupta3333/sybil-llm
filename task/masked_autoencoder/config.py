"""Settings of one masked-autoencoder run; written to `config.json` in the run folder.

The model re-implements pytorch-widedeep 1.7.0's encoder-decoder self-supervised pretraining (`EncoderDecoderTrainer`
with `TabMlp` / `TabResnet` encoders, `RandomObfuscator`, `EncoderDecoderLoss`) in plain PyTorch; see `model.py` for the
source lines. The data protocol (splits, early stopping patience, LR schedule) of each window size is the one of the
one-class ConvAE package it is compared with:

    T = 64  task/encoder (mode ae)        benign_gridsybil/T64, benign train, val = 10% of benign train vehicles
                                          (VehicleSplit, split seed 0), test = benign + GridSybil test
    T = 48  task/autoencoder_benign_t48   benign_gridsybil/T48_benign, on-disk train / val / test + test_gridsybil
    T = 32  task/autoencoder_benign_t32   all/T32_benign, on-disk train / val / test + four test_<attack> splits
"""

from __future__ import annotations

import dataclasses

RUNS_DIR = "task/masked_autoencoder/runs"
N_FEATURES = 13
ENCODERS = ("mlp", "resnet")
EVAL_MASK_SEED = 20261117  # fixed for every window, run and seed: the deterministic scoring masks


@dataclasses.dataclass(frozen=True)
class WindowProtocol:
    """Data source and early-stopping protocol of one window size (copied from the sibling ConvAE package).

    Attributes:
        seq_len: window length T.
        sibling: the one-class ConvAE package whose splits and caches are reused.
        data_dir: encoder input folder (read only).
        cache_dir: the sibling's decoded-shard cache (read only while valid).
        attack_splits: on-disk attack test splits (T = 64: the attack windows sit in `test` next to the benign ones).
        patience, plateau_patience, min_delta: the sibling's early stopping / ReduceLROnPlateau settings.
    """

    seq_len: int
    sibling: str
    data_dir: str
    cache_dir: str
    attack_splits: tuple[str, ...]
    patience: int
    plateau_patience: int
    min_delta: float


PROTOCOLS = {
    64: WindowProtocol(
        64,
        "task/encoder (mode ae)",
        "src/data/encoder_input/benign_gridsybil/T64",
        "task/encoder/runs/cache",
        (),
        patience=5,
        plateau_patience=3,
        min_delta=0.0,
    ),
    48: WindowProtocol(
        48,
        "task/autoencoder_benign_t48",
        "src/data/encoder_input/benign_gridsybil/T48_benign",
        "task/autoencoder_benign_t48/cache",
        ("test_gridsybil",),
        patience=10,
        plateau_patience=5,
        min_delta=1e-4,
    ),
    32: WindowProtocol(
        32,
        "task/autoencoder_benign_t32",
        "src/data/encoder_input/all/T32_benign",
        "task/autoencoder_benign_t32/cache",
        ("test_gridsybil", "test_datareplay", "test_dosrandom", "test_dosdisruptive"),
        patience=10,
        plateau_patience=5,
        min_delta=1e-4,
    ),
}


@dataclasses.dataclass(frozen=True)
class MAEConfig:
    """Every setting of a run.

    Attributes:
        window: T (64, 48 or 32); selects the `WindowProtocol`.
        encoder: "mlp" (TabMlp + TabMlpDecoder) or "resnet" (TabResnet + TabResnetDecoder).
        seed: weights, batch order and training masks.
        masked_prob: share of real cells hidden per training step (widedeep `masked_prob`, default 0.2).
        mlp_hidden_dims: TabMlp hidden widths; the last one is z. The decoder mirrors them.
        mlp_dropout: dropout after every TabMlp dense layer (widedeep default 0.1).
        blocks_dims: TabResnet block widths; the last one is z. The decoder mirrors them.
        blocks_dropout: dropout inside every residual block (widedeep default 0.1).
        feed_masked_input: True = the encoder sees the obfuscated input (the masked-autoencoder objective, as in
            widedeep's TabNet path); False = widedeep 1.7.0's literal TabMlp / TabResnet path, where the masked input
            is computed but the encoder receives the clean input (see `model.py`).
        linear_mlp_output: True = the TabMlpDecoder's last layer is a plain Linear; False = widedeep's literal MLP,
            which puts ReLU + dropout after the reconstruction layer too (cannot output negative values).
        n_eval_draws: number of deterministic scoring masks; they partition the T x 13 cells into equal groups,
            so every real cell is hidden exactly once (= round(1 / masked_prob)).
        eval_mask_seed: seed of that partition (fixed constant, not the run seed).
        lr, weight_decay: AdamW (widedeep's default optimizer is `torch.optim.AdamW(params)`: lr 1e-3, wd 0.01).
        batch_size: training batch size (widedeep default 32; 256 here, like the sibling ConvAEs).
        max_epochs, patience, plateau_factor, plateau_patience, min_delta: early stopping / ReduceLROnPlateau on the
            benign val masked-reconstruction MSE (sibling values per window, filled by `for_window`).
        grad_clip: max global gradient norm (0 = off; widedeep does not clip). The pre-clip norm is logged.
        threshold_quantiles: benign val quantiles used as alarm thresholds.
        mahalanobis_max_cond, mahalanobis_shrinkage: `task.autoencoder_benign_t32.scoring.Mahalanobis` settings.
        device: "auto" (mps > cuda > cpu) or a torch device.
        eval_batch: batch size for validation and scoring.
        workers: JSON decoding processes (only used when a sibling cache is missing).
        runs_dir: parent of the run folders.
        max_windows: cap per split for quick checks (0 = all); a capped run gets a `-check` suffix.
    """

    window: int = 32
    encoder: str = "mlp"
    seed: int = 0
    masked_prob: float = 0.2
    mlp_hidden_dims: tuple[int, ...] = (64, 16)
    mlp_dropout: float = 0.1
    blocks_dims: tuple[int, ...] = (64, 32, 16)
    blocks_dropout: float = 0.1
    feed_masked_input: bool = True
    linear_mlp_output: bool = True
    n_eval_draws: int = 5
    eval_mask_seed: int = EVAL_MASK_SEED
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 256
    max_epochs: int = 300
    patience: int = 10
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    min_delta: float = 1e-4
    grad_clip: float = 0.0
    threshold_quantiles: tuple[float, ...] = (0.95, 0.99)
    mahalanobis_max_cond: float = 1e6
    mahalanobis_shrinkage: float = 0.01
    device: str = "auto"
    eval_batch: int = 4096
    workers: int = 8
    runs_dir: str = RUNS_DIR
    max_windows: int = 0

    def __post_init__(self) -> None:
        if self.window not in PROTOCOLS:
            raise ValueError(f"window must be one of {sorted(PROTOCOLS)}, got {self.window}")
        if self.encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {ENCODERS}, got {self.encoder!r}")
        if not 0.0 < self.masked_prob < 1.0:
            raise ValueError(f"masked_prob must be in (0, 1), got {self.masked_prob}")

    @classmethod
    def for_window(cls, window: int, **changes) -> "MAEConfig":
        """Config with the sibling's early-stopping / LR-schedule protocol of `window`, then `changes` on top."""
        p = PROTOCOLS[window]
        base = {"patience": p.patience, "plateau_patience": p.plateau_patience, "min_delta": p.min_delta}
        return cls(window=window, **{**base, **changes})

    @property
    def protocol(self) -> WindowProtocol:
        return PROTOCOLS[self.window]

    @property
    def seq_len(self) -> int:
        return self.window

    @property
    def n_features(self) -> int:
        return N_FEATURES

    @property
    def run_id(self) -> str:
        """Run folder name, e.g. `T32-mlp-seed0` (`-check` when windows are capped)."""
        return f"T{self.window}-{self.encoder}-seed{self.seed}" + ("-check" if self.max_windows else "")

    def to_dict(self) -> dict:
        """JSON-ready settings (plus the resolved data protocol, for the record)."""
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        d["protocol"] = dataclasses.asdict(self.protocol)
        d["protocol"]["attack_splits"] = list(self.protocol.attack_splits)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MAEConfig":
        """Inverse of `to_dict`."""
        d = {k: v for k, v in d.items() if k != "protocol"}
        for k in ("mlp_hidden_dims", "blocks_dims", "threshold_quantiles"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)
