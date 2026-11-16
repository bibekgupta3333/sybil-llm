"""Settings of the TimesNet pretraining run on all/T32_benign (benign windows of all 8 folders, T = 32).

`T32BenignConfig` extends `PretrainConfig` (`base_config.py`, this package's own copy of the original settings class):
same encoder (d = 128, d_ff = 64, D12), same objectives and lambdas (D7), robust scaling of claimed position and range
(D13) and length-bucketed batches (D14). What changes: the data (all/T32_benign, T = 32, read from a numpy cache),
the check set (the on-disk benign `val` split; nothing is carved out of train, X5) and an opt-in filter of very short
windows from the SSL train set (`min_train_rows`, X3, default 1 = keep every window).

Note on the data choice: the train split holds benign windows only, selected with attack labels when the encoder input
was built (one-class setting, user's choice 2026-11-15). The pretraining code itself never opens a label file.
"""

from __future__ import annotations

import dataclasses

from src.model.all_t32_benign.timesnet.base_config import PretrainConfig

DATASET_NAME = "t32b"
DATA_SELECTION_NOTE = (
    "train / val hold benign windows only, selected with attack labels when the encoder input was built "
    "(one-class setting, user's choice 2026-11-15); this code never opens a label file"
)


@dataclasses.dataclass
class T32BenignConfig(PretrainConfig):
    """Every knob of a T32_benign pretraining run; saved to config.json.

    Inherits every field of `PretrainConfig` and changes only the defaults listed here.

    Attributes:
        cache_dir: numpy feature cache built by `src.pipeline.all_t32_benign.cache` (JSON fallback if missing).
        check_split: on-disk split used for label-free checkpoint selection (the benign `val` split).
        min_train_rows: windows with fewer real rows are dropped from the SSL train set only (X3); 1 keeps all.
        val_shards_per_split: source shards of the check split to load; None follows `shards_per_split`.
        short: a quick complete run on a few shards (`--short`, plumbing for the notebook, never a result).
    """

    dataset: str = DATASET_NAME
    data_dir: str = "src/data/encoder_input/all/T32_benign"
    runs_dir: str = "src/runs/pretraining/all/T32_benign"
    holdout_frac: float = 0.0  # nothing is carved out of train: the check set is the on-disk val split
    seq_len: int = 32
    max_hours: float = 24.0  # a time stop truncates the cosine schedule; keep it above the expected run time
    cache_dir: str = "src/data/cache/all/T32_benign"
    check_split: str = "val"
    min_train_rows: int = 1
    val_shards_per_split: int | None = None
    short: bool = False

    @classmethod
    def preset(cls, name: str) -> "T32BenignConfig":
        """Named presets: `joint` (the plan, D7) and `recon_only` (sanity baseline)."""
        if name == "joint":
            return cls()
        if name == "recon_only":
            return cls(use_nce=False, use_physics=False)
        raise ValueError(f"unknown preset {name!r}")

    def validate(self) -> None:
        """Fails early on settings that cannot work (the parent's checks plus the T = 32 ones).

        Raises:
            ValueError: on any invalid value.
        """
        super().validate()
        if self.dataset != DATASET_NAME:
            raise ValueError(f"dataset must be {DATASET_NAME!r}, got {self.dataset!r}")
        if self.seq_len != 32:
            raise ValueError(f"all/T32_benign windows have 32 rows, got seq_len {self.seq_len}")
        if self.check_split != "val":
            raise ValueError(f"the check set is the on-disk val split, got {self.check_split!r}")
        if not 1 <= self.min_train_rows <= self.seq_len:
            raise ValueError(f"min_train_rows must be in 1..{self.seq_len}, got {self.min_train_rows}")
        if self.val_shards_per_split is not None and self.val_shards_per_split < 1:
            raise ValueError(f"val_shards_per_split must be >= 1 or None, got {self.val_shards_per_split}")
        if self.min_rows_aux > self.seq_len:
            raise ValueError(f"min_rows_aux {self.min_rows_aux} exceeds seq_len {self.seq_len}")
        if self.min_freq > self.seq_len // 2:
            raise ValueError(f"min_freq {self.min_freq} leaves no FFT bin at seq_len {self.seq_len}")
        if not 0.0 < self.block_ratio[0] <= self.block_ratio[1] < 1.0:
            raise ValueError(f"block_ratio must satisfy 0 < low <= high < 1, got {self.block_ratio}")
