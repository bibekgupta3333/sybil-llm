"""Labels and bookkeeping from the `part-XXXXX_info.json` files: the `task/autoencoder_t32` module, imported
unchanged. Evaluation only: nothing in `data.py`, `trainer.py`, `train.py` or `config.py` imports it."""

from __future__ import annotations

from task.autoencoder_t32.labels import INFO_FIELDS, INFO_GLOB, WindowLabels, read_info_shard

__all__ = ["INFO_FIELDS", "INFO_GLOB", "WindowLabels", "read_info_shard"]
