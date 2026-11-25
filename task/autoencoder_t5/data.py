"""Label-free reader for the all/T5_benign_grid encoder input: the `task/autoencoder_t32` reader, imported unchanged.

It opens the feature shards (`b*/part-?????.json`: id, x, mask) only, never the `_info.json` label files, and takes
the window length from the shards; `LabelFileGuard` is the same process-wide audit hook.
"""

from __future__ import annotations

from task.autoencoder_t32.data import (
    INFO_SUFFIX,
    SHARD_FIELDS,
    SHARD_GLOB,
    FeatureCache,
    LabelFileGuard,
    UnlabelledWindows,
    read_feature_shard,
    track_id,
    vehicle_key,
)

__all__ = [
    "INFO_SUFFIX",
    "SHARD_FIELDS",
    "SHARD_GLOB",
    "FeatureCache",
    "LabelFileGuard",
    "UnlabelledWindows",
    "read_feature_shard",
    "track_id",
    "vehicle_key",
]
