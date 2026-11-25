"""Reader for the all/T20_benign encoder input: the `task/autoencoder_benign_t32` reader, imported unchanged.

The T = 20 input has the T = 32 layout (`<split>/b20/part-XXXXX.json` + `_info.json`, same info fields, same split);
the reader globs `b*/part-?????.json` and takes the window length from the shards, so nothing in it depends on T.
"""

from __future__ import annotations

from task.autoencoder_benign_t32.data import INFO_FIELDS, SplitCache, WindowDataset, read_shard, track_id, vehicle_key

__all__ = ["INFO_FIELDS", "SplitCache", "WindowDataset", "read_shard", "track_id", "vehicle_key"]
