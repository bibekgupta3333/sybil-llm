# `src/pipeline/all_t32_benign/` — feature cache for the T32_benign TimesNet run

`cache.py` converts the feature shards of `src/data/encoder_input/all/T32_benign/<split>/b32/part-XXXXX.json`
(sorted order; `_info` label files are never opened) into `src/data/cache/all/T32_benign/<split>/`:
`x.npy` (N, 32, 13) float32 exactly as stored, `mask.npy` (N, 32) bool, `ids.json`, `manifest.json`
(source shard names + sha256 + window counts, metadata.json sha256, created time). No labels in the cache.

```
.venv-train/bin/python -m src.pipeline.all_t32_benign.cache            # build every split (skips current ones)
.venv-train/bin/python -m src.pipeline.all_t32_benign.cache --verify   # counts, shapes, sha256s, sampled windows
.venv-train/bin/python -m src.pipeline.all_t32_benign.cache --splits train val --workers 4
```

Building every split took ~10 s on the Mac (12 cores, 2026-11-15); expect ~1 min on a 4-vCPU EC2 instance.
The cache is gitignored (`src/data/`) and is rebuilt automatically when a source shard changes (sha256 mismatch).
The pretraining package `src/model/all_t32_benign/timesnet/` reads it through `data.load_split` and falls back to
the JSON shards when it is missing.
