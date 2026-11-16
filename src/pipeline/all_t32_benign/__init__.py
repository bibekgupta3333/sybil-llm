"""Data pipeline for the all/T32_benign TimesNet pretraining on EC2: `cache` turns the JSON feature shards into a
memory-mapped numpy cache (features only, never labels)."""
