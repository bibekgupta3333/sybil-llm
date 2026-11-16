"""Self-supervised TimesNet pretraining on all/T32_benign (T = 32), a self-contained package.

Modules: `encoder` (TimesNet: embedding, TimesBlock, Inception, FFT periods, masked pooling), `heads`
(reconstruction, P1-P3, projection, `PretrainingModel`), `views` (masking, augmentations, injected violations),
`losses` (NT-Xent, `LossScales`, `Objective`), `monitors` (label-free `Evaluator`), `base_config` (`PretrainConfig`),
`base_data` (feature order, shard readers, `FeatureShards`, `FeatureSpace`, `LabelFirewall`), `base_train` (base
`Trainer`, `preflight`, `BatchPrefetcher`, resources), then the T32 run itself: `config` (T32BenignConfig), `data`
(firewalls, cached split loader, feature-space save / restore), `train` (BatchPlan, T32Trainer, command line),
`history`, `evaluate`, `scoring`, `plots`.

The model code is this package's own copy of `src/model/benign_gridsybil/timesnet/` (copied 2026-11-15, same math,
parameter names and random-number order); nothing is imported from there (`tests/test_isolation.py` checks it).

    .venv-train/bin/python -m src.pipeline.all_t32_benign.cache              # build the numpy feature cache once
    .venv-train/bin/python -m src.model.all_t32_benign.timesnet.train --check
    .venv-train/bin/python -m src.model.all_t32_benign.timesnet.train --smoke
    .venv-train/bin/python -m src.model.all_t32_benign.timesnet.train --seed 0
"""
