"""Self-supervised pretraining of the TimesNet encoder (benign + GridSybil trial, T = 64).

Modules: `config` (PretrainConfig), `data` (label firewall, feature shards), `encoder` (TimesNet), `heads`
(reconstruction, P1-P3, projection, PretrainingModel), `views` (masking, augmentations, injected violations), `losses`,
`monitors` (label-free evaluation) and `train` (pre-flight checks, Trainer, command line).

Run from the repository root with the training environment:

    .venv-train/bin/python -m src.model.benign_gridsybil.timesnet.train --check    # pre-flight checks, no training
    .venv-train/bin/python -m src.model.benign_gridsybil.timesnet.train --smoke    # ~30 steps on one shard
    .venv-train/bin/python -m src.model.benign_gridsybil.timesnet.train            # the controlled run
    .venv-train/bin/python -m src.model.benign_gridsybil.timesnet.train --preset recon_only
    .venv-train/bin/python -m src.model.benign_gridsybil.timesnet.train --resume src/runs/pretraining/benign_gridsybil/T64/<run_id>
"""
