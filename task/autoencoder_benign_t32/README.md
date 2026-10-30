# `task/autoencoder_benign_t32/`: one-class autoencoder on the T = 32 benign windows of all 8 folders

The professor's one-class baseline: a ConvAutoencoder (input 14 × 32 = 13 features + mask, z = 16, masked MSE on real
rows, 63,293 parameters) trained on **benign windows only** from `src/data/encoder_input/all/T32_benign/`, using the
80 / 10 / 10 split by sender vehicle fixed on disk. Train fits the weights; benign val drives early stopping, the LR
schedule and the alarm thresholds (val p95 / p99); `train.py` never opens `test` or `test_<attack>`. Scores:
reconstruction error and Mahalanobis distance of z (fitted on benign train z), per window and per track (max / mean).
Not the thesis TimesNet model.

**Run (repo root):** `npm run aeb32:train` (seed 0 → `runs/seed0/`, about 20 min on MPS; decoded shards cached in
`cache/`) · `npm run aeb32:test` (pytest, synthetic shards) · `npm run aeb32:notebook` (executes
`autoencoder_benign_T32.ipynb` on the newest run, about 15 s with the cache). Run folder (gitignored): config, env,
history, train.log, curves.png, best / last.pt, mahalanobis.npz, summary.json.

**Results (`seed0`, one seed):** best benign val masked MSE 0.1050 at epoch 151, early stopping at epoch 161 after 7 LR
halvings, 19.4 min on MPS. Benign test FPR at val p95 0.048. Test window AUROC, each attack split vs the benign test
windows (labels read only after scoring):

| | GridSybil | DataReplay | DoSRandom | DoSDisruptive |
|---|---|---|---|---|
| reconstruction error | **0.736** | 0.391 | **0.973** | 0.611 |
| Mahalanobis on z | 0.510 | 0.336 | 0.837 | 0.556 |
| window length alone (reference) | 0.559 | 0.952 | 0.967 | 0.967 |
| recall at val p95 (reconstruction) | 0.298 | 0.046 | 0.880 | 0.292 |

**Reading:** only on GridSybil does the reconstruction error add signal beyond window length (0.778 on windows with
≥ 4 rows; errors in log-Δτ and range). DataReplay replays real trajectories and is scored below chance. DataReplay / DoS
windows are 95-100% 1-3 messages (F11), so length alone matches or beats the model there; DoSRandom is caught by its
out-of-range claimed values. Per-track aggregation does not help. Caveats: one seed, F14 (GridSybil_0709 near-copies can
straddle splits), benign traffic repeated per family (F1). Details: notebook section 7.
