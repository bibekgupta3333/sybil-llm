# Protocol review — methods (anomaly-detection methodology + vehicular-security reviewer)

Reviewed 2026-11-10 against the gap analysis, paper plan, inventory, notes A–C, `agent.md`, the `all/T32_benign`
metadata and the two T32 packages. Nothing was run.

## Verdict: major changes to the text, small changes to the work

Three blockers, all fixable before the first run (about 2 days, about 6 h extra compute). The data, split, seeds,
score files and bootstrap unit are sound.

## Issues

1. **BLOCKER — the "unlabelled" regimes use benign labels three times.** Normalisation is fitted on benign train,
   early stopping uses benign val loss, the threshold is a benign-val percentile. Benign-only statistics also change
   the scale of attack rows (`range` std 108 vs 405–419 when both classes are fitted), with unknown direction of
   effect. *Change:* for `mix_*`, primary run = normalisation re-fitted on the regime's own training pool (an affine
   map at load time), early stopping on a val pool of the same composition, operating point = 95th percentile of
   that mixed val; keep the benign-fitted version as a labelled "oracle calibration" row. For `benign_only` and
   `supervised` the benign-val threshold is consistent with their assumption: keep. *Cost:* 3 h code, about 5 h
   compute (mix runs twice).
2. **BLOCKER — length is not removed and the strata do not control it.** The autoencoder takes the mask as its 14th
   input channel; for a 1-row window std = 0, last − first = 0, min = max = mean, and `log_dtau` is exactly 0 (missing
   value → train mean). DoS windows average 1.14 rows, DataReplay 1.5; benign windows in the 1–3 stratum are spread
   differently, so length still separates classes inside it. The ≥ 4 stratum has 523 DoS windows. *Change:* delete
   "the number of rows is not a feature"; say every detector can see length. Report length-only AUROC inside each
   stratum, add exact strata n = 1, 2, 3, 4–16, ≥ 17, and a length-matched AUROC (weighted mean over exact n). Fix
   the length score now as −n (shorter = more anomalous; the inventory shows the direction was once chosen on test
   labels). State a minimum of attack windows and vehicles for a stratum to be reported. *Cost:* 3 h, no compute.
3. **BLOCKER — hyperparameters are not written down, so nothing is pre-registered.** k, ν, γ, contamination, PCA
   components, subsample size, LSTM layers, epochs and patience are absent. The convolutional autoencoder was
   developed with its seed-0 test result on this very test set already seen (0.736 / 0.391 / 0.973 / 0.611).
   *Change:* add a table with every value (library defaults, named), one fixed subsample size, and a disclosure of
   the prior test exposure. *Cost:* 2 h.
4. **SHOULD FIX — headline for DataReplay / DoS.** 95–100% of their windows are 1–3 rows, mostly one message. Short
   pseudonym life is the attack's real, receiver-observable signature, not only a shortcut; pooled window AUROC then
   measures length. Reconstruction error "below chance" (0.391) may be a length effect (in-distribution predicts
   0.5). *Change:* for these two attack families the headline is the length-matched AUROC and the paired difference
   to length-only; pooled AUROC is secondary. *Cost:* in item 2.
5. **SHOULD FIX — the two mix regimes differ by benign weight only.** DoSRandom and DoSDisruptive benign train counts
   are identical (102,127) and DataReplay's nearly so: `mix_all` is `mix_gridfolders` with the same benign traffic
   weighted about 4×. *Change:* state the shares from counts (about 31% and 64%, to be computed before running), say
   they are two confounded points and not a dose-response, and add `benign_gridfolders` (benign of the two GridSybil
   folders only) as the matched control. *Cost:* 1 h, about 1.5 h compute.
6. **SHOULD FIX — bootstrap.** Resample benign and attack keys separately; report paired intervals of differences
   (detector − length-only, regime − regime) from the same resamples; name at most five primary comparisons now.
   The interval covers test sampling only (one split seed, fixed threshold): say so. *Cost:* 2 h.
7. **SHOULD FIX — F14 and folder composition, from the score files.** Near-copies in train shrink distances for
   benign test windows; kNN is hit hardest. *Change:* add sensitivity rows: 1416 only; GridSybil_0709 benign
   excluded; benign test restricted to the attack's own family folders. Measuring F14 on this split stays E7.
   *Cost:* 2 h, no compute.
8. **SHOULD FIX — LSTM autoencoder wording.** The published models work on fixed 15–20-message sequences grouped by
   true sender id with their own features. *Change:* call it "an LSTM autoencoder of matched size on our input",
   not a replication.
9. **SHOULD FIX — "unseen attack" scope.** One training attack, one direction. A model trained with GridSybil
   (20% short windows vs 14% benign) can learn "short = attack" and appear to transfer. *Change:* word it as
   "transfer from GridSybil", read only within exact-length strata; no general unseen-attack claim (that is E8).
10. **NOTE** — add a low false-positive point (benign-val p99), free; say "sender-held-out" (receiver position /
    velocity, 4 of 13 features, come from receivers that also hear train senders); add a vehicle-macro average
    (173 attack vehicles give 200k DoS windows); say PCA and kNN are fitted on the window statistics.

## Claim in gap-analysis §3 that this protocol would not support

**Claim 2**, "much weaker than reported *once thresholds come from validation and vehicles are held out*". The
protocol never varies the threshold source, the split (E7) or the sequence unit (E9), so the gap to the published
0.93–0.99 cannot be attributed; pseudonym links versus sender-id sequences is an equally likely cause. Supported
wording: "under this protocol reconstruction reaches X". Also short: claim 1 lists rules,
but the plausibility detector is per message (E10 out of scope), and claim 3 rests on two confounded points until E6.

## Fine as it is

- One input, one split by vehicle key across the four families, no new split.
- AUROC plus average precision with the share stated, recall and false-positive rate at a validation threshold.
- Cluster bootstrap over vehicle keys (it absorbs the F1 duplication, which does not bias AUROC itself).
- Test labels read only by the evaluation module; tables rebuilt from score files; deviations logged in §9.
- Seeds 0–4 (a rough standard deviation only); supervised model selection on val AUROC; the re-normalisation check.
