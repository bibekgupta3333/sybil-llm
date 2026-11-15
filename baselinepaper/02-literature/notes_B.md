# Notes B — reconstruction-based / one-class anomaly detection in vehicular networks

Slice B of the literature scan, 2026-11-10. 21 records in `papers_B.json` (B01–B21). Every record was checked in this
session by a page fetch, a full-text PDF read, or the Crossref / Semantic Scholar APIs; the `verified_via` field says
which. Fields I could not confirm say "not stated in abstract / not verified". Full text was read for B01, B03, B04,
B05, B06, B07, B12, B14, B15, B17, B19, B20; the rest are abstract-level (B13 is bibliographic only).

## 1. What unsupervised / one-class methods claim

- **V2X papers on VeReMi / VeReMi Extension report 0.93–0.99.** Accuracy 0.98 (B01, B02), F1 0.96 / AUC 0.97 (B06),
  F1 0.99 (B07), accuracy 0.93 (B08), AUC 0.9987 (B09), AUC up to 0.98 (B03). One-class models are often reported at
  or above supervised baselines (B06: one-class F1 0.963 vs supervised 0.928–0.943).
- **The four Sybil attacks are reported as easy.** B01: recall 0.98–1.0 for types 16–19; B03: F1 0.94–0.95 for
  A16–A19; B06: Sybil F1 0.930, replay 0.901. No paper found reports DataReplaySybil or the DoS Sybil types as hard,
  and none controls for the sequence-length / message-rate shortcut. B03 needs sequences of ≥ 15 messages, which
  removes the 1–3-message pseudonym tracks that make up most DataReplay / DoS data.
- **Numbers are lower where the protocol is stricter or the data is not VeReMi:** B05 full pipeline accuracy 0.824,
  plain VAE 0.587; B12 F1 0.912 vs 0.9999 supervised; B04 average AUROC 0.89–0.93 with chance-level rows; B15
  accuracy 0.51–0.75.

## 2. How "normal" is obtained

- Almost every one-class paper trains on **benign-only data selected with the ground-truth labels** (B01, B03, B04,
  B05, B06, B12, B14, B17–B20). This is a label-dependent assumption that the papers call "unsupervised".
- **Training on unlabelled mixed (contaminated) data is rare**: only B15 (DAGMM-style) and the clustering work B16.
  B15 is the one that lands near chance. No VeReMi paper found trains an autoencoder on mixed benign + attack data,
  so our AUROC ≈ 0.48 result has no direct counterpart.

## 3. How thresholds are chosen

- **On labelled test data:** B01 picks the threshold with maximum test accuracy (Algorithm 1); B15 sweeps energy
  percentiles and reports the best; B20 fixes the threshold statistic after comparing candidates on the attack tests.
- **With labels on validation:** B07 fits an SVM on validation reconstruction errors; B04 selects which models enter
  the ensemble by AUROC on a validation set that contains attacks; B03 uses "optimal F1" in its cross-scenario part.
- **From benign-only held-out data (the defensible protocol):** B17 (99.99th percentile per signal), B19 (separate
  recording), B06 (95th percentile), B03 main setting (98th percentile of benign validation senders), B12
  (median + 5 MAD), B04 (percentile of benign training scores).
- Not stated in several abstracts (B02, B08–B11). Threshold-free AUROC is given in a minority (B03, B04, B06, B14).

## 4. Split protocol

Only B03 states a split **by sender identity** (70 / 15 / 15). B19 splits by recording. The other VeReMi papers
give no vehicle-level separation, and none mentions that VeReMi Extension scenarios reuse the same traffic (our F1
finding) or the receiver-copy duplication.

## 5. Does anyone compare several AE variants under one protocol?

Not for Sybil attacks. The nearest: B04 (WGAN ensemble vs CNN autoencoder on raw and on engineered features vs PCA /
KNN / GMM, per-attack AUROC, 35 attacks, no Sybil — the plain autoencoder on engineered features has the best average,
0.93 vs 0.89 for the proposed GAN, and raw features give 0.66); B05 (AE vs VAE, federated vs not); B11 (AE vs
LSTM-VAE vs CNN-VAE); B01 (CNN-LSTM vs LSTM); B19 (CANdito vs CANnolo). None compares contaminated vs benign-only
training, reconstruction vs latent Mahalanobis scoring, and a supervised head on the same encoder and split.

## 6. Negative or weak results that are reported

- **Attacks that respect physics are missed:** constant position offset is at chance for every model in B04
  (0.48–0.53) and F1 0.19 in B03; eventual stop recall 0.52 (B01), F1 0.73 (B03); delayed messages 0.64 (B01).
- **Replay of valid data defeats reconstruction in CAN work:** B19 full replay detection rate 0.57 (CANnolo 0.55),
  with the explanation that replayed sequences are valid; B17's plain autoencoder baseline has replay TPR 0.03.
  This is the published analogue of our DataReplay result; V2X papers report the opposite.
- **Missing / rate-type attacks:** suppress TPR 0.08–0.25 on real CAN data (B17), drop 0.23 (B19), DoS F1 0.75 in
  B03, attributed to the lack of message-frequency features.
- **Small falsifications are undetectable** below 30 m in B14 (AE and one-class SVM about random under 10 m).
- **Mahalanobis scoring** was tried and dropped as worse than the l2 reconstruction error in B19.
- **Contaminated training near chance:** B15, 51% accuracy at the 99th percentile.

## 7. Gap for our paper

No study found evaluates reconstruction-based detectors on the VeReMi-Extension Sybil attacks with (i) a split by
vehicle, (ii) thresholds from benign validation only, (iii) a length-only baseline, and (iv) contaminated vs
benign-only vs supervised training of one architecture. The reported near-perfect Sybil numbers (B01, B06, B07) come
from protocols with at least one of: a test-tuned threshold, an unstated split, labels in the decision stage.

Not included as records (not peer-reviewed papers, abstract only): two University of Windsor MSc theses — a
dual-branch denoising autoencoder on VeReMi Extension calibrated on held-out benign data (2026) and DBSCAN-based
unsupervised detection on VeReMi (2024).
