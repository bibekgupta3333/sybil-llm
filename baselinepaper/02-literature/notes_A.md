# Notes A — ML / DL misbehaviour and Sybil detection on VeReMi and VeReMi-Extension

Scan date 2026-11-10. 30 records in `papers_A.json` (A01–A30): 6 dataset / framework / survey / rule-based
records and 24 learning papers (A04–A27). Every record's bibliographic data was checked this session (OpenAlex,
Crossref, arXiv / ar5iv, publisher or repository pages). Content fields come only from what was read; where only
an abstract or a record was reachable the field says "not stated in abstract / not verified". Full text was read
for A01, A11 (partial), A14, A17, A18, A19, A21, A22, A27, A28. Counts below are over this sample, not the field.

## 1. How the data are split

- **No paper in the sample holds out physical vehicles**, and none holds out a scenario or checks that the attack
  scenarios replay the same traffic. Neither dataset paper (A01, A02) defines a split.
- **16 of 24** learning papers: split not visible in what could be read (abstract-level for most, so this is an
  upper bound on "not stated").
- **8 of 24** with a readable protocol:
  - window / sequence-level random split: A23 VeMisNet ("single, stratified 80/20 … at the sequence level"),
    A22 Franceschini (77 / 23 over balanced 5-message windows, scaler fitted before the split);
  - per-dataset 70 / 30: A27 ADVENT; "split the entire dataset", unit and ratio not given: A14 Uprety;
  - within-vehicle split (each vehicle trains and tests on its own log): A17 Mármol Campos (80 / 20 of each
    client's benign data), A18 Demir;
  - full text read, split still not specified: A19 MistralBSM;
  - temporal + leave-attack-out: A21 Sedar (the only unseen-attack protocol).
- "Leakage" is mentioned once (A23), and only about applying SMOTE after the split.
- The successor dataset VeReMi NextGen (A29) ships train / val / test sets and gives the missing splits of
  earlier datasets as a motivation; the original VeReMi authors "caution against using this data … as the sole
  foundation" for evaluating learned models (A01).

## 2. Typical reported numbers

- Supervised, pooled accuracy / F1 of **95–99.9%**: 98% (A09 DeepADV, reconstruction), F1 95.6–96.8% over 18
  classes (A10), 98% (A19), 0.9882 (A20), 95.22% (A22), 99.81% (A23), 98% (A24), 98.60% (A25), 99.91% F1 (A26).
- Lower numbers appear where the protocol is stricter or more local: 81–92.5% (A11), ≈ 0.80 (A14), 0.82
  accuracy / 0.78 F1 unsupervised (A17), 0.68 → 0.80 per-vehicle (A18), **F-score 0.48–0.49 on unseen DoS Sybil
  variants, 0.71 with transfer** (A21).
- The dataset's own plausibility benchmark (A02) is far below the ML claims for the Sybil attacks (repo notes).

## 3. Learning setting

- Unsupervised / benign-only: **3 of 24** (A08, A09 reconstruction + threshold on 20-message sequences; A17
  GMM + VAE). Reinforcement learning with label-derived rewards: 2 (A20, A21). The rest are supervised.
- The survey (A28) states most ML detectors are supervised and that the unseen-attack ability of the
  unsupervised ones is unverified because they are tested only on known attacks.

## 4. Sybil coverage

- Sybil-only: A15 (back-end reports, LSTM, 4 Sybil types), A25 (CatBoost), A26 (GCN + GRU, own extended data).
- Sybil as some classes among many: A19, A21, A23, A27 — mostly the two DoS Sybil variants.
- No verified per-attack result for GridSybil or DataReplaySybil from a learned receiver-side model.
- Position-falsification-only work on the original VeReMi is still the largest group (A04, A05, A11, A12, A14).

## 5. What nobody checks (in this sample)

- Vehicle-disjoint or scenario-disjoint evaluation; the repeated traffic across attack folders.
- A trivial-feature control (window length, message rate, inter-message time). DoS-family results (A23, A27)
  rely on features where such a control is decisive.
- The unit under pseudonym change: windows are grouped by the true sender id (A19) or pseudo-id is an input
  field (A06, A21); no paper discusses 1–2-message pseudonym lifetimes.
- Dependence between overlapping windows (A24); thresholds set on labelled / test data (A08, per the survey);
  seeds, confidence intervals, benign false-positive rate at a fixed operating point. Code is public for the
  datasets, F2MD and A17 only.
- One paper (A22) says explicitly that detectors must be tested beyond "partitions of the training datasets".

## Caveats

- A13 (Hsu et al.), A16 (Mahmoudi et al.), A30 (Kamel et al. VNC 2019): record verified, content not read.
- A02 Table III values are taken from this repo's reproduction notes, not re-fetched (HAL blocked the fetch).
- Details of A06–A09 and A15 come from the A28 survey's description, not the papers themselves.
- Checked and left out: VehiGAN (VASP data, not VeReMi), Gyawali & Qian (VeReMi use not confirmed), the
  original ADVENT preprint (other datasets).
