# Plausibility-check baseline (tracker 4.5)

Offline re-implementation of the VeReMi-Extension paper's "simple detection algorithm" (Kamel et al., ICC 2020,
§V / Table III; `varemi.txt`): F2MD plausibility / consistency checks + ThresholdApp (flag a received BSM when any
check ≤ 0.28125), per received message at every receiver. Spec, sources and every parameter:
`docs/research-notes/veremi-plausibility-baseline-spec.md`. Not a learned model; reads `data/VeReMi-Dataset/` only.
Run: `npm run baseline:plausibility` (all 8 subsets, 3 variants, 431 s with 10 workers) · tests:
`npm run baseline:plausibility:test`. Output: `src/runs/baselines/plausibility/<subset>/results.json` (headline
`catch`), `variants/<name>/`, `summary.json` (paper vs ours). Label: misbehaving iff the sender's own trace file has
attack code ≠ 0 (the ground-truth file has no attacker field). Map check C10 and proximity C0 not computed.

| Subset | Paper Acc / P / R / **F1** | Ours `catch` (headline) | ΔF1 | Ours `legacy` | ΔF1 |
|---|---|---|---|---|---|
| GridSybil_0709 | 0.820 / 0.997 / 0.590 / **0.742** | 0.734 / 0.997 / 0.572 / **0.727** | −0.015 | 0.747 / 0.996 / 0.593 / 0.743 | +0.002 |
| GridSybil_1416 | 0.800 / 0.997 / 0.584 / **0.737** | 0.694 / 0.997 / 0.530 / **0.692** | −0.045 | 0.712 / 0.995 / 0.558 / 0.715 | −0.021 |
| DataReplaySybil_0709 | 0.801 / 0.920 / 0.353 / **0.510** | 0.799 / 0.915 / 0.353 / **0.510** | −0.000 | 0.840 / 0.882 / 0.530 / 0.663 | +0.153 |
| DataReplaySybil_1416 | 0.795 / 0.892 / 0.371 / **0.524** | 0.796 / 0.910 / 0.381 / **0.537** | +0.013 | 0.845 / 0.871 / 0.589 / 0.702 | +0.179 |
| DoSRandomSybil_0709 | 0.990 / 0.999 / 0.978 / **0.989** | 0.908 / 0.997 / 0.799 / **0.887** | **−0.101** | 0.908 / 0.996 / 0.802 / 0.888 | −0.100 |
| DoSRandomSybil_1416 | 0.989 / 0.999 / 0.977 / **0.988** | 0.902 / 0.997 / 0.796 / **0.885** | **−0.103** | 0.903 / 0.995 / 0.798 / 0.886 | −0.102 |
| DoSDisruptiveSybil_0709 | 0.770 / 0.982 / 0.501 / **0.664** | 0.766 / 0.980 / 0.497 / **0.660** | −0.004 | 0.792 / 0.963 / 0.566 / 0.713 | +0.049 |
| DoSDisruptiveSybil_1416 | 0.779 / 0.988 / 0.532 / **0.691** | 0.778 / 0.986 / 0.539 / **0.697** | +0.006 | 0.803 / 0.973 / 0.600 / 0.742 | +0.051 |

`catch` reproduces 6 / 8 subsets within 0.05 F1 (4 within 0.015). Gaps > 0.05 F1, in plain words:

- **DoSRandomSybil (−0.10, both groups, both variants):** the Sybil pseudonym changes almost every message, so the
  history checks run on only 1.6% of attack messages; range plausibility alone catches 80% (random positions more
  than 420 m away). The other 20% are random positions near the receiver; F2MD most likely caught them with the
  map check (C10, position off the road), which needs the LuST road network we do not have.
- **`legacy` on DataReplaySybil (+0.15 / +0.18) and DoSDisruptiveSybil (+0.05):** the binary overlap check fires
  on 37% of replayed messages vs 19% with CaTCh's confidence ellipses, so `legacy` beats the paper. `catch`
  matches the paper here, which supports the spec's guess that Table III used the CaTCh / Experi checks.
- **Accuracy on GridSybil (−0.09 / −0.11) while F1 is close:** our local GridSybil data has 62–65% attack
  messages; the paper's numbers imply 44–48%. GridSybil_0709 is a 2025 re-simulation (agent.md F14), so the class
  balance differs and accuracy is not comparable. DataReplay / DoS balances match the paper within 0.01.

Verified (2026-10-29): metrics recomputed from the counts (Δ 0); received / attack / unknown counts equal an
independent pass over the raw files; C1, C2 (CaTCh), C3, C8 and the label hand-checked on 30 messages (3 subsets,
half of them firing): 0 mismatches. Ground truth by messageID agrees with the file-code label: attacker
messages identical to their ground-truth row are 0–0.15% for DataReplay / DoS (GridSybil: 100% in 0709, 22% in
1416: its ground truth stores the sent ghost values, so it cannot label them). One deterministic run (RULE 1).
