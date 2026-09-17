# Data Understanding (Phase 1) — Research Notes

All research-note artifacts for `docs/plan/research-plan.md`'s **Phase 1 — Data Understanding**, grouped here
so the VeReMi-Extension groundwork is one browsable unit instead of scattered among the proposal-stage notes
in the parent `docs/research-notes/` folder.

| Task | File | Covers |
|------|------|--------|
| 1.1 | [`veremi-dataset-structure.md`](veremi-dataset-structure.md) | Folder layout, filename schema, worked example tracing one raw message to `X_windows.npy[0]` |
| 1.2 | [`attack-taxonomy.md`](attack-taxonomy.md) | The 4 Sybil attack families vs benign — mechanisms, confirmed attacker codes (A16–A19), trajectory examples |
| 1.3 | [`field-reference.md`](field-reference.md) | Message field semantics: units, coordinate frame, noise-variant handling, main-vs-noise rationale |
| 1.4 | [`class-balance.md`](class-balance.md) | Class × scenario-group window/sender counts and imbalance ratios |
| 1.5 | [`split-protocol.md`](split-protocol.md) | Split design and the leakage audit (companion script: `scripts/audit_splits.py`) |
| 1.7 | [`data-quality-checks.md`](data-quality-checks.md) | Duplicates, monotonicity, gaps, coordinate-frame sanity checks |
| — | [`gridsybil-windowing-defect.md`](gridsybil-windowing-defect.md) | ⚠ Open defect found during Phase 1 — blocks Phase 2 task 2.6 |

Task 1.6 (EDA) has no note here — its artifacts are `notebooks/eda_veremi.ipynb` and `results/figures/eda/`.

See `docs/plan/research-plan.md` (Phase 1 section) for status, acceptance criteria, and known risks, and
`tracker.html` for the live progress view.
