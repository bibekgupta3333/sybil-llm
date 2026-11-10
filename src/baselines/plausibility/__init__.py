"""Plausibility-check baseline of the VeReMi-Extension paper (Table III): F2MD checks + ThresholdApp decision.

Modules: `config` (parameters, variants, paper numbers), `parser` (raw traces -> receiver streams + labels),
`node_table` (per-receiver node table, Kalman filters), `checks` (LegacyChecks, one class per check),
`catch_checks` (CaTChChecks), `detector` (threshold decision per BSM), `metrics` (counts, scores), `run` (CLI).
Spec: `docs/research-notes/veremi-plausibility-baseline-spec.md`.
"""
