"""Settings of the plausibility-check baseline (F2MD LegacyChecks or CaTChChecks + ThresholdApp decision).

Implements tracker 4.5. Every value follows `docs/research-notes/veremi-plausibility-baseline-spec.md` §5 / §8:
F2MD `veins-f2md` commit 3b306aa (2020-01-10), `LuSTScenario/omnetpp.ini` and `F2MDParameters.h`; receiver vType
limits = LuST passenger1 (UNKNOWN in the data, spec §7). `PlausibilityConfig` is written to every results.json
(RULE 4). `VARIANTS` are the configurations that are run: `catch` is the headline (the spec's closer-to-paper
configuration, chosen before any result was seen), `legacy` the exact binary port, `catch_accel301` the spec's
sensitivity run (most permissive LuST acceleration).
"""

from __future__ import annotations

import dataclasses

# Checks of ThresholdApp's rule (spec §4), in LegacyChecks order. The map check (C10) and the proximity check (C0)
# cannot be computed offline (spec §5, §7) and are absent; sudden appearance (C9) is computed but never decides.
DECISION_CHECKS: tuple[str, ...] = (
    "range_plausibility",  # C1
    "speed_plausibility",  # C2
    "intersection",  # C6
    "position_consistency",  # C3
    "speed_consistency",  # C4
    "beacon_frequency",  # C8
    "position_heading_consistency",  # C7
    "position_speed_consistency",  # C5
    "position_speed_max_consistency",  # C5b
    "kalman_pscp",  # C11 position part
    "kalman_pscs",  # C11 speed part
    "kalman_pscsp",  # C12 position part
    "kalman_pscss",  # C12 speed part
    "kalman_pcc",  # C13
    "kalman_pacs",  # C14
    "kalman_scc",  # C15
)
DIAGNOSTIC_CHECKS: tuple[str, ...] = ("sudden_appearance",)  # C9, not part of ThresholdApp's rule
ALL_CHECKS: tuple[str, ...] = DECISION_CHECKS + DIAGNOSTIC_CHECKS

SUBSETS: tuple[str, ...] = (
    "GridSybil_0709",
    "GridSybil_1416",
    "DataReplaySybil_0709",
    "DataReplaySybil_1416",
    "DoSRandomSybil_0709",
    "DoSRandomSybil_1416",
    "DoSDisruptiveSybil_0709",
    "DoSDisruptiveSybil_1416",
)

# VeReMi-Extension paper Table III (per-message Accuracy / Precision / Recall / F1); GridSybil = TrafficSybil.
PAPER_TABLE_III: dict[str, dict[str, float]] = {
    "GridSybil_0709": {"accuracy": 0.8204, "precision": 0.9973, "recall": 0.5902, "f1": 0.7415},
    "GridSybil_1416": {"accuracy": 0.8001, "precision": 0.9972, "recall": 0.5842, "f1": 0.7367},
    "DataReplaySybil_0709": {"accuracy": 0.8011, "precision": 0.9200, "recall": 0.3527, "f1": 0.5099},
    "DataReplaySybil_1416": {"accuracy": 0.7948, "precision": 0.8920, "recall": 0.3705, "f1": 0.5235},
    "DoSRandomSybil_0709": {"accuracy": 0.9898, "precision": 0.9990, "recall": 0.9784, "f1": 0.9886},
    "DoSRandomSybil_1416": {"accuracy": 0.9890, "precision": 0.9991, "recall": 0.9773, "f1": 0.9881},
    "DoSDisruptiveSybil_0709": {"accuracy": 0.7699, "precision": 0.9822, "recall": 0.5014, "f1": 0.6639},
    "DoSDisruptiveSybil_1416": {"accuracy": 0.7787, "precision": 0.9880, "recall": 0.5318, "f1": 0.6914},
}


@dataclasses.dataclass(frozen=True)
class PlausibilityConfig:
    """Every parameter of the checks and the decision; saved to results.json.

    Attributes:
        max_plausible_range: m (C1).
        max_plausible_speed, max_plausible_accel, max_plausible_decel: receiver vType limits + 0.01 (m/s, m/s^2);
            spec default passenger1 (70, 2.6, 4.5), UNKNOWN per receiver.
        max_time_delta: s, C5 / C5b apply only below it.
        max_mgt_rng, max_mgt_rng_down, max_mgt_rng_up: m (C5b / C5 / C5).
        max_delta_inter: s, overlap test window (C6).
        receiver_inter_dt: s, fixed time difference used for the overlap with the receiver itself (C6).
        intersection_limit: C6 value is 0 when RectRectFactor * time weight exceeds it.
        vehicle_width, vehicle_length: m, every vehicle's rectangle (UNKNOWN in the data; passenger1 length).
        max_heading_change: deg; pos_heading_time: s; pos_heading_min_distance: m; pos_heading_min_speed: m/s (C7).
        max_beacon_frequency: s (C8).
        max_sa_range: m; max_sa_time: s (C9, diagnostic only).
        max_kalman_time: s; kalman_pos_range, kalman_speed_range, kalman_min_pos_range, kalman_min_speed_range
            (C11-C15); kalman_binarise: Kalman outputs below it become 0, else 1.
        svi_q, svi_p0, sc_q, sc_p0, si_q, si_p0: Kalman process noise / initial covariance (diagonal, per state).
        max_nodes: node table size (pseudonyms kept per receiver; the oldest latest BSM is evicted).
        threshold: ThresholdApp threshold; a decision check with value <= threshold flags the BSM.
        decision_checks: the checks of the decision rule (subset of DECISION_CHECKS).
        checks_version: "legacy" (LegacyChecks, binary) or "catch" (CaTChChecks, continuous with confidences).
        ellipse_vertices: polygon vertices used to approximate each ellipse in the CaTCh overlap check (F2MD
            computes the exact ellipse-ellipse area with a root solver).
    """

    max_plausible_range: float = 420.0
    max_plausible_speed: float = 70.01
    max_plausible_accel: float = 2.61
    max_plausible_decel: float = 4.51
    max_time_delta: float = 3.1
    max_mgt_rng: float = 4.0
    max_mgt_rng_down: float = 6.2
    max_mgt_rng_up: float = 2.1
    max_delta_inter: float = 2.0
    receiver_inter_dt: float = 0.11
    intersection_limit: float = 0.5
    vehicle_width: float = 1.8
    vehicle_length: float = 5.0
    max_heading_change: float = 90.0
    pos_heading_time: float = 1.1
    pos_heading_min_distance: float = 1.0
    pos_heading_min_speed: float = 1.0
    max_beacon_frequency: float = 0.99
    max_sa_range: float = 210.0
    max_sa_time: float = 2.1
    max_kalman_time: float = 3.1
    kalman_pos_range: float = 1.0
    kalman_speed_range: float = 4.0
    kalman_min_pos_range: float = 4.0
    kalman_min_speed_range: float = 1.0
    kalman_binarise: float | None = 0.5
    svi_q: tuple[float, float] = (50.0, 10.0)
    svi_p0: tuple[float, float] = (1.0, 1.0)
    sc_q: tuple[float, float] = (10.0, 10.0)
    sc_p0: tuple[float, float] = (10.0, 10.0)
    si_q: float = 10.0
    si_p0: float = 10.0
    max_nodes: int = 200
    threshold: float = 0.28125
    decision_checks: tuple[str, ...] = DECISION_CHECKS
    checks_version: str = "legacy"
    ellipse_vertices: int = 256

    def __post_init__(self) -> None:
        if self.checks_version not in ("legacy", "catch"):
            raise ValueError(f"checks_version must be legacy or catch, got {self.checks_version!r}")
        unknown = sorted(set(self.decision_checks) - set(DECISION_CHECKS))
        if unknown:
            raise ValueError(f"unknown decision checks {unknown}; known: {DECISION_CHECKS}")
        if not 0.0 <= self.threshold < 1.0:
            raise ValueError(f"threshold must be in [0, 1), got {self.threshold}")
        for name in ("max_plausible_range", "max_plausible_speed", "max_plausible_accel", "max_plausible_decel"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.vehicle_length <= 0 or self.vehicle_width <= 0:
            raise ValueError(f"vehicle size must be > 0, got {self.vehicle_width} x {self.vehicle_length}")
        if self.max_nodes < 1:
            raise ValueError(f"max_nodes must be >= 1, got {self.max_nodes}")

    def to_dict(self) -> dict:
        """JSON-ready settings."""
        out = dataclasses.asdict(self)
        out["decision_checks"] = list(self.decision_checks)
        return out


VARIANTS: dict[str, PlausibilityConfig] = {
    "catch": PlausibilityConfig(checks_version="catch", kalman_binarise=None),
    "legacy": PlausibilityConfig(checks_version="legacy"),
    "catch_accel301": PlausibilityConfig(checks_version="catch", kalman_binarise=None, max_plausible_accel=3.01),
}
HEADLINE_VARIANT = "catch"
