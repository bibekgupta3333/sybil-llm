"""Tests of the plausibility-check baseline (src/baselines/plausibility) on synthetic traces (no data files)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.baselines.plausibility import catch_checks as catch_lib
from src.baselines.plausibility import checks as checks_lib
from src.baselines.plausibility.config import ALL_CHECKS, VARIANTS, PlausibilityConfig
from src.baselines.plausibility.detector import ThresholdDetector
from src.baselines.plausibility.metrics import DetectionCounts, confusion_scores
from src.baselines.plausibility.parser import (
    UNKNOWN_LABEL,
    DatasetLayout,
    ReceiverStream,
    RunIndex,
    TraceFileName,
    TraceFormatError,
    TraceReader,
)
from src.baselines.plausibility.run import BaselineRun, Job, JobRunner

CFG = PlausibilityConfig()


@pytest.fixture(params=["legacy", "catch"])
def cfg(request) -> PlausibilityConfig:
    """Both check versions: every per-check test runs on LegacyChecks and on CaTChChecks."""
    return VARIANTS[request.param]


def make_stream(
    pos: list[tuple[float, float]],
    spd: list[tuple[float, float]] | None = None,
    times: list[float] | None = None,
    pseudo: list[int] | None = None,
    hed: list[tuple[float, float]] | None = None,
    rx_pos: tuple[float, float] = (0.0, 0.0),
    label: list[int] | None = None,
) -> ReceiverStream:
    """A receiver stream with the given claims; defaults: one pseudonym, 1 Hz, 10 m/s along +x, heading +x."""
    n = len(pos)
    times = times if times is not None else [float(i) for i in range(n)]
    spd = spd if spd is not None else [(10.0, 0.0)] * n
    hed = hed if hed is not None else [(1.0, 0.0)] * n
    pseudo = pseudo if pseudo is not None else [7] * n
    arr = lambda v: np.asarray(v, dtype=np.float64).reshape(n, 2)  # noqa: E731
    return ReceiverStream(
        receiver=1,
        receiver_attack_code=0,
        rcv_time=np.asarray(times, dtype=np.float64),
        send_time=np.asarray(times, dtype=np.float64),
        sender=np.asarray(pseudo, dtype=np.int64),
        pseudo=np.asarray(pseudo, dtype=np.int64),
        message_id=np.arange(n, dtype=np.int64),
        pos=arr(pos),
        spd=arr(spd),
        acl=np.zeros((n, 2)),
        hed=arr(hed),
        pos_conf=np.full((n, 2), 3.0),
        spd_conf=np.full((n, 2), 0.01),
        hed_conf=np.full((n, 2), 2.0),
        rx_pos=np.tile(np.asarray(rx_pos, dtype=np.float64), (n, 1)),
        rx_spd=np.zeros((n, 2)),
        rx_hed=np.tile([1.0, 0.0], (n, 1)),
        rx_pos_conf=np.full((n, 2), 3.0),
        label=np.asarray(label if label is not None else [0] * n, dtype=np.int8),
    )


def straight_track(n: int = 6, start: float = 300.0, speed: float = 10.0) -> list[tuple[float, float]]:
    """Positions of a car driving along +x at `speed` m/s, 1 Hz, `start` m from the receiver (in range, no SA)."""
    return [(start + speed * i, 0.0) for i in range(n)]


def fired(stream: ReceiverStream, config: PlausibilityConfig) -> dict[str, np.ndarray]:
    """Check name -> (n,) bool fired (value <= θ) on each BSM."""
    return ThresholdDetector(config).detect(stream).fired


def test_plausible_track_passes_every_check(cfg):
    decision = ThresholdDetector(cfg).detect(make_stream(straight_track(8)))
    assert not decision.flagged.any()
    for name in ALL_CHECKS:
        assert not decision.fired[name].any(), name


def test_range_plausibility(cfg):
    stream = make_stream([(300.0, 0.0), (cfg.max_plausible_range + 50.0, 0.0)], pseudo=[1, 2])
    assert fired(stream, cfg)["range_plausibility"].tolist() == [False, True]


def test_speed_plausibility(cfg):
    spd = [(10.0, 0.0), (cfg.max_plausible_speed + 1.0, 0.0)]
    stream = make_stream([(300.0, 0.0), (300.0, 50.0)], spd=spd, pseudo=[1, 2])
    assert fired(stream, cfg)["speed_plausibility"].tolist() == [False, True]


def test_position_consistency_jump_fires(cfg):
    pos = straight_track(3)
    pos[2] = (pos[1][0] + cfg.max_plausible_speed + 30.0, 0.0)
    assert fired(make_stream(pos), cfg)["position_consistency"].tolist() == [False, False, True]


def test_speed_consistency_accel_and_decel(cfg):
    spd = [(10.0, 0.0), (15.0, 0.0), (15.0, 0.0), (9.0, 0.0)]  # +5 > 2.61 m/s^2, -6 > 4.51 m/s^2
    assert fired(make_stream(straight_track(4), spd=spd), cfg)["speed_consistency"].tolist() == [
        False,
        True,
        False,
        True,
    ]


def test_position_speed_consistency_standing_still_fires(cfg):
    stopped = [(300.0, 0.0)] * 3  # claims 10 m/s but does not move
    assert fired(make_stream(stopped), cfg)["position_speed_consistency"].tolist() == [False, True, True]


def test_position_speed_max_consistency(cfg):
    pos = straight_track(3)
    pos[2] = (pos[1][0] + 30.0, 0.0)  # 30 m in 1 s while claiming 10 m/s
    assert fired(make_stream(pos), cfg)["position_speed_max_consistency"].tolist() == [False, False, True]


def test_position_heading_consistency(cfg):
    hed = [(1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)]  # last BSM claims to drive backwards
    assert fired(make_stream(straight_track(3), hed=hed), cfg)["position_heading_consistency"].tolist() == [
        False,
        False,
        True,
    ]


def test_beacon_frequency(cfg):
    stream = make_stream(straight_track(3), times=[0.0, 1.0, 1.5])
    assert fired(stream, cfg)["beacon_frequency"].tolist() == [False, False, True]


def test_sudden_appearance_is_diagnostic_only(cfg):
    stream = make_stream([(100.0, 0.0), (110.0, 0.0), (300.0, 50.0)], pseudo=[1, 1, 2])
    decision = ThresholdDetector(cfg).detect(stream)
    assert decision.fired["sudden_appearance"].tolist() == [True, False, False]
    assert not decision.flagged.any()


def test_intersection_overlap_fires(cfg):
    stream = make_stream([(300.0, 0.0), (300.3, 0.2), (330.0, 0.0)], pseudo=[1, 2, 3], times=[0.0, 0.1, 0.2])
    assert fired(stream, cfg)["intersection"].tolist() == [False, True, False]


def test_intersection_ignores_stale_bsm(cfg):
    stream = make_stream([(300.0, 0.0), (300.3, 0.2)], pseudo=[1, 2], times=[0.0, cfg.max_delta_inter + 0.5])
    assert not fired(stream, cfg)["intersection"].any()


def test_intersection_with_receiver(cfg):
    stream = make_stream([(0.2, 0.1)], rx_pos=(0.0, 0.0))
    assert fired(stream, cfg)["intersection"].tolist() == [True]


def test_kalman_position_jump_fires(cfg):
    pos = straight_track(8)
    pos[6] = (pos[6][0] + 60.0, 0.0)
    out = fired(make_stream(pos), cfg)
    for name in ("kalman_pcc", "kalman_pscp"):
        assert out[name][6] and not out[name][:6].any(), name


def test_kalman_speed_jump_fires(cfg):
    spd = [(10.0, 0.0)] * 8
    spd[6] = (60.0, 0.0)
    out = fired(make_stream(straight_track(8), spd=spd), cfg)
    assert out["kalman_scc"][6] and not out["kalman_scc"][:6].any()


def test_kalman_reinitialises_after_gap(cfg):
    pos = straight_track(4)
    times = [0.0, 1.0, 2.0, 2.0 + cfg.max_kalman_time + 1.0]
    pos[3] = (900.0, 0.0)
    detection = ThresholdDetector(cfg).detect(make_stream(pos, times=times))
    assert not detection.evaluated["kalman_pcc"][3]


def test_node_table_evicts_oldest():
    config = PlausibilityConfig(max_nodes=2)
    stream = make_stream([(300.0, 0.0), (300.0, 50.0), (300.0, 100.0), (310.0, 0.0)], pseudo=[1, 2, 3, 1])
    detection = ThresholdDetector(config).detect(stream)
    assert not detection.evaluated["position_consistency"][3]  # pseudonym 1 was evicted by 3
    assert ThresholdDetector(CFG).detect(stream).evaluated["position_consistency"][3]


def test_geometry_helpers():
    assert checks_lib.heading_angle(1.0, 0.0) == pytest.approx(0.0)
    assert checks_lib.heading_angle(0.0, 1.0) == pytest.approx(270.0)
    assert checks_lib.heading_angle(0.0, -1.0) == pytest.approx(90.0)
    assert checks_lib.rect_rect_factor((0.0, 0.0), (0.0, 0.0), 30.0, 30.0, (1.8, 5.0)) == pytest.approx(1.0)
    assert checks_lib.rect_rect_factor((0.0, 0.0), (10.0, 0.0), 0.0, 0.0, (1.8, 5.0)) == 0.0
    half = checks_lib.rect_rect_factor((0.0, 0.0), (0.9, 0.0), 0.0, 0.0, (1.8, 5.0))  # half the width overlaps
    assert half == pytest.approx(0.5 / 1.5)
    min_d, max_d = checks_lib.max_min_dist(10.0, 10.0, 1.0, 2.61, 4.51, 70.01)
    assert min_d < 10.0 < max_d
    assert math.isinf(checks_lib.ieee_div(1.0, 0.0)) and math.isnan(checks_lib.ieee_div(0.0, 0.0))


def test_detector_flags_when_any_check_fires(cfg):
    pos = straight_track(4)
    pos[3] = (pos[3][0] + 200.0, 0.0)
    decision = ThresholdDetector(cfg).detect(make_stream(pos))
    assert decision.flagged.tolist() == [False, False, False, True]
    assert decision.fired["position_consistency"][3]


def test_detector_respects_decision_checks():
    config = PlausibilityConfig(decision_checks=("speed_plausibility",))
    pos = straight_track(4)
    pos[3] = (pos[3][0] + 200.0, 0.0)
    assert not ThresholdDetector(config).detect(make_stream(pos)).flagged.any()


def test_config_rejects_unknown_check():
    with pytest.raises(ValueError):
        PlausibilityConfig(decision_checks=("nope",))


def test_confusion_scores_known_values():
    scores = confusion_scores(tp=6, fp=2, tn=10, fn=2)
    assert scores["accuracy"] == pytest.approx(16 / 20)
    assert scores["precision"] == pytest.approx(0.75)
    assert scores["recall"] == pytest.approx(0.75)
    assert scores["f1"] == pytest.approx(0.75)


def test_counts_add_merge_and_unknown():
    label = np.array([1, 1, 0, 0, UNKNOWN_LABEL])
    flagged = np.array([True, False, True, False, True])
    fired = {"a": flagged.copy(), "b": np.array([True, False, False, False, False])}
    evaluated = {"a": np.ones(5, dtype=bool), "b": np.ones(5, dtype=bool)}
    one = DetectionCounts()
    one.add_messages(label, flagged, fired, evaluated)
    assert (one.tp, one.fn, one.fp, one.tn, one.unknown) == (1, 1, 1, 1, 1)
    assert one.only_check_fires["a"] == [1, 0]  # genuine message flagged by "a" alone; malicious one by both
    two = DetectionCounts()
    two.merge(one)
    two.merge(one)
    assert (two.tp, two.fp, two.check_fires["b"]) == (2, 2, [0, 2])
    rates = two.check_rates()["a"]
    assert rates["fire_rate_genuine"] == pytest.approx(0.5)
    assert rates["fire_rate_malicious"] == pytest.approx(0.5)


def _write_trace(path: Path, own: list[dict], received: list[dict]) -> None:
    lines = [json.dumps({"type": 2, **r}) for r in own] + [json.dumps({"type": 3, **r}) for r in received]
    path.write_text("\n".join(lines) + "\n")


def _bsm(t: float, sender: int, pseudo: int, x: float, mid: int) -> dict:
    vec = lambda a, b: [a, b, 0.0]  # noqa: E731
    return {
        "rcvTime": t,
        "sendTime": t,
        "sender": sender,
        "senderPseudo": pseudo,
        "messageID": mid,
        "pos": vec(x, 0.0),
        "pos_noise": vec(3.0, 3.0),
        "spd": vec(10.0, 0.0),
        "spd_noise": vec(0.0, 0.0),
        "acl": vec(0.0, 0.0),
        "acl_noise": vec(0.0, 0.0),
        "hed": vec(1.0, 0.0),
        "hed_noise": vec(1.0, 1.0),
    }


def _own(t: float, x: float) -> dict:
    return {
        k: v for k, v in _bsm(t, 0, 0, x, 0).items() if k not in ("sendTime", "sender", "senderPseudo", "messageID")
    }


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A synthetic run: receiver 10 (benign) hears benign 20 and attacker 30 (A16); 40 has no trace file."""
    run = tmp_path / "GridSybil_0709" / "VeReMi_0_3600_x"
    run.mkdir(parents=True)
    received = [_bsm(1.5, 20, 201, 300.0, 1), _bsm(1.6, 30, 301, 0.2, 2), _bsm(2.5, 20, 201, 310.0, 3)]
    received.append(_bsm(2.6, 40, 401, 600.0, 4))
    _write_trace(run / "traceJSON-10-101-A0-0-1.json", [_own(1.0, 0.0), _own(2.0, 1.0)], received)
    _write_trace(run / "traceJSON-20-201-A0-0-1.json", [_own(1.0, 300.0)], [])
    _write_trace(run / "traceJSON-30-301-A16-0-1.json", [_own(1.0, 0.2)], [])
    (run / "traceGroundTruthJSON-1.json").write_text("")
    return run


def test_trace_file_name():
    meta = TraceFileName.parse("traceJSON-21615-21613-A17-28800-8.json")
    assert (meta.vehicle, meta.pseudo, meta.attack_code, meta.start, meta.sim) == (21615, 21613, 17, 28800, 8)
    with pytest.raises(TraceFormatError):
        TraceFileName.parse("traceGroundTruthJSON-8.json")


def test_reader_labels_and_receiver_position(run_dir: Path):
    index = RunIndex.scan(run_dir)
    assert index.attack_code == {10: 0, 20: 0, 30: 16}
    stream = TraceReader(index).read(run_dir / "traceJSON-10-101-A0-0-1.json")
    assert len(stream) == 4
    assert stream.label.tolist() == [0, 1, 0, UNKNOWN_LABEL]
    assert stream.rx_pos[:, 0].tolist() == [0.0, 0.0, 1.0, 1.0]  # latest own fix at or before reception
    assert stream.pseudo.tolist() == [201, 301, 201, 401]


def test_run_writes_results(run_dir: Path, tmp_path: Path):
    out = tmp_path / "out"
    layout = DatasetLayout(run_dir.parent.parent)
    variants = {"catch": VARIANTS["catch"], "legacy": VARIANTS["legacy"]}
    summary = BaselineRun(variants, layout, out, workers=1).run(["GridSybil_0709"])
    for path in (
        out / "GridSybil_0709" / "results.json",
        out / "variants" / "legacy" / "GridSybil_0709" / "results.json",
    ):
        result = json.loads(path.read_text())
        counts = result["counts"]
        assert counts["tp"] + counts["fn"] == 1 and counts["fp"] + counts["tn"] == 2 and counts["unknown_label"] == 1
        assert counts["tp"] == 1  # the attacker's shape sits on the receiver's own (overlap check)
        assert "git_commit" in result["env"] and result["config"]["threshold"] == pytest.approx(0.28125)
    row = summary["subsets"]["GridSybil_0709"]
    assert row["paper"]["f1"] == pytest.approx(0.7415)
    assert set(row) >= {"ours", "ours_legacy", "sent_messages", "received_messages"}
    assert row["sent_messages"] == {"genuine": 0, "malicious": 0, "unknown_label": 0}  # empty ground-truth file


def test_job_runner_runs_every_variant(run_dir: Path):
    index = RunIndex.scan(run_dir)
    job = Job("GridSybil_0709", run_dir, tuple(index.files), index.attack_code)
    result = JobRunner({"legacy": VARIANTS["legacy"], "catch": VARIANTS["catch"]})(job)
    assert result.subset == "GridSybil_0709"
    assert {name: c.receivers for name, c in result.counts.items()} == {"legacy": 3, "catch": 3}


def test_ground_truth_rows_counted(run_dir: Path):
    rows = [{"type": 4, "sender": 20}, {"type": 4, "sender": 30}, {"type": 4, "sender": 30}, {"type": 4, "sender": 99}]
    (run_dir / "traceGroundTruthJSON-1.json").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    index = RunIndex.scan(run_dir)
    result = JobRunner({"legacy": VARIANTS["legacy"]})(Job("GridSybil_0709", run_dir, (), index.attack_code))
    assert result.sent == {"genuine": 1, "malicious": 2, "unknown_label": 1}


def test_catch_factor_functions():
    assert catch_lib.circle_circle_intersection(1.0, 1.0, 0.0) == pytest.approx(math.pi)
    assert catch_lib.circle_circle_intersection(1.0, 1.0, 3.0) == 0.0
    assert catch_lib.circle_circle_factor(10.0, 0.0, 0.0, 20.0) == 1.0  # no confidence: plain range test
    assert catch_lib.circle_circle_factor(30.0, 0.0, 0.0, 20.0) == 0.0
    inside = catch_lib.circle_circle_factor(100.0, 3.0, 3.0, 420.0)
    outside = catch_lib.circle_circle_factor(600.0, 3.0, 3.0, 420.0)
    assert inside == pytest.approx(1.0) and outside == pytest.approx(0.0)
    assert catch_lib.segment_segment_factor(1.0, 0.1, 0.1, 2.61) == pytest.approx(1.0)
    assert catch_lib.segment_segment_factor(5.0, 0.1, 0.1, 2.61) == pytest.approx(0.0)
    assert catch_lib.gaussian_sum(0.0, 1.0) == pytest.approx(0.5)
    assert catch_lib.gaussian_sum(1.96, 1.0) == pytest.approx(0.975, abs=1e-3)


def test_ellipse_overlap_matches_circle_formula():
    overlap = catch_lib.EllipseOverlap(4096)
    for d in (0.0, 0.5, 1.0, 1.5):
        exact = catch_lib.circle_circle_intersection(1.0, 1.0, d)
        approx = overlap.area((0.0, 0.0), 1.0, 1.0, 0.3, (d, 0.0), 1.0, 1.0, 0.3)
        assert approx == pytest.approx(exact, abs=0.01)
