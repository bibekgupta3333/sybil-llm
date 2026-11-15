"""Tests of the protocol package on synthetic encoder inputs (CPU, a few hundred windows)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from baselinepaper.experiments import config as cfg
from baselinepaper.experiments.aggregate import Aggregator
from baselinepaper.experiments.data import LabelledWindowSet, WindowCache, WindowSet, is_prefix_mask
from baselinepaper.experiments.detectors import Detector, available, get_detector, register
from baselinepaper.experiments.detectors.length import LENGTH_ONLY, LengthOnly
from baselinepaper.experiments.evaluation import Evaluator, ScoreRanking, auroc, average_precision, pair_wins
from baselinepaper.experiments.run import Runner, score_file
from baselinepaper.experiments.verify_data import CacheVerifier

T, F = cfg.SEQ_LEN, cfg.N_FEATURES
FEATURES = [f"f{i}" for i in range(F - 1)] + ["log_dtau"]
SCENARIOS = (
    "GridSybil_0709",
    "GridSybil_1416",
    "DataReplaySybil_0709",
    "DoSRandomSybil_0709",
    "DoSDisruptiveSybil_1416",
)
ATTACK_OF_SCENARIO = {"GridSybil": 16, "DataReplaySybil": 17, "DoSRandomSybil": 18, "DoSDisruptiveSybil": 19}
TEST_SPLIT_OF_CODE = {16: "test_gridsybil", 17: "test_datareplay", 18: "test_dosrandom", 19: "test_dosdisruptive"}


class SyntheticInputs:
    """Writes a small T32_benign-like and T32_benign_grid-like input from the same raw links."""

    def __init__(self, root: Path, seed: int = 0) -> None:
        self.root = root
        rng = np.random.default_rng(seed)
        self.benign_stats = (rng.normal(size=F), rng.uniform(0.5, 2.0, size=F))
        self.grid_stats = (rng.normal(size=F), rng.uniform(0.5, 2.0, size=F))
        self.windows = []  # dicts: id, scenario, sender, k, n, code, raw (n, F)
        for scenario in SCENARIOS:
            family, group = scenario.rsplit("_", 1)
            for sender in range(60):
                # senders 0-35 benign everywhere; 36-59 attackers of this folder's family
                code = 0 if sender < 36 else ATTACK_OF_SCENARIO[family]
                for link in range(2):
                    short = code in (17, 18, 19)
                    n_link = int(rng.integers(1, 4)) if short else int(rng.choice([1, 2, 3, 9, 20, 40, 70]))
                    raw = rng.normal(size=(n_link, F)) * 3 + (2.0 if code == 16 else 0.0)
                    for k, start in enumerate(range(0, n_link, T)):
                        wid = f"{scenario}/run/traceJSON-{sender}-{link}.json#{100 + link}-{sender}#{k}"
                        self.windows.append(
                            {
                                "id": wid,
                                "scenario": scenario,
                                "sender": sender,
                                "k": k,
                                "code": code,
                                "raw": raw[start : start + T],
                                "key": f"{group}:{sender}",
                            }
                        )
        keys = sorted({w["key"] for w in self.windows})
        self.benign_split = {k: ("test", "val", "train", "train", "train")[i % 5] for i, k in enumerate(keys)}
        self.grid_split = {
            k: ("train", "train", "test", "val", "train", "train", "train")[i % 7] for i, k in enumerate(keys)
        }

    def _encode(self, w: dict, stats: tuple[np.ndarray, np.ndarray]) -> tuple[list, list]:
        n = len(w["raw"])
        x = np.zeros((T, F))
        x[:n] = (w["raw"] - stats[0]) / stats[1]
        if w["k"] == 0:
            x[0, F - 1] = 0.0  # first message of a link: no previous message, log_dtau stored as 0
        return x.round(4).tolist(), [1] * n + [0] * (T - n)

    def _write(self, root: Path, split: str, windows: list[dict], stats, grid: bool) -> None:
        folder = root / split / "b32"
        folder.mkdir(parents=True, exist_ok=True)
        half = max(1, len(windows) // 2)
        for part, chunk in enumerate([windows[:half], windows[half:]]):
            data, info = [], []
            for w in chunk:
                x, mask = self._encode(w, stats)
                data.append({"id": w["id"], "x": x, "mask": mask})
                record = {
                    "id": w["id"],
                    "family": w["scenario"].rsplit("_", 1)[0],
                    "scenario": w["scenario"],
                    "sender": w["sender"],
                    "link_window_index": w["k"],
                    "n_messages": len(w["raw"]),
                    "label": w["code"],
                }
                if grid:
                    record.update(label=int(w["code"] == 16), attack_code=w["code"])
                info.append(record)
            (folder / f"part-{part:05d}.json").write_text(json.dumps({"windows": data}))
            (folder / f"part-{part:05d}_info.json").write_text(json.dumps({"windows": info}))

    def _metadata(self, root: Path, stats, split: dict, counts: dict) -> None:
        ids = {s: sorted(k for k, v in split.items() if v == s) for s in ("train", "val", "test")}
        meta = {
            "normalisation": {"features": FEATURES, "mean": stats[0].tolist(), "std": stats[1].tolist()},
            "split": {"vehicle_ids": ids},
            "counts_by_split": counts,
        }
        (root / "metadata.json").write_text(json.dumps(meta))

    def write(self) -> tuple[Path, Path]:
        benign_root, grid_root = self.root / "T32_benign", self.root / "T32_benign_grid"
        by_split: dict[str, list] = {}
        for w in self.windows:
            part = self.benign_split[w["key"]]
            if w["code"] == 0:
                by_split.setdefault(part, []).append(w)
            elif part == "test":
                by_split.setdefault(TEST_SPLIT_OF_CODE[w["code"]], []).append(w)
        counts = {}
        for split, windows in by_split.items():
            self._write(benign_root, split, windows, self.benign_stats, grid=False)
            counts[split] = {"windows": len(windows), "messages": sum(len(w["raw"]) for w in windows)}
        self._metadata(benign_root, self.benign_stats, self.benign_split, counts)
        counts = {}
        for split in ("train", "val", "test"):
            windows = [w for w in self.windows if w["code"] in (0, 16) and self.grid_split[w["key"]] == split]
            self._write(grid_root, split, windows, self.grid_stats, grid=True)
            counts[split] = {"windows": len(windows), "messages": sum(len(w["raw"]) for w in windows)}
        self._metadata(grid_root, self.grid_stats, self.grid_split, counts)
        return benign_root, grid_root


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> tuple[cfg.ProtocolConfig, SyntheticInputs, WindowCache]:
    root = tmp_path_factory.mktemp("protocol")
    inputs = SyntheticInputs(root)
    benign_root, grid_root = inputs.write()
    config = cfg.ProtocolConfig(
        benign_input_dir=str(benign_root),
        grid_input_dir=str(grid_root),
        cache_dir=str(root / "cache"),
        runs_dir=str(root / "runs"),
        results_dir=str(root / "results"),
        seeds=(0, 1),
        bootstrap_resamples=40,
        min_attack_windows=5,
        min_attack_vehicles=2,
        workers=1,
        min_free_gb=0.0,
    )
    cache = WindowCache(config)
    cache.build(log=lambda *_: None)
    return config, inputs, cache


@register
class NoiseDetector(Detector):
    """Test detector: seeded noise plus the mean of the first feature; records what it was given."""

    name = "test_noise"
    regimes = (cfg.BENIGN_ONLY, cfg.MIX_ALL, cfg.SUPERVISED)
    is_stochastic = True
    score_names = ("noisy", "plain")
    seen: dict = {}

    def fit(self, train, val, seed):
        self.rng = np.random.default_rng(seed)
        NoiseDetector.seen[self.regime] = (type(train), type(val), len(train), len(val), self.statistics is not None)

    def score(self, x, mask):
        plain = np.asarray(self.standardise(x, mask))[:, :, 0].sum(axis=1) / mask.sum(axis=1)
        return {"noisy": plain + self.rng.normal(size=len(x)), "plain": plain}

    def history(self):
        return [{"epoch": 1, "loss": 0.5}]


# ------------------------------------------------------------------------------------------------ data


def test_cache_splits_and_rederivation(setup):
    config, inputs, cache = setup
    for name in cfg.ALL_SPLITS:
        s = cache.split(name)
        assert s.x.shape == (len(s), T, F) and s.x.dtype == np.float32 and s.mask.dtype == bool
        assert is_prefix_mask(s.mask) and not np.count_nonzero(np.asarray(s.x)[~np.asarray(s.mask)])
        assert set(s.group.tolist()) <= {"0709", "1416"} and len(s.family) == len(s)
    check = cache.rederivation_check()
    assert check["passed"] and check["same_id_set"] and check["max_abs_diff"] < 1e-3
    # train GridSybil windows = the raw rows normalised with the benign input's statistics
    raw = {w["id"]: w for w in inputs.windows}
    g = cache.split(cfg.TRAIN_GRIDSYBIL)
    assert len(g) == sum(w["code"] == 16 and inputs.benign_split[w["key"]] == "train" for w in inputs.windows)
    for i in range(0, len(g), 7):
        w = raw[g.window_id[i].decode()]
        expected = (w["raw"] - inputs.benign_stats[0]) / inputs.benign_stats[1]
        if w["k"] == 0:
            expected[0, F - 1] = 0.0
        assert np.abs(np.asarray(g.x[i, : len(expected)]) - expected).max() < 1e-3
        assert inputs.benign_split[g.vehicle_key[i]] == "train"


def test_regime_sets_and_labels(setup):
    config, _, cache = setup
    b, g = cache.split(cfg.TRAIN_BENIGN), cache.split(cfg.TRAIN_GRIDSYBIL)
    in_folders = int(np.isin(b.scenario, config.gridsybil_scenarios).sum())
    expected = {
        cfg.BENIGN_ONLY: len(b),
        cfg.BENIGN_GRIDFOLDERS: in_folders,
        cfg.MIX_ALL: len(b) + len(g),
        cfg.MIX_GRIDFOLDERS: in_folders + len(g),
        cfg.SUPERVISED: len(b) + len(g),
    }
    for regime, n in expected.items():
        train, val = cache.training_set(regime), cache.validation_set(regime)
        assert len(train) == n == cache.regime_composition(regime)["windows"]
        assert len(val) == cache.regime_composition(regime, "val")["windows"]
        if regime == cfg.SUPERVISED:
            assert isinstance(train, LabelledWindowSet) and isinstance(val, LabelledWindowSet)
            assert int(train.labels.sum()) == len(g)
            # the labelled rows are exactly the GridSybil windows, and the pooled order is not sorted by class
            assert np.allclose(
                np.sort(train.x[train.labels == 1].sum(axis=(1, 2))), np.sort(np.asarray(g.x).sum(axis=(1, 2)))
            )
            assert 0 < train.labels[: len(train) // 2].sum() < len(g)
        else:
            assert type(train) is WindowSet and type(val) is WindowSet
            assert not hasattr(train, "labels") and not hasattr(val, "labels")
    again = cache.training_set(cfg.MIX_ALL)
    assert np.array_equal(again.x, cache.training_set(cfg.MIX_ALL).x)
    with pytest.raises(ValueError):
        cache.training_set("none")


def test_regime_statistics(setup):
    config, _, cache = setup
    stats = cache.regime_statistics(cfg.MIX_GRIDFOLDERS)
    train = cache.training_set(cfg.MIX_GRIDFOLDERS)
    rows = train.x[train.mask]
    assert np.allclose(stats.mean[:-1], rows[:, :-1].mean(axis=0), atol=1e-5)
    assert np.allclose(stats.std[:-1], rows[:, :-1].std(axis=0), atol=1e-5)
    assert stats.n_values[-1] < stats.n_values[0] == len(rows)  # missing log_dtau cells left out
    out = stats.apply(train.x, train.mask)
    assert not np.count_nonzero(out[~train.mask]) and abs(out[train.mask][:, 0].mean()) < 1e-4
    assert np.allclose(cache.regime_statistics(cfg.MIX_GRIDFOLDERS).mean, stats.mean)  # stored copy


def test_verify_data_passes(setup):
    config, _, _ = setup
    report = CacheVerifier(config).run()
    assert report["passed"], report
    assert (Path(config.cache_dir) / "verify_report.json").exists()
    assert not any(report["b_disjoint_vehicles"]["keys_in_two_parts"].values())


# ------------------------------------------------------------------------------------------------ detectors, runner


def test_length_only_and_registry(setup):
    assert get_detector(LENGTH_ONLY) is LengthOnly and LENGTH_ONLY in available()
    assert not LengthOnly.is_stochastic and LengthOnly.regimes == (cfg.CONTROL_REGIME,)
    mask = np.zeros((3, T), bool)
    mask[0, :1], mask[1, :5], mask[2, :] = True, True, True
    scores = LengthOnly(cfg.CONTROL_REGIME).score(np.zeros((3, T, F), np.float32), mask)["length"]
    assert scores.tolist() == [-1.0, -5.0, -32.0]  # shorter = more anomalous
    with pytest.raises(ValueError):
        LengthOnly(cfg.BENIGN_ONLY)
    with pytest.raises(KeyError):
        get_detector("no_such_detector")
    with pytest.raises(ValueError):

        @register
        class Bad(Detector):
            name = "bad"
            regimes = (cfg.BENIGN_ONLY,)
            is_stochastic = False
            score_names = ("with_underscore",)


def test_runner_writes_run_folder(setup):
    config, _, cache = setup
    with pytest.raises(ValueError):
        Runner(LENGTH_ONLY, cfg.CONTROL_REGIME, seed=1, config=config)  # deterministic: one run
    with pytest.raises(ValueError):
        Runner(LENGTH_ONLY, cfg.BENIGN_ONLY, seed=0, config=config)
    log = []
    Runner(LENGTH_ONLY, cfg.CONTROL_REGIME, 0, "cpu", config).run(log=log.append)
    for regime in NoiseDetector.regimes:
        for seed in config.seeds:
            runner = Runner("test_noise", regime, seed, "cpu", config)
            summary = runner.run(log=log.append)
            assert summary["status"] == "complete" and runner.finished()
            for name in NoiseDetector.score_names:
                for split in cfg.SCORED_SPLITS:
                    assert np.load(score_file(runner.run_dir, name, split)).shape == (len(cache.split(split)),)
    run_dir = config.run_dir(cfg.MIX_ALL, "test_noise", 1)
    env = json.loads((run_dir / "env.json").read_text())
    assert any(k.endswith("experiments/run.py") for k in env["source_sha256"]) and env["device"] == "cpu"
    assert env["cache_meta_sha256"] == cache.fingerprint() and "numpy" in env["packages"]
    run_config = json.loads((run_dir / "config.json").read_text())
    assert run_config["seed"] == 1 and run_config["training_pool"]["attack_share"] > 0
    assert run_config["restandardised_with_regime_statistics"] is not None
    assert json.loads((run_dir / "history.json").read_text()) == [{"epoch": 1, "loss": 0.5}]
    assert "auroc" not in (run_dir / "summary.json").read_text()  # no metric in a run folder
    # what the detector was given: unlabelled sets outside the supervised regime
    assert (
        NoiseDetector.seen[cfg.BENIGN_ONLY][:2] == (WindowSet, WindowSet) and not NoiseDetector.seen[cfg.BENIGN_ONLY][4]
    )
    assert NoiseDetector.seen[cfg.MIX_ALL][:2] == (WindowSet, WindowSet) and NoiseDetector.seen[cfg.MIX_ALL][4]
    assert NoiseDetector.seen[cfg.SUPERVISED][:2] == (LabelledWindowSet, LabelledWindowSet)
    # a finished run is skipped, --force redoes it
    before = (run_dir / "summary.json").stat().st_mtime_ns
    Runner("test_noise", cfg.MIX_ALL, 1, "cpu", config).run(log=log.append)
    assert (run_dir / "summary.json").stat().st_mtime_ns == before and "skipped" in log[-1]
    Runner("test_noise", cfg.MIX_ALL, 1, "cpu", config).run(force=True, log=log.append)
    assert (run_dir / "summary.json").stat().st_mtime_ns != before


# ------------------------------------------------------------------------------------------------ evaluation


def test_point_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    benign, attack = rng.integers(0, 6, 300).astype(float), rng.integers(2, 9, 200).astype(float)  # many ties
    y, s = np.r_[np.zeros(300), np.ones(200)], np.r_[benign, attack]
    assert auroc(benign, attack) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert average_precision(benign, attack) == pytest.approx(average_precision_score(y, s), abs=1e-12)
    assert np.isnan(auroc(benign, np.array([])))


def test_cluster_bootstrap_matches_weighted_sklearn():
    rng = np.random.default_rng(1)
    benign, attack = rng.integers(0, 8, 400).astype(float), rng.integers(3, 12, 250).astype(float)
    cb, ca = rng.integers(0, 9, 400), rng.integers(0, 5, 250)
    wins = pair_wins(benign, cb, 9, attack, ca, 5)
    assert wins.sum() / (400 * 250) == pytest.approx(auroc(benign, attack), abs=1e-12)
    wb, wa = rng.multinomial(9, np.full(9, 1 / 9), size=6).astype(float), rng.multinomial(
        5, np.full(5, 1 / 5), size=6
    ).astype(float)
    y, s = np.r_[np.ones(250), np.zeros(400)], np.r_[attack, benign]
    ranking = ScoreRanking(benign, attack)
    ap = ranking.average_precision(np.concatenate([wa, wb], axis=1), np.r_[ca, 5 + cb], chunk=4)
    for r in range(6):
        weights = np.r_[wa[r][ca], wb[r][cb]]
        keep = weights > 0
        resampled = (wa[r] @ wins @ wb[r]) / (
            wa[r] @ np.bincount(ca, minlength=5) * (wb[r] @ np.bincount(cb, minlength=9))
        )
        assert resampled == pytest.approx(roc_auc_score(y[keep], s[keep], sample_weight=weights[keep]), abs=1e-10)
        assert ap[r] == pytest.approx(average_precision_score(y[keep], s[keep], sample_weight=weights[keep]), abs=1e-10)


def test_evaluation_and_aggregate(setup):
    config, _, cache = setup
    if not Runner(LENGTH_ONLY, cfg.CONTROL_REGIME, 0, "cpu", config).finished():
        test_runner_writes_run_folder(setup)
    evaluator = Evaluator(config, cache)
    groups = {g.key: g for g in evaluator.discover()}
    assert f"none/{LENGTH_ONLY}/length" in groups and "mix_all/test_noise/noisy" in groups
    assert groups["mix_all/test_noise/noisy"].seeds == config.seeds
    length = evaluator.evaluate(groups[f"none/{LENGTH_ONLY}/length"])
    benign = cache.split(cfg.TEST_BENIGN)
    for attack_split in cfg.ATTACK_TEST_SPLITS:
        attack, record = cache.split(attack_split), length.summary["tasks"][attack_split]
        y = np.r_[np.zeros(len(benign)), np.ones(len(attack))]
        s = -np.r_[benign.n_messages, attack.n_messages].astype(float)
        assert record["auroc"]["mean"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
        assert record["average_precision"]["mean"] == pytest.approx(average_precision_score(y, s), abs=1e-12)
        assert record["length_matched_auroc"]["mean"] == pytest.approx(0.5)  # ties only within one length
        assert (
            record["auroc"]["std"] is None
            and record["auroc"]["ci"][0] <= record["auroc"]["mean"] <= record["auroc"]["ci"][1]
        )
        stratum = record["strata"]["rows_ge_17"]
        assert (stratum["auroc"] is None) == (not stratum["reported"])
        if attack_split != "test_gridsybil":
            assert not stratum["reported"] and stratum["n_attack"] == 0  # 1-3-row attacks: n/a with counts

    noisy = evaluator.evaluate(groups["mix_all/test_noise/noisy"])
    summary = noisy.summary
    assert (
        summary["primary_operating_point"] == "own_val_p95"
        and len(summary["tasks"]["test_gridsybil"]["auroc"]["per_seed"]) == 2
    )
    assert (
        evaluator.evaluate(groups["benign_only/test_noise/noisy"]).summary["primary_operating_point"]
        == "benign_val_p95"
    )
    # thresholds and length-matched AUROC recomputed by hand for seed 0
    run_dir = Path(groups["mix_all/test_noise/noisy"].run_dirs[0])
    val_b, val_g = (np.load(score_file(run_dir, "noisy", s)) for s in (cfg.VAL_BENIGN, cfg.VAL_GRIDSYBIL))
    assert summary["thresholds_per_seed"]["benign_val_p99"][0] == pytest.approx(np.percentile(val_b, 99))
    assert summary["thresholds_per_seed"]["own_val_p95"][0] == pytest.approx(np.percentile(np.r_[val_b, val_g], 95))
    sb, sa = np.load(score_file(run_dir, "noisy", cfg.TEST_BENIGN)), np.load(
        score_file(run_dir, "noisy", "test_gridsybil")
    )
    attack = cache.split("test_gridsybil")
    wins = pairs = 0.0
    for n in range(1, T + 1):
        b, a = sb[benign.n_messages == n], sa[attack.n_messages == n]
        if len(a) and len(b):
            wins, pairs = wins + auroc(b, a) * len(a) * len(b), pairs + len(a) * len(b)
    task = summary["tasks"]["test_gridsybil"]
    assert task["length_matched_auroc"]["per_seed"][0] == pytest.approx(wins / pairs)
    threshold = summary["thresholds_per_seed"]["own_val_p95"][0]
    assert task["recall"]["own_val_p95"]["per_seed"][0] == pytest.approx((sa > threshold).mean())
    assert summary["benign_test_fpr"]["own_val_p95"]["per_seed"][0] == pytest.approx((sb > threshold).mean())
    in_group = (benign.group == "1416"), (attack.group == "1416")
    assert task["sensitivity"]["group_1416_only"]["auroc"]["per_seed"][0] == pytest.approx(
        auroc(sb[in_group[0]], sa[in_group[1]])
    )

    results = Aggregator(config, jobs=1).run(log=lambda *_: None)
    differences = results["paired_differences"]
    d = differences["detector_minus_length_only"]["mix_all/test_noise/noisy"]["test_datareplay"]["auroc"]
    assert d["difference"] == pytest.approx(
        noisy.summary["tasks"]["test_datareplay"]["auroc"]["mean"]
        - length.summary["tasks"]["test_datareplay"]["auroc"]["mean"]
    )
    assert d["ci"][0] <= d["ci"][1]
    assert "test_noise/noisy: mix_all - benign_only" in differences["regime_minus_regime"]
    table = (Path(config.results_dir) / "main_table.md").read_text()
    assert (
        "length_only / length (deterministic)" in table and "test_noise / noisy (2 seeds)" in table and "n/a (" in table
    )
    assert json.loads((Path(config.results_dir) / "results.json").read_text())["entries"]
