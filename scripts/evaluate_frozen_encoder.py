"""Frozen-encoder evaluation of a pretraining run (plan tasks S1.4.1, S1.4.2, S1.4.4 and the S1.4.5 random-init control).

Representations compared on the same windows:

* ``model-all``: z of the run's frozen encoder (masked mean pool, L2-normalised; loading and scaling reused from
  ``scripts/export_detection_sample.py``).
* ``random-init``: the same TimesNet architecture with random weights (torch seed = the evaluation seed), the
  "no pretraining" control.
* ``length``: the number of real rows n (the length shortcut, F11).
* ``stats``: per-window mean / std / min / max of the 13 scaled features over real rows, plus n.

Data:

* Probe-fit set: a seeded, class-stratified sample of the ``train`` split (up to ``per_class`` windows per class; both
  scenario groups; check-set vehicles allowed). Train labels (``_info`` files) are read here, only to stratify the
  sample and fit the probes.
* Test: every ``test`` window of scenario group 1416 (0709 waits on F14). Test label files are opened only after every
  test prediction and anomaly score exists (``LabelVault`` from the exporter).

Probes per representation and seed: logistic regression (standardised, class_weight balanced) for binary
attack-vs-benign and 5-class, and kNN (k = 10, cosine; Euclidean for the 1-d length feature) whose votes give both.
Label-free kNN anomaly score (reused ``KnnScorer`` / ``Threshold`` / ``TrainSampler``): bank and calibration from
unlabeled train windows.

Metrics on the 1416 test windows, mean ± std over seeds and a 95% bootstrap CI of the seed mean, resampling sender
vehicles (windows of one vehicle are correlated), for the main numbers. Every per-family number is also reported on
windows with n <= 3 and n >= 4 separately.

Writes ``src/runs/evaluation/all/T24/model-all/`` (results.json, config.json, env.json, report_data.json). Nothing
under ``data/``, ``src/data/`` or ``src/runs/pretraining/`` is written.

Usage:
    .venv-train/bin/python scripts/evaluate_frozen_encoder.py --run src/runs/pretraining/all/T24/model-all
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gzip
import json
import logging
import pathlib
import platform
import sys
import time
import types
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
for _p in (REPO_ROOT, SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import export_detection_sample as exp  # noqa: E402
from src.model.benign_gridsybil.timesnet.data import (  # noqa: E402
    FEATURES,
    CompactWindowReader,
    FeatureShards,
)
from src.model.benign_gridsybil.timesnet.encoder import TimesNetEncoder  # noqa: E402

_LOG = logging.getLogger("evaluate_frozen_encoder")

CLASSES: tuple[str, ...] = exp.CLASSES
FAMILIES: tuple[str, ...] = CLASSES[1:]
STRATA: dict[str, tuple[int, int]] = {"all": (1, 10_000), "n<=3": (1, 3), "n>=4": (4, 10_000)}
REPRESENTATIONS: tuple[str, ...] = ("model-all", "random-init", "length", "stats")
WRITE_GUARDS = ("data", "src/data", "src/runs/pretraining")


class EvalError(ValueError):
    """Raised when inputs or outputs fail a consistency check."""


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    """Settings of one evaluation (all saved to config.json).

    Attributes:
        run_dir: pretraining run folder (read only).
        checkpoint: checkpoint inside run_dir.
        out_dir: results folder; must lie under allowed_out_root.
        allowed_out_root: the only directory written under.
        group: scenario group of the test windows.
        per_class: probe-fit windows per class per seed (all if fewer).
        seeds: seeds of the fit-sample draw, random-init weights and anomaly bank.
        knn_k: neighbours of the kNN probe.
        lr_c: inverse L2 strength of the logistic regression.
        lr_max_iter: lbfgs iterations.
        bank: anomaly memory-bank windows (unlabeled train, trained-on vehicles).
        calib: anomaly calibration windows (unlabeled train, check-set vehicles).
        anomaly_k: neighbours of the anomaly score.
        quantile: θ = this quantile of the calibration scores.
        bootstrap: resamples of the vehicle bootstrap.
        bootstrap_seed: seed of the bootstrap.
        bank_candidate_rate: share of train windows kept as bank / calibration candidates (exporter's sampler).
        candidate_margin: oversampling of the per-class candidate rate (keeps the draw exact and uniform).
        device: torch device ("auto" = MPS, CUDA, CPU).
        batch_size: windows per encoder batch.
        knn_batch: queries per kNN batch.
        workers: parallel shard readers.
    """

    run_dir: pathlib.Path = pathlib.Path("src/runs/pretraining/all/T24/model-all")
    checkpoint: str = "best.pt"
    out_dir: pathlib.Path = pathlib.Path("src/runs/evaluation/all/T24/model-all")
    allowed_out_root: pathlib.Path = pathlib.Path("src/runs/evaluation")
    group: str = "1416"
    per_class: int = 20_000
    seeds: tuple[int, ...] = (0, 1, 2)
    knn_k: int = 10
    lr_c: float = 1.0
    lr_max_iter: int = 2000
    bank: int = 60_000
    calib: int = 20_000
    anomaly_k: int = 10
    quantile: float = 0.95
    bootstrap: int = 1000
    bootstrap_seed: int = 0
    candidate_margin: float = 1.5
    bank_candidate_rate: float = 0.06
    device: str = "auto"
    batch_size: int = 2048
    knn_batch: int = 1024
    workers: int = 8

    def to_json(self) -> dict[str, Any]:
        """JSON-safe dict."""
        out = dataclasses.asdict(self)
        for key, value in out.items():
            if isinstance(value, pathlib.Path):
                out[key] = str(value)
            elif isinstance(value, tuple):
                out[key] = list(value)
        return out


def absolute(path: pathlib.Path) -> pathlib.Path:
    """Path relative to the repo root made absolute."""
    return path if path.is_absolute() else REPO_ROOT / path


# ----------------------------------------------------------------------------------------------------------- data


@dataclasses.dataclass
class FitSet:
    """Probe-fit windows (union over seeds) with their classes, groups and per-seed row selections."""

    windows: exp.WindowSet
    classes: np.ndarray
    groups: np.ndarray
    rows: dict[int, np.ndarray]


class FitCandidateReader:
    """Reads one train shard and its `_info` file; keeps windows whose per-seed uniform key is below the class rate."""

    def __init__(self, seq_len: int, seeds: Sequence[int], rates: dict[str, float]) -> None:
        self.reader = CompactWindowReader(seq_len)
        self.seeds, self.rates = tuple(seeds), dict(rates)

    def __call__(self, job: tuple[int, str]) -> tuple[exp.ShardPart, np.ndarray, np.ndarray, np.ndarray]:
        index, path = job
        part = self.reader.read(path)
        info_path = pathlib.Path(path).with_name(pathlib.Path(path).name.replace(".json.gz", "_info.json.gz"))
        with gzip.open(info_path, "rt") as f:
            info = json.load(f)["windows"]
        if [w["id"] for w in info] != part.ids:
            raise EvalError(f"{info_path}: ids do not match the feature shard")
        classes = np.array([w["label_name"] for w in info])
        groups = np.array([w["scenario"].rsplit("_", 1)[1] for w in info])
        keys = np.stack([np.random.default_rng([s, index, 1]).random(len(part)) for s in self.seeds], axis=1)
        rate = np.array([self.rates.get(c, 0.0) for c in classes])
        keep = (keys < rate[:, None]).any(axis=1)
        return exp.select_part(part, keep), keys[keep], classes[keep], groups[keep]


class ProbeFitSampler:
    """Seeded class-stratified sample of the train split: per seed, the `per_class` smallest keys of each class."""

    def __init__(self, data_dir: pathlib.Path, seq_len: int, cfg: EvalConfig) -> None:
        self.data_dir, self.seq_len, self.cfg = data_dir, seq_len, cfg

    def class_counts(self) -> dict[str, int]:
        """Train windows per class from metadata.json (summed over families)."""
        metadata = json.loads((self.data_dir / "metadata.json").read_text())
        counts = {c: 0 for c in CLASSES}
        for family in metadata["counts"]["train"].values():
            for name, entry in family.items():
                counts[name] = counts.get(name, 0) + int(entry["windows"])
        return counts

    def sample(self) -> FitSet:
        cfg = self.cfg
        counts = self.class_counts()
        rates = {c: min(1.0, cfg.candidate_margin * cfg.per_class / max(n, 1)) for c, n in counts.items()}
        paths = CompactWindowReader(self.seq_len).find(self.data_dir, "train")
        if not paths:
            raise EvalError(f"no train shards under {self.data_dir}")
        reader = FitCandidateReader(self.seq_len, cfg.seeds, rates)
        results = exp.ShardPool(cfg.workers).map(reader, list(enumerate(str(p) for p in paths)))
        pooled = exp.WindowSet.from_parts([r[0] for r in results], self.seq_len)
        keys = np.concatenate([r[1] for r in results])
        classes = np.concatenate([r[2] for r in results])
        groups = np.concatenate([r[3] for r in results])
        rows: dict[int, np.ndarray] = {}
        for si, seed in enumerate(cfg.seeds):
            picked = []
            for name in CLASSES:
                cand = np.flatnonzero((classes == name) & (keys[:, si] < rates.get(name, 0.0)))
                want = min(cfg.per_class, counts.get(name, 0))
                if len(cand) < want:
                    raise EvalError(f"seed {seed}: {len(cand)} {name} candidates for {want}; raise candidate_margin")
                picked.append(cand[np.argsort(keys[cand, si], kind="stable")[:want]])
            rows[seed] = np.sort(np.concatenate(picked))
        used = np.unique(np.concatenate(list(rows.values())))
        remap = np.full(len(pooled), -1)
        remap[used] = np.arange(len(used))
        return FitSet(pooled.take(used), classes[used], groups[used], {s: remap[r] for s, r in rows.items()})


# ------------------------------------------------------------------------------------------------ representations


class EncoderFeatures:
    """WindowSet -> L2-normalised z of an encoder (reuses the exporter's Embedder)."""

    metric = "cosine"

    def __init__(self, run: exp.TrainedRun, encoder: torch.nn.Module, device: torch.device, batch: int) -> None:
        holder = types.SimpleNamespace(encoder=encoder.eval(), stored=run.stored, space=run.space)
        self.embed = exp.Embedder(holder, device, batch)  # type: ignore[arg-type]

    def __call__(self, windows: exp.WindowSet) -> np.ndarray:
        return self.embed(windows).numpy().astype(np.float32)


class LengthFeatures:
    """WindowSet -> n real rows (1 column)."""

    metric = "euclidean"

    def __call__(self, windows: exp.WindowSet) -> np.ndarray:
        return windows.lengths.astype(np.float32)[:, None]


class StatsFeatures:
    """WindowSet -> mean / std / min / max of the 13 scaled features over real rows, plus n (53 columns)."""

    metric = "cosine"

    def __init__(self, run: exp.TrainedRun, chunk: int = 50_000) -> None:
        self.stored, self.space, self.chunk = run.stored, run.space, chunk

    def __call__(self, windows: exp.WindowSet) -> np.ndarray:
        out = []
        for start in range(0, len(windows), self.chunk):
            x, m = windows.x[start : start + self.chunk], windows.mask[start : start + self.chunk]
            x = self.space.norm(self.stored.raw(x), m).double()
            w = m.unsqueeze(-1).double()
            n = w.sum(1).clamp(min=1)
            mean = (x * w).sum(1) / n
            std = (((x - mean.unsqueeze(1)) ** 2) * w).sum(1).div(n).sqrt()
            big = torch.tensor(1e9, dtype=x.dtype)
            mn = torch.where(m.unsqueeze(-1), x, big).amin(1)
            mx = torch.where(m.unsqueeze(-1), x, -big).amax(1)
            out.append(torch.cat([mean, std, mn, mx, n[:, :1]], dim=1).float().numpy())
        return np.concatenate(out) if out else np.zeros((0, 4 * len(FEATURES) + 1), np.float32)


# --------------------------------------------------------------------------------------------------------- probes


class LogisticProbe:
    """StandardScaler + logistic regression (class_weight balanced); returns class probabilities."""

    def __init__(self, c: float, max_iter: int, seed: int) -> None:
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=max_iter, class_weight="balanced", random_state=seed),
        )
        self.converged = True

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticProbe":
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.model.fit(x, y)
        self.converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        return self

    def proba(self, x: np.ndarray, n_classes: int) -> np.ndarray:
        out = np.zeros((len(x), n_classes))
        out[:, self.model.classes_] = self.model.predict_proba(x)
        return out


class KnnProbe:
    """k nearest fit windows (cosine on L2-normalised rows, or Euclidean); returns class vote shares."""

    def __init__(self, k: int, metric: str, device: torch.device, batch: int) -> None:
        self.k, self.metric, self.device, self.batch = k, metric, device, batch

    def _prep(self, x: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        if self.metric == "cosine":
            t = torch.nn.functional.normalize(t, dim=1)
        return t.to(self.device)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KnnProbe":
        if self.k > len(x):
            raise EvalError("k exceeds the fit set")
        self.bank, self.labels = self._prep(x), torch.as_tensor(y, dtype=torch.long, device=self.device)
        return self

    @torch.no_grad()
    def votes(self, x: np.ndarray, n_classes: int) -> np.ndarray:
        q = self._prep(x)
        out = []
        for start in range(0, len(q), self.batch):
            part = q[start : start + self.batch]
            sims = part @ self.bank.T if self.metric == "cosine" else -torch.cdist(part, self.bank)
            idx = sims.topk(self.k, dim=1).indices
            lab = self.labels[idx]
            counts = torch.zeros(len(part), n_classes, device=self.device)
            counts.scatter_add_(1, lab, torch.ones_like(lab, dtype=counts.dtype))
            out.append((counts / self.k).cpu())
        return torch.cat(out).numpy().astype(np.float64) if out else np.zeros((0, n_classes))


class Standardiser:
    """Mean / std fitted on one set, applied to others (for kNN and anomaly on non-encoder features)."""

    def __init__(self, x: np.ndarray) -> None:
        self.mean = x.mean(0)
        self.std = np.where(x.std(0) > 1e-9, x.std(0), 1.0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


# -------------------------------------------------------------------------------------------------------- metrics


class Metrics:
    """Weighted metrics (weights = bootstrap multiplicities; all ones for the point estimate)."""

    @staticmethod
    def auroc(scores: np.ndarray, pos: np.ndarray, neg: np.ndarray, w: np.ndarray | None = None) -> float | None:
        """AUROC of rows `pos` over rows `neg` (ties count half)."""
        if len(pos) == 0 or len(neg) == 0:
            return None
        return AurocSpec(scores, pos, neg)(w)

    @staticmethod
    def confusion(true: np.ndarray, pred: np.ndarray, n: int, w: np.ndarray | None = None) -> np.ndarray:
        """(n, n) weighted counts, rows = true class."""
        return np.bincount(true * n + pred, weights=w, minlength=n * n).reshape(n, n)

    @staticmethod
    def f1_from_confusion(conf: np.ndarray) -> np.ndarray:
        """Per-class F1 (0 where undefined)."""
        tp = np.diag(conf)
        precision = np.divide(tp, conf.sum(0), out=np.zeros_like(tp, dtype=float), where=conf.sum(0) > 0)
        recall = np.divide(tp, conf.sum(1), out=np.zeros_like(tp, dtype=float), where=conf.sum(1) > 0)
        denom = precision + recall
        return np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)

    @staticmethod
    def recall(conf: np.ndarray) -> list[float | None]:
        rows = conf.sum(1)
        return [float(conf[i, i] / rows[i]) if rows[i] > 0 else None for i in range(len(conf))]


class AurocSpec:
    """AUROC of a fixed (score, positive rows, negative rows) triple, fast under resampling weights."""

    def __init__(self, scores: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> None:
        self.pos, self.neg = np.asarray(pos), np.asarray(neg)
        values = np.concatenate([scores[self.pos], scores[self.neg]]).astype(np.float64)
        _, inverse = np.unique(values, return_inverse=True)
        self.n_unique = int(inverse.max()) + 1 if len(inverse) else 0
        self.inv_pos, self.inv_neg = inverse[: len(self.pos)], inverse[len(self.pos) :]

    def __call__(self, w: np.ndarray | None = None) -> float | None:
        if len(self.pos) == 0 or len(self.neg) == 0:
            return None
        wp = None if w is None else w[self.pos]
        wn = None if w is None else w[self.neg]
        pw = np.bincount(self.inv_pos, weights=wp, minlength=self.n_unique)
        nw = np.bincount(self.inv_neg, weights=wn, minlength=self.n_unique)
        total = pw.sum() * nw.sum()
        if total <= 0:
            return None
        below = np.cumsum(nw) - nw
        return float((pw * (below + 0.5 * nw)).sum() / total)


class VehicleBootstrap:
    """Resamples sender vehicles with replacement; a window's weight = how often its vehicle was drawn."""

    def __init__(self, vehicles: Sequence[str], n: int, seed: int) -> None:
        _, self.codes = np.unique(np.asarray(vehicles), return_inverse=True)
        self.n_vehicles = int(self.codes.max()) + 1 if len(self.codes) else 0
        self.n, self.seed = n, seed

    def weights(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n):
            drawn = np.bincount(rng.integers(0, self.n_vehicles, self.n_vehicles), minlength=self.n_vehicles)
            yield drawn[self.codes].astype(np.float64)


def summarise(values: Sequence[float | None]) -> dict[str, Any]:
    """mean / std (ddof 1) / per-seed values; None when any value is undefined."""
    vals = list(values)
    if not vals or any(v is None for v in vals):
        return {"mean": None, "std": None, "values": vals}
    arr = np.asarray(vals, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, "values": vals}


# ----------------------------------------------------------------------------------------------------- evaluation


@dataclasses.dataclass
class Prediction:
    """Test outputs of one (representation, probe, seed)."""

    score: np.ndarray  # attack score (P(attack), attack vote share or anomaly score)
    pred_binary: np.ndarray | None  # bool
    pred_class: np.ndarray | None  # int index into CLASSES
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


class ResultWriter:
    """Writes JSON files under the allowed root only (never data/, src/data/, src/runs/pretraining/)."""

    def __init__(self, out_dir: pathlib.Path, allowed_root: pathlib.Path) -> None:
        out, root = absolute(out_dir).resolve(), absolute(allowed_root).resolve()
        if out != root and root not in out.parents:
            raise EvalError(f"refusing to write {out}: outside {root}")
        for name in WRITE_GUARDS:
            guard = (REPO_ROOT / name).resolve()
            if out == guard or guard in out.parents:
                raise EvalError(f"refusing to write under protected {guard}")
        self.out = out

    def write(self, name: str, obj: dict[str, Any]) -> pathlib.Path:
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / name
        path.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")
        return path


class FrozenEncoderEvaluation:
    """Loads the run, builds the fit / test / bank sets, predicts every test window, then opens the test labels."""

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg
        self.writer = ResultWriter(cfg.out_dir, cfg.allowed_out_root)
        self.run = exp.TrainedRun(absolute(cfg.run_dir), cfg.checkpoint, None)
        self.device = exp.pick_device(cfg.device)
        self.timings: dict[str, float] = {}

    def _tick(self, name: str, t0: float) -> float:
        self.timings[name] = round(time.time() - t0, 2)
        _LOG.info("%s done in %.1f s", name, self.timings[name])
        return time.time()

    def read_test(self) -> tuple[exp.WindowSet, list[pathlib.Path]]:
        seq_len = self.run.settings.seq_len
        paths = CompactWindowReader(seq_len).find(self.run.data_dir, "test")
        if not paths:
            raise EvalError(f"no test shards under {self.run.data_dir}")
        parts = exp.ShardPool(self.cfg.workers).map(
            exp.GroupTestReader(seq_len, self.cfg.group), [str(p) for p in paths]
        )
        infos = [p.with_name(p.name.replace(".json.gz", "_info.json.gz")) for p in paths]
        return exp.WindowSet.from_parts(parts, seq_len), infos

    def anomaly_sets(self, seed: int) -> tuple[exp.WindowSet, exp.WindowSet, dict[str, Any]]:
        """Unlabeled bank + calibration windows (the exporter's TrainSampler; no label file opened)."""
        dcfg = exp.DetectionConfig(
            seed=seed,
            bank=self.cfg.bank,
            calib=self.cfg.calib,
            k=self.cfg.anomaly_k,
            workers=self.cfg.workers,
            candidate_rate=self.cfg.bank_candidate_rate,
        )
        return exp.TrainSampler(self.run.data_dir, self.run.settings, dcfg).sample()

    def featurisers(self, seed: int) -> dict[str, Any]:
        """Representation name -> featuriser for this seed (random-init weights depend on the seed)."""
        torch.manual_seed(seed)
        random_encoder = TimesNetEncoder(self.run.settings)
        return {
            "model-all": self._model_feats,
            "random-init": EncoderFeatures(self.run, random_encoder, self.device, self.cfg.batch_size),
            "length": LengthFeatures(),
            "stats": StatsFeatures(self.run),
        }

    def run_all(self) -> dict[str, Any]:
        cfg, t_start = self.cfg, time.time()
        t0 = t_start
        fit = ProbeFitSampler(self.run.data_dir, self.run.settings.seq_len, cfg).sample()
        t0 = self._tick("read_fit_sample", t0)
        test, info_paths = self.read_test()
        t0 = self._tick("read_test", t0)
        anomaly = {s: self.anomaly_sets(s) for s in cfg.seeds}
        t0 = self._tick("read_anomaly_bank", t0)
        _LOG.info(
            "fit union %d, test %s %d, bank %d x %d seeds",
            len(fit.windows),
            cfg.group,
            len(test),
            cfg.bank,
            len(cfg.seeds),
        )

        self._model_feats = EncoderFeatures(self.run, self.run.encoder, self.device, cfg.batch_size)
        model_cache = {
            "fit": self._model_feats(fit.windows),
            "test": self._model_feats(test),
        }
        t0 = self._tick("embed_model_all_fit_test", t0)
        y_fit_all = np.array([CLASSES.index(c) for c in fit.classes])
        preds: dict[tuple[str, str, int], Prediction] = {}
        probe_info: dict[str, Any] = {}
        for seed in cfg.seeds:
            rows = fit.rows[seed]
            fit_set = fit.windows.take(rows)
            y5 = y_fit_all[rows]
            yb = (y5 > 0).astype(int)
            bank, calib, bank_report = anomaly[seed]
            for name, feat in self.featurisers(seed).items():
                ts = time.time()
                if name == "model-all":
                    x_fit, x_test = model_cache["fit"][rows], model_cache["test"]
                else:
                    x_fit, x_test = feat(fit_set), feat(test)
                x_bank, x_calib = feat(bank), feat(calib)
                preds.update(self._probe(name, seed, feat.metric, x_fit, y5, yb, x_test, probe_info))
                preds[(name, "anomaly", seed)] = self._anomaly(name, feat.metric, x_bank, x_calib, x_test, bank_report)
                _LOG.info("seed %d %s: %.1f s", seed, name, time.time() - ts)
        t0 = self._tick("probes_and_anomaly", t0)

        # Every test window now has every prediction; only now are the test labels opened.
        guard = preds[("model-all", "anomaly", cfg.seeds[0])].score
        labels = exp.LabelVault(info_paths).reveal(test.ids, guard)
        y_test = np.array([CLASSES.index(r["class"]) for r in labels])
        t0 = self._tick("reveal_test_labels", t0)

        lengths = test.lengths
        vehicles = [FeatureShards.vehicle_of(i) for i in test.ids]
        results = self._metrics(preds, y_test, lengths)
        t0 = self._tick("metrics", t0)
        boot = self._bootstrap(preds, y_test, lengths, vehicles)
        t0 = self._tick("bootstrap", t0)
        data_report = self._data_report(fit, y_test, lengths, vehicles)
        self.timings["total"] = round(time.time() - t_start, 2)
        out = {
            "schema": 1,
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "scope": {
                "test_split": "test",
                "group": cfg.group,
                "note": "0709 excluded until F14 is decided; labels used only for probe fitting and metrics",
            },
            "data": data_report,
            "probes": probe_info,
            "metrics": results,
            "bootstrap": boot,
            "timings_s": self.timings,
        }
        self.writer.write("results.json", out)
        self.writer.write("config.json", self._config())
        self.writer.write("env.json", self._env())
        self.writer.write("report_data.json", ReportData.build(out, cfg))
        _LOG.info("wrote %s", self.writer.out)
        return out

    def _probe(
        self,
        name: str,
        seed: int,
        metric: str,
        x_fit: np.ndarray,
        y5: np.ndarray,
        yb: np.ndarray,
        x_test: np.ndarray,
        info: dict[str, Any],
    ) -> dict[tuple[str, str, int], Prediction]:
        cfg, k = self.cfg, len(CLASSES)
        lr_b = LogisticProbe(cfg.lr_c, cfg.lr_max_iter, seed).fit(x_fit, yb)
        lr_5 = LogisticProbe(cfg.lr_c, cfg.lr_max_iter, seed).fit(x_fit, y5)
        p_b = lr_b.proba(x_test, 2)[:, 1]
        p_5 = lr_5.proba(x_test, k)
        info[f"{name}/seed{seed}"] = {
            "fit_windows": int(len(x_fit)),
            "dim": int(x_fit.shape[1]),
            "lr_binary_converged": lr_b.converged,
            "lr_5class_converged": lr_5.converged,
        }
        if name in ("model-all", "random-init"):
            kx_fit, kx_test = x_fit, x_test  # z is already L2-normalised: cosine as is
        else:
            scale = Standardiser(x_fit)
            kx_fit, kx_test = scale(x_fit), scale(x_test)
        votes = KnnProbe(cfg.knn_k, metric, self.device, cfg.knn_batch).fit(kx_fit, y5).votes(kx_test, k)
        attack_share = 1.0 - votes[:, 0]
        return {
            (name, "lr", seed): Prediction(p_b, p_b > 0.5, p_5.argmax(1)),
            (name, "knn", seed): Prediction(attack_share, attack_share > 0.5, votes.argmax(1)),
        }

    def _anomaly(
        self,
        name: str,
        metric: str,
        x_bank: np.ndarray,
        x_calib: np.ndarray,
        x_test: np.ndarray,
        bank_report: dict[str, Any],
    ) -> Prediction:
        """Label-free kNN anomaly score; for the 1-d length feature the score is -n (no bank needed)."""
        if name == "length":
            calib = -x_calib[:, 0].astype(np.float64)
            th = exp.Threshold(calib, self.cfg.quantile)
            score = -x_test[:, 0].astype(np.float64)
            return Prediction(score, th.flags(score), None, {"theta": th.theta, "rule": "score = -n"})
        if name == "stats":
            scale = Standardiser(x_bank)
            x_bank, x_calib, x_test = scale(x_bank), scale(x_calib), scale(x_test)
        norm = torch.nn.functional.normalize
        bank = norm(torch.from_numpy(np.ascontiguousarray(x_bank)), dim=1)
        scorer = exp.KnnScorer(bank, self.cfg.anomaly_k, self.device)
        th = exp.Threshold(scorer(norm(torch.from_numpy(np.ascontiguousarray(x_calib)), dim=1)), self.cfg.quantile)
        score = scorer(norm(torch.from_numpy(np.ascontiguousarray(x_test)), dim=1))
        return Prediction(score, th.flags(score), None, {"theta": th.theta, "bank_report": bank_report})

    @staticmethod
    def strata_rows(lengths: np.ndarray) -> dict[str, np.ndarray]:
        return {s: (lengths >= lo) & (lengths <= hi) for s, (lo, hi) in STRATA.items()}

    def _metrics(self, preds: dict, y: np.ndarray, lengths: np.ndarray) -> dict[str, Any]:
        """Point metrics per (representation, probe): per seed, then mean / std."""
        strata = self.strata_rows(lengths)
        out: dict[str, Any] = {}
        for rep in REPRESENTATIONS:
            out[rep] = {}
            for probe in ("lr", "knn", "anomaly"):
                per_seed = [self._one(preds[(rep, probe, s)], y, strata) for s in self.cfg.seeds]
                out[rep][probe] = {"per_seed": per_seed, "summary": self._aggregate(per_seed)}
        return out

    @staticmethod
    def _one(p: Prediction, y: np.ndarray, strata: dict[str, np.ndarray]) -> dict[str, Any]:
        k = len(CLASSES)
        attack, benign = y > 0, y == 0
        res: dict[str, Any] = {
            "binary_auroc": Metrics.auroc(p.score, np.flatnonzero(attack), np.flatnonzero(benign)),
        }
        if p.pred_binary is not None:
            conf_b = Metrics.confusion(attack.astype(int), p.pred_binary.astype(int), 2)
            f1 = Metrics.f1_from_confusion(conf_b)
            res.update(
                {
                    "binary_f1": float(f1[1]),
                    "binary_tpr": Metrics.recall(conf_b)[1],
                    "binary_fpr": float(conf_b[0, 1] / conf_b[0].sum()) if conf_b[0].sum() else None,
                    "binary_confusion": conf_b.astype(int).tolist(),
                }
            )
        if p.pred_class is not None:
            conf = Metrics.confusion(y, p.pred_class, k)
            res.update(
                {
                    "macro_f1": float(Metrics.f1_from_confusion(conf).mean()),
                    "recall": dict(zip(CLASSES, Metrics.recall(conf))),
                    "confusion": conf.astype(int).tolist(),
                }
            )
        if "theta" in p.extra:
            res["theta"] = p.extra["theta"]
        by_stratum: dict[str, Any] = {}
        for sname, srows in strata.items():
            entry: dict[str, Any] = {}
            neg = np.flatnonzero(benign & srows)
            for ci, cname in enumerate(CLASSES):
                rows = (y == ci) & srows
                e: dict[str, Any] = {"n": int(rows.sum())}
                if ci > 0:
                    e["auroc_vs_benign"] = Metrics.auroc(p.score, np.flatnonzero(rows), neg)
                if p.pred_binary is not None:
                    e["flagged_rate"] = float(p.pred_binary[rows].mean()) if rows.any() else None
                if p.pred_class is not None:
                    e["recall_5class"] = float((p.pred_class[rows] == ci).mean()) if rows.any() else None
                entry[cname] = e
            by_stratum[sname] = entry
        res["by_length"] = by_stratum
        return res

    @staticmethod
    def _aggregate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
        first = per_seed[0]
        out: dict[str, Any] = {}
        for key in ("binary_auroc", "binary_f1", "binary_tpr", "binary_fpr", "macro_f1", "theta"):
            if key in first:
                out[key] = summarise([r[key] for r in per_seed])
        if "recall" in first:
            out["recall"] = {c: summarise([r["recall"][c] for r in per_seed]) for c in CLASSES}
            conf = np.mean([np.asarray(r["confusion"], float) for r in per_seed], axis=0)
            rows = conf.sum(1, keepdims=True)
            out["confusion_row_normalised_mean"] = np.divide(
                conf, rows, out=np.zeros_like(conf), where=rows > 0
            ).tolist()
        out["by_length"] = {}
        for sname in STRATA:
            out["by_length"][sname] = {}
            for cname in CLASSES:
                e = first["by_length"][sname][cname]
                agg: dict[str, Any] = {"n": e["n"]}
                for key in ("auroc_vs_benign", "flagged_rate", "recall_5class"):
                    if key in e:
                        agg[key] = summarise([r["by_length"][sname][cname][key] for r in per_seed])
                out["by_length"][sname][cname] = agg
        return out

    def _bootstrap(self, preds: dict, y: np.ndarray, lengths: np.ndarray, vehicles: list[str]) -> dict[str, Any]:
        """95% CI of the seed-mean of the main numbers, resampling sender vehicles (paired across representations)."""
        cfg, k = self.cfg, len(CLASSES)
        strata = self.strata_rows(lengths)
        benign = y == 0
        specs: dict[tuple[str, str, str], list[Any]] = {}
        for rep in REPRESENTATIONS:
            for probe in ("lr", "knn", "anomaly"):
                for seed in cfg.seeds:
                    p = preds[(rep, probe, seed)]
                    add = specs.setdefault
                    add((rep, probe, "binary_auroc"), []).append(
                        ("auc", AurocSpec(p.score, np.flatnonzero(y > 0), np.flatnonzero(benign)))
                    )
                    for sname in ("all", "n<=3", "n>=4"):
                        neg = np.flatnonzero(benign & strata[sname])
                        for ci, fam in enumerate(FAMILIES, start=1):
                            pos = np.flatnonzero((y == ci) & strata[sname])
                            add((rep, probe, f"auroc[{sname}][{fam}]"), []).append(
                                ("auc", AurocSpec(p.score, pos, neg))
                            )
                    if p.pred_binary is not None:
                        add((rep, probe, "binary_f1"), []).append(
                            ("f1", (y > 0).astype(int), p.pred_binary.astype(int))
                        )
                    if p.pred_class is not None:
                        add((rep, probe, "macro_f1"), []).append(("mf1", y, p.pred_class))
        draws: dict[tuple[str, str, str], list[float]] = {key: [] for key in specs}
        for w in VehicleBootstrap(vehicles, cfg.bootstrap, cfg.bootstrap_seed).weights():
            for key, items in specs.items():
                vals = []
                for item in items:
                    if item[0] == "auc":
                        vals.append(item[1](w))
                    elif item[0] == "f1":
                        vals.append(float(Metrics.f1_from_confusion(Metrics.confusion(item[1], item[2], 2, w))[1]))
                    else:
                        vals.append(float(Metrics.f1_from_confusion(Metrics.confusion(item[1], item[2], k, w)).mean()))
                if all(v is not None for v in vals):
                    draws[key].append(float(np.mean(vals)))
        out: dict[str, Any] = {
            "method": (
                f"{cfg.bootstrap} resamples of sender vehicles with replacement (group:sender from the window id), "
                "same draws for every representation; statistic = mean over seeds; percentile 95% interval"
            ),
            "n_vehicles": int(len(set(vehicles))),
            "ci": {},
        }
        for (rep, probe, metric), vals in draws.items():
            entry = (
                {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)), "n": len(vals)}
                if vals
                else {"lo": None, "hi": None, "n": 0}
            )
            out["ci"].setdefault(rep, {}).setdefault(probe, {})[metric] = entry
        return out

    def _data_report(self, fit: FitSet, y: np.ndarray, lengths: np.ndarray, vehicles: list[str]) -> dict[str, Any]:
        seq_len = self.run.settings.seq_len
        fit_len = fit.windows.lengths

        def hist(values: np.ndarray) -> list[int]:
            return np.bincount(values, minlength=seq_len + 1)[1:].astype(int).tolist()

        fit_report = {}
        for seed, rows in fit.rows.items():
            cls, grp, ln = fit.classes[rows], fit.groups[rows], fit_len[rows]
            fit_report[f"seed{seed}"] = {
                c: {
                    "n": int((cls == c).sum()),
                    "by_group": {g: int(((cls == c) & (grp == g)).sum()) for g in sorted(set(grp.tolist()))},
                    "share_n_le_3": float((ln[cls == c] <= 3).mean()) if (cls == c).any() else None,
                    "length_hist": hist(ln[cls == c]),
                }
                for c in CLASSES
            }
        test_report = {}
        for ci, c in enumerate(CLASSES):
            rows = y == ci
            test_report[c] = {
                "n": int(rows.sum()),
                "vehicles": int(len({vehicles[i] for i in np.flatnonzero(rows)})),
                "n_le_3": int((lengths[rows] <= 3).sum()),
                "share_n_le_3": float((lengths[rows] <= 3).mean()) if rows.any() else None,
                "mean_rows": float(lengths[rows].mean()) if rows.any() else None,
                "length_hist": hist(lengths[rows]),
            }
        return {
            "fit": fit_report,
            "fit_union_windows": int(len(fit.windows)),
            "test": test_report,
            "test_windows": int(len(y)),
            "length_hist_bins": list(range(1, seq_len + 1)),
        }

    def _config(self) -> dict[str, Any]:
        s = self.run.settings
        return {
            "evaluation": self.cfg.to_json(),
            "classes": list(CLASSES),
            "representations": {
                "model-all": "frozen encoder of the run, z = masked mean pool, L2-normalised",
                "random-init": "TimesNetEncoder(settings) after torch.manual_seed(seed), eval mode, same scaling",
                "length": "n real rows",
                "stats": "mean / std / min / max of the 13 scaled features over real rows + n (53 columns)",
            },
            "probes": {
                "lr": "StandardScaler + LogisticRegression(lbfgs, class_weight balanced); binary and 5-class fitted "
                "separately; binary decision at P(attack) > 0.5",
                "knn": "k nearest probe-fit windows by cosine (encoder z as is; stats standardised first) or Euclidean "
                "(length, standardised); attack score = 1 - Benign vote share; decision at > 0.5; 5-class = most "
                "votes (ties -> lower class index); not class-weighted",
                "anomaly": "label-free: 1 - mean cosine similarity to the k nearest of an unlabeled train bank "
                "(trained-on vehicles); θ = quantile of unlabeled check-set-vehicle scores; length: score = -n",
            },
            "strata": {k: list(v) for k, v in STRATA.items()},
            "run": {
                "run_dir": str(self.cfg.run_dir),
                "checkpoint": self.cfg.checkpoint,
                "sha256": self.run.sha256,
                "epoch": self.run.epoch,
                "best_val_joint": self.run.best_val_joint,
                "dataset": s.dataset,
                "seq_len": s.seq_len,
                "d_model": s.d_model,
                "d_ff": s.d_ff,
                "pretrain_seed": s.seed,
            },
            "data_dir": (
                str(self.run.data_dir.relative_to(REPO_ROOT))
                if REPO_ROOT in self.run.data_dir.parents
                else str(self.run.data_dir)
            ),
        }

    def _env(self) -> dict[str, Any]:
        return {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "git_commit": exp.git_short_sha(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
            "device": str(self.device),
            "argv": sys.argv,
            "timings_s": self.timings,
        }


class ReportData:
    """Compact tables for the report page, taken verbatim from results.json."""

    @staticmethod
    def build(results: dict[str, Any], cfg: EvalConfig) -> dict[str, Any]:
        metrics, ci = results["metrics"], results["bootstrap"]["ci"]

        def cell(rep: str, probe: str, key: str, ci_key: str | None = None) -> dict[str, Any]:
            summ = metrics[rep][probe]["summary"].get(key, {"mean": None, "std": None})
            return {"mean": summ["mean"], "std": summ["std"], "ci95": ci[rep][probe].get(ci_key or key)}

        main = []
        for rep in REPRESENTATIONS:
            row: dict[str, Any] = {"representation": rep}
            for probe in ("lr", "knn"):
                row[f"{probe}_binary_auroc"] = cell(rep, probe, "binary_auroc")
                row[f"{probe}_binary_f1"] = cell(rep, probe, "binary_f1")
                row[f"{probe}_macro_f1"] = cell(rep, probe, "macro_f1")
            row["anomaly_binary_auroc"] = cell(rep, "anomaly", "binary_auroc")
            main.append(row)
        per_family = []
        for rep in REPRESENTATIONS:
            for probe in ("lr", "knn", "anomaly"):
                for fam in FAMILIES:
                    entry: dict[str, Any] = {"representation": rep, "probe": probe, "family": fam}
                    for sname in STRATA:
                        e = metrics[rep][probe]["summary"]["by_length"][sname][fam]
                        a = e["auroc_vs_benign"]
                        entry[sname] = {
                            "n": e["n"],
                            "auroc_mean": a["mean"],
                            "auroc_std": a["std"],
                            "auroc_ci95": ci[rep][probe].get(f"auroc[{sname}][{fam}]"),
                            "recall_5class_mean": e.get("recall_5class", {}).get("mean"),
                            "flagged_rate_mean": e.get("flagged_rate", {}).get("mean"),
                        }
                    per_family.append(entry)
        confusion = {
            rep: {probe: metrics[rep][probe]["summary"].get("confusion_row_normalised_mean") for probe in ("lr", "knn")}
            for rep in REPRESENTATIONS
        }
        benign_rates = {
            rep: {
                probe: {
                    s: metrics[rep][probe]["summary"]["by_length"][s]["Benign"].get("flagged_rate", {}).get("mean")
                    for s in STRATA
                }
                for probe in ("lr", "knn", "anomaly")
            }
            for rep in REPRESENTATIONS
        }
        return {
            "schema": 1,
            "created": results["created"],
            "scope": results["scope"],
            "classes": list(CLASSES),
            "seeds": list(cfg.seeds),
            "main_table": main,
            "per_family": per_family,
            "confusion_row_normalised_mean": confusion,
            "benign_flagged_rate": benign_rates,
            "data": results["data"],
            "timings_s": results["timings_s"],
            "notes": [
                "mean ± std over seeds (ddof 1); ci95 = vehicle-bootstrap percentile interval of the seed mean",
                "per-family AUROC = the family's windows vs the benign windows of the same length stratum",
                "the probe-fit set is class-balanced; test 1416 is not (DataReplay / DoS dominate), so F1 depends on "
                "the prevalence",
                "random-init uses torch seed = evaluation seed; model-all is one pretraining run (seed 0)",
            ],
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    d = EvalConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--run", type=pathlib.Path, default=d.run_dir)
    p.add_argument("--checkpoint", default=d.checkpoint)
    p.add_argument("--out", type=pathlib.Path, default=d.out_dir)
    p.add_argument("--per-class", type=int, default=d.per_class)
    p.add_argument("--seeds", type=int, nargs="+", default=list(d.seeds))
    p.add_argument("--bootstrap", type=int, default=d.bootstrap)
    p.add_argument("--bank", type=int, default=d.bank)
    p.add_argument("--calib", type=int, default=d.calib)
    p.add_argument("--device", default=d.device)
    p.add_argument("--workers", type=int, default=d.workers)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args(argv)
    cfg = EvalConfig(
        run_dir=a.run,
        checkpoint=a.checkpoint,
        out_dir=a.out,
        per_class=a.per_class,
        seeds=tuple(a.seeds),
        bootstrap=a.bootstrap,
        bank=a.bank,
        calib=a.calib,
        device=a.device,
        workers=a.workers,
    )
    FrozenEncoderEvaluation(cfg).run_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
