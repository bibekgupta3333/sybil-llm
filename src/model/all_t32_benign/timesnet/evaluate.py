"""One-class evaluation of a pretrained TimesNet on `all/T32_benign` (WBS S1.4, `src/model/all_t32_benign/PLAN.md`).

Loads a run's checkpoint, rebuilds `PretrainingModel` (`heads.py` + `encoder.py`, this package's own copies),
restores the run's `FeatureSpace` and re-scales every split identically (stored train z-scores -> the trained
scaling). The frozen encoder (eval mode) gives per window: z (masked mean of H) and the masked-reconstruction error
(seeded masks, averaged over `--recon-passes` passes, error on hidden real rows only). Scores: reconstruction error,
Mahalanobis on z (fitted on benign train z), kNN cosine distance to a benign train memory bank (K = 5, 10, 20).

Label discipline (RULE 1): every score file `scores/<split>.npz` is written before any `_info` shard is opened;
`LabelReader` refuses to start while one is missing and the summary records both time stamps. Labels are then
aligned by window id (every id is also rebuilt from its info fields; any mismatch raises). Thresholds come from
benign `val` only; each `test_<attack>` is compared with benign `test`. `--random-init` runs the same pipeline with an
untrained, seed-fixed model (the no-pretraining control); the scaling is still the run's.

Reads the run folder and `src/data/encoder_input/all/T32_benign` (via the package's `data.py` contract); writes
`<out>/scores/*.npz`, `<out>/fit/*`, `<out>/summary.json`, `<out>/summary.md`.

Usage:
    .venv-train/bin/python -m src.model.all_t32_benign.timesnet.evaluate --run <run_dir> [--checkpoint best.pt]
        [--cache-dir DIR] [--data-dir DIR] [--out DIR] [--device auto] [--batch-size 1024] [--recon-passes 4]
        [--seed 0] [--smoke] [--random-init]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import importlib
import inspect
import json
import logging
import platform
import sys
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch

from src.model.all_t32_benign.timesnet import scoring
from src.model.all_t32_benign.timesnet.base_config import PretrainConfig
from src.model.all_t32_benign.timesnet.base_data import FeatureSpace, file_sha256
from src.model.all_t32_benign.timesnet.heads import PretrainingModel
from src.model.all_t32_benign.timesnet.views import ReconMasker

_LOG = logging.getLogger(__name__)

BENIGN_SPLITS = ("val", "test")
ATTACK_CODES = {"test_gridsybil": 16, "test_datareplay": 17, "test_dosrandom": 18, "test_dosdisruptive": 19}
SCORED_SPLITS = (*BENIGN_SPLITS, *ATTACK_CODES)
DEFAULT_DATA_DIR = "src/data/encoder_input/all/T32_benign"
DEFAULT_CACHE_DIR = "src/data/cache/all/T32_benign"
DEFAULT_EVAL_ROOT = "src/runs/evaluation/all/T32_benign"


class MetadataMismatchError(ValueError):
    """The encoder input metadata.json differs from the one the run (or the cache) was built from."""


class AlignmentError(ValueError):
    """Score ids and label shards disagree (order, id fields, lengths or labels)."""


@dataclasses.dataclass(frozen=True)
class EvalSettings:
    """Every knob of one evaluation; saved in summary.json.

    Attributes:
        run_dir: pretraining run folder (config.json, env.json, checkpoints).
        checkpoint: checkpoint file name inside run_dir.
        data_dir: encoder input folder (metadata.json, <split>/b32/ shards).
        cache_dir: decoded feature cache of the package's data.py.
        out_dir: output folder (must not exist).
        device: torch device name or "auto".
        batch_size: windows per encoder batch.
        recon_passes: seeded masking passes averaged per window.
        seed: seed of masks, kNN bank subsample and the random-init control.
        smoke: plumbing check on a few windows per split (numbers are not results).
        random_init: untrained, seed-fixed model instead of the checkpoint's weights.
        k_values: K of the kNN score.
        max_bank: largest kNN memory bank (train windows, seeded subsample).
        quantiles: benign val quantiles used as alarm thresholds.
        smoke_windows: windows per split in smoke mode.
        allow_metadata_mismatch: continue when metadata.json differs from the run's or the cache's (off by default).
        val_attack_codes: attack codes allowed in the val split besides 0 (contamination experiments with a mixed,
            unlabelled-in-use val, e.g. `all/T32_mix10_*`; the alarm thresholds then come from that mixed val).
            Empty (default) = val must be benign only, as for `all/T32_benign`.
    """

    run_dir: str
    checkpoint: str = "best.pt"
    data_dir: str = DEFAULT_DATA_DIR
    cache_dir: str = DEFAULT_CACHE_DIR
    out_dir: str = ""
    device: str = "auto"
    batch_size: int = 1024
    recon_passes: int = 4
    seed: int = 0
    smoke: bool = False
    random_init: bool = False
    k_values: tuple[int, ...] = (5, 10, 20)
    max_bank: int = 200_000
    quantiles: tuple[float, ...] = (0.95, 0.99)
    smoke_windows: int = 2048
    allow_metadata_mismatch: bool = False
    val_attack_codes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.recon_passes < 1:
            raise ValueError(f"recon_passes must be >= 1, got {self.recon_passes}")
        if not self.out_dir:
            raise ValueError("out_dir must be set")
        if any(not 0.0 < q < 1.0 for q in self.quantiles):
            raise ValueError(f"quantiles must be in (0, 1), got {self.quantiles}")


class DataAccess:
    """The seam to the package's `data.py` (agent A's contract): split loading, scaling restore, firewall.

    Attributes:
        load_split: `(cache_dir, data_dir, split, firewall, ...) -> object with .x (N, T, 13), .mask (N, T), .ids`.
        restore_space: `(metadata, state) -> FeatureSpace` of the trained scaling.
        make_firewall: `() -> firewall` that allows feature arrays of any split, never labels.
    """

    MODULE = "src.model.all_t32_benign.timesnet.data"

    def __init__(
        self, load_split: Callable[..., Any], restore_space: Callable[..., Any], make_firewall: Callable[[], Any]
    ):
        self.load_split, self.restore_space, self.make_firewall = load_split, restore_space, make_firewall

    @classmethod
    def from_package(cls) -> "DataAccess":
        """Imports the contract from the package's data.py."""
        module = importlib.import_module(cls.MODULE)
        return cls(module.load_split, module.restore_space, module.EvalFirewall)

    def load(self, settings: EvalSettings, split: str, firewall: Any) -> Any:
        """One split's feature arrays (one shard per split in smoke mode when the loader supports it)."""
        kwargs: dict[str, Any] = {}
        if settings.smoke and "max_shards" in inspect.signature(self.load_split).parameters:
            kwargs["max_shards"] = 1
        return self.load_split(Path(settings.cache_dir), Path(settings.data_dir), split, firewall, **kwargs)


class Rescaler:
    """Stored train z-scores -> the scaling the model was trained in (`FeatureSpace.rescale`, per batch)."""

    def __init__(self, stored: FeatureSpace, trained: FeatureSpace) -> None:
        self.stored, self.trained = stored, trained

    def to(self, device: torch.device) -> "Rescaler":
        """Moves both scalings to `device`."""
        self.stored.to(device)
        self.trained.to(device)
        return self

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, T, 13) stored scaling + (B, T) mask -> (B, T, 13) trained scaling, padding rows exactly 0."""
        return self.trained.norm(self.stored.raw(x), mask)


class FrozenScorer:
    """Frozen encoder (eval mode, no grad): z per window and the seeded masked-reconstruction error.

    Windows are processed sorted by real-row count (fewer period groups per TimesBlock) and returned in input order.
    A window with fewer than 2 real rows has nothing to hide (as in training); its error is NaN and it is flagged.
    """

    def __init__(
        self,
        model: PretrainingModel,
        masker: ReconMasker,
        rescaler: Rescaler,
        settings: EvalSettings,
        device: torch.device,
    ) -> None:
        self.model, self.masker, self.rescaler, self.s, self.device = model.eval(), masker, rescaler, settings, device

    def _gen(self, split: str, pass_index: int, batch_index: int) -> torch.Generator:
        seq = np.random.SeedSequence([self.s.seed, zlib.crc32(split.encode()), pass_index, batch_index])
        return torch.Generator().manual_seed(int(seq.generate_state(1)[0]))

    @torch.no_grad()
    def run(self, split: str, x: torch.Tensor, mask: torch.Tensor, recon: bool) -> dict[str, np.ndarray]:
        """x (N, T, 13) stored scaling + mask (N, T) -> {"z": (N, d) float32, "recon": (N,) float64}.

        `recon` is NaN where no row could be hidden (< 2 real rows) or when `recon` is False.
        """
        n = len(x)
        lengths = mask.sum(1)
        order = torch.argsort(lengths, stable=True)
        z_parts: list[np.ndarray] = []
        err = np.full(n, np.nan, dtype=np.float64)
        for b, start in enumerate(range(0, n, self.s.batch_size)):
            rows = order[start : start + self.s.batch_size]
            m = torch.as_tensor(mask[rows]).to(self.device)
            xb = self.rescaler(torch.as_tensor(x[rows]).to(self.device, torch.float32), m)
            _, zb = self.model.encoder(xb, m)
            z_parts.append(zb.float().cpu().numpy())
            if recon:
                err[rows.numpy()] = self._recon_error(split, b, xb, m)
        width = z_parts[0].shape[1] if z_parts else int(self.model.recon_head.net[0].in_features)
        z = np.empty((n, width), dtype=np.float32)
        if z_parts:
            z[order.numpy()] = np.concatenate(z_parts)
        return {"z": z, "recon": err}

    def _recon_error(self, split: str, batch_index: int, x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        """(B,) mean over passes of the masked MSE on hidden real rows (NaN where nothing can be hidden)."""
        total = torch.zeros(len(x), dtype=torch.float64)
        counted = torch.zeros(len(x), dtype=torch.float64)
        for p in range(self.s.recon_passes):
            hidden = self.masker(mask, self._gen(split, p, batch_index))
            h, _ = self.model.encoder(self.model.hide(x, hidden), mask)
            pred = self.model.recon_head(h)
            rows = hidden.unsqueeze(-1).to(pred.dtype)
            sse = (((pred - x) ** 2) * rows).sum((1, 2)).cpu().double()
            n_hidden = hidden.sum(1).cpu().double()
            ok = n_hidden > 0
            total[ok] += sse[ok] / (n_hidden[ok] * x.shape[-1])
            counted[ok] += 1
        out = torch.full((len(x),), float("nan"), dtype=torch.float64)
        ok = counted > 0
        out[ok] = total[ok] / counted[ok]
        return out.numpy()


class LabelReader:
    """Reads `_info` label shards only after every score file exists; aligns them to the score ids."""

    def __init__(
        self, data_dir: Path, scores_dir: Path, splits: Sequence[str], val_attack_codes: Sequence[int] = ()
    ) -> None:
        self.data_dir, self.scores_dir, self.splits = data_dir, scores_dir, tuple(splits)
        self.val_attack_codes = frozenset(int(c) for c in val_attack_codes)
        self.opened: list[str] = []

    def _guard(self) -> None:
        missing = [s for s in self.splits if not (self.scores_dir / f"{s}.npz").is_file()]
        if missing:
            raise RuntimeError(f"labels requested before the score files of {missing} were written")

    @staticmethod
    def rebuild_id(info: dict[str, Any]) -> str:
        """`<receiver_file>#<sender_pseudo>-<sender>#<link_window_index>` from the info fields."""
        return f"{info['receiver_file']}#{info['sender_pseudo']}-{info['sender']}#{info['link_window_index']}"

    def read(self, split: str, ids: np.ndarray, n_real_rows: np.ndarray, complete: bool) -> tuple[np.ndarray, bool]:
        """Labels (N,) int aligned to `ids`; also returns whether the info order equals the score order.

        Args:
            split: split name.
            ids: (N,) window ids of the saved scores.
            n_real_rows: (N,) real rows per window (must equal info n_messages).
            complete: True if every window of the split was scored (then the id sets must be equal).

        Raises:
            AlignmentError: on any id, field, length or count mismatch.
        """
        self._guard()
        wanted = {str(i): k for k, i in enumerate(ids)}
        if len(wanted) != len(ids):
            raise AlignmentError(f"{split}: duplicate window ids in the scores")
        labels = np.full(len(ids), -1, dtype=np.int64)
        info_ids: list[str] = []
        found = 0
        for path in sorted((self.data_dir / split).glob("b*/part-?????_info.json")):
            self.opened.append(str(path))
            for w in json.loads(path.read_text())["windows"]:
                if self.rebuild_id(w) != w["id"]:
                    raise AlignmentError(f"{split}: id {w['id']!r} != rebuilt {self.rebuild_id(w)!r} ({path})")
                info_ids.append(w["id"])
                k = wanted.get(w["id"])
                if k is None:
                    continue
                if int(w["n_messages"]) != int(n_real_rows[k]):
                    raise AlignmentError(f"{split}: {w['id']} n_messages {w['n_messages']} != {n_real_rows[k]} rows")
                if int(w["is_attack"]) != int(int(w["label"]) != 0):
                    raise AlignmentError(f"{split}: {w['id']} label {w['label']} vs is_attack {w['is_attack']}")
                labels[k] = int(w["label"])
                found += 1
            if not complete and found == len(ids):
                break
        if found != len(ids) or (labels < 0).any():
            raise AlignmentError(f"{split}: {len(ids) - found} scored windows have no label row")
        if complete and len(info_ids) != len(ids):
            raise AlignmentError(f"{split}: {len(info_ids)} info windows but {len(ids)} scored windows")
        expected = ATTACK_CODES.get(split, 0)
        allowed = {expected} | (set(self.val_attack_codes) if split == "val" else set())
        bad = sorted(set(labels.tolist()) - allowed)
        if bad:
            raise AlignmentError(f"{split}: expected labels {sorted(allowed)} only, found {bad}")
        same_order = info_ids[: len(ids)] == [str(i) for i in ids]
        return labels, same_order


class MetadataCheck:
    """metadata.json of the data dir must equal the run's (env.json) and the cache's (manifest.json) by sha256."""

    def __init__(self, data_dir: Path, run_dir: Path, cache_dir: Path, splits: Sequence[str]) -> None:
        self.data_dir, self.run_dir, self.cache_dir, self.splits = data_dir, run_dir, cache_dir, tuple(splits)

    def run(self, allow_mismatch: bool) -> dict[str, Any]:
        """Compares the hashes; raises MetadataMismatchError on any difference unless `allow_mismatch`."""
        actual = file_sha256(self.data_dir / "metadata.json")
        env_path = self.run_dir / "env.json"
        env = json.loads(env_path.read_text()) if env_path.is_file() else {}
        expected = {"run env.json": env.get("encoder_input_metadata_sha256")}
        for split in self.splits:
            manifest = self.cache_dir / split / "manifest.json"
            if manifest.is_file():
                expected[f"cache {split}"] = json.loads(manifest.read_text()).get("metadata_sha256")
        problems = []
        if expected["run env.json"] is None:
            problems.append(f"{env_path} has no encoder_input_metadata_sha256")
        problems += [f"{k}: {v} != {actual}" for k, v in expected.items() if v is not None and v != actual]
        report = {"metadata_sha256": actual, "expected": expected, "problems": problems, "allowed": allow_mismatch}
        if problems and not allow_mismatch:
            raise MetadataMismatchError(
                "metadata.json differs from the run / cache (" + "; ".join(problems) + "); "
                "pass --allow-metadata-mismatch only if you know why"
            )
        for problem in problems:
            _LOG.warning("metadata mismatch allowed: %s", problem)
        return report


class Provenance:
    """Environment and run facts for summary.json (RULE 4)."""

    @staticmethod
    def git(*args: str) -> str:
        """Output of a git command, or "unknown"."""
        import subprocess

        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    @classmethod
    def environment(cls, device: torch.device) -> dict[str, Any]:
        """Versions, device, GPU name, git commit + dirty flag."""
        gpu = None
        if device.type == "cuda":
            gpu = torch.cuda.get_device_name(device)
        elif device.type == "mps":
            gpu = "Apple MPS"
        req = Path("requirements-train.txt")
        return {
            "created_utc": now_utc(),
            "git_commit": cls.git("rev-parse", "HEAD"),
            "git_dirty": bool(cls.git("status", "--porcelain")),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "gpu": gpu,
            "requirements_train_sha256": file_sha256(req) if req.exists() else None,
        }


def now_utc() -> str:
    """ISO time stamp (UTC, microseconds) for ordering evidence."""
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def pick_device(name: str) -> torch.device:
    """`auto` = CUDA, then MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Evaluation:
    """Scores every split with the frozen model, saves the scores, then reads labels and writes the summary."""

    def __init__(self, settings: EvalSettings, access: DataAccess) -> None:
        self.s, self.access = settings, access
        self.run_dir, self.out = Path(settings.run_dir), Path(settings.out_dir)
        self.device = pick_device(settings.device)
        self.events: dict[str, str] = {}

    def _build(self) -> tuple[PretrainConfig, PretrainingModel, Rescaler, dict[str, Any]]:
        """Model settings, model (checkpoint weights or random init) and the scaling, from the run's checkpoint.

        Everything stays on the CPU here: the splits are loaded (possibly by a fork pool) before CUDA is touched.
        """
        path = self.run_dir / self.s.checkpoint
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "space" not in ckpt:
            raise KeyError(f"{path} has no 'space' (FeatureSpace state); cannot reproduce the trained scaling")
        if ckpt.get("config_class") != "T32BenignConfig":
            _LOG.warning("checkpoint config_class is %r, expected 'T32BenignConfig'", ckpt.get("config_class"))
        model_settings = PretrainConfig.from_dict(ckpt["settings"])
        if self.s.random_init:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.s.seed)
                model = PretrainingModel(model_settings)
        else:
            model = PretrainingModel(model_settings)
            model.load_state_dict(ckpt["model"])
        metadata = json.loads((Path(self.s.data_dir) / "metadata.json").read_text())
        rescaler = Rescaler(FeatureSpace(metadata), self.access.restore_space(metadata, ckpt["space"]))
        info = {
            "path": str(path),
            "sha256": file_sha256(path),
            "config_class": ckpt.get("config_class"),
            "step": ckpt.get("step"),
            "epoch": ckpt.get("epoch"),
            "best": ckpt.get("best"),
            "weights": "random init (seed-fixed control)" if self.s.random_init else "checkpoint",
            "metadata_sha256": file_sha256(Path(self.s.data_dir) / "metadata.json"),
            "scaling": getattr(rescaler.trained, "report", None),
            "min_train_rows": int(ckpt["settings"].get("min_train_rows", 1)),
        }
        return model_settings, model, rescaler, info

    def _load(self, split: str, firewall: Any) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, bool]:
        """(x, mask, ids, complete) of a split; smoke keeps the first `smoke_windows`."""
        data = self.access.load(self.s, split, firewall)
        x, mask, ids = torch.as_tensor(data.x), torch.as_tensor(data.mask).bool(), np.asarray(data.ids)
        if len(x) != len(mask) or len(x) != len(ids):
            raise AlignmentError(f"{split}: x {len(x)}, mask {len(mask)}, ids {len(ids)} rows")
        complete = not self.s.smoke
        if self.s.smoke:
            x, mask, ids = x[: self.s.smoke_windows], mask[: self.s.smoke_windows], ids[: self.s.smoke_windows]
        return x, mask, ids, complete

    def _load_all(self, firewall: Any, min_train_rows: int) -> tuple[dict[str, tuple], dict[str, Any]]:
        """Every split's (x, mask, ids, complete), loaded before the model touches the device.

        Train keeps only windows with >= `min_train_rows` real rows, the filter the run trained with (X3).
        """
        splits = {split: self._load(split, firewall) for split in ("train", *SCORED_SPLITS)}
        x, mask, ids, complete = splits["train"]
        before = len(ids)
        if min_train_rows > 1:
            keep = mask.sum(1) >= min_train_rows
            splits["train"] = (x[keep], mask[keep], ids[keep.numpy()], complete)
        train_filter = {
            "min_train_rows": min_train_rows,
            "windows_before": before,
            "windows_used": len(splits["train"][2]),
            "rule": "Mahalanobis / kNN fitted on the train windows the run trained on (real rows >= min_train_rows)",
        }
        return splits, train_filter

    def _score_all(self, scorer: FrozenScorer, splits: dict[str, tuple]) -> tuple[dict[str, Any], dict[str, bool]]:
        """Fits Mahalanobis / kNN on train z, scores every split and saves `scores/<split>.npz` (no labels)."""
        x, mask, _, _ = splits.pop("train")
        z_train = scorer.run("train", x, mask, recon=False)["z"]
        n_train = len(z_train)
        del x, mask
        maha = scoring.Mahalanobis().fit(z_train)
        knn = scoring.KNNScorer(self.s.k_values, self.s.max_bank, self.s.seed).fit(z_train)
        (self.out / "fit").mkdir(parents=True)
        np.savez(self.out / "fit" / "mahalanobis.npz", mean=maha.mean, precision=maha.precision)
        np.save(self.out / "fit" / "knn_bank_rows.npy", knn.bank_rows)
        del z_train
        (self.out / "scores").mkdir()
        fill, complete, counts = None, {}, {}
        for split in SCORED_SPLITS:
            x, mask, ids, complete[split] = splits.pop(split)
            out = scorer.run(split, x, mask, recon=True)
            n_rows = mask.sum(1).numpy().astype(np.int64)
            del x, mask
            recon_raw = out["recon"]
            defined = np.isfinite(recon_raw)
            if not np.array_equal(defined, n_rows >= scoring.MIN_RECON_ROWS):
                raise RuntimeError(f"{split}: reconstruction error defined on other windows than n >= 2 real rows")
            if fill is None:  # benign val is scored first: its median is the label-free fill value
                fill = float(np.median(recon_raw[defined])) if defined.any() else 0.0
            arrays = {"recon": np.where(defined, recon_raw, fill), "mahalanobis": maha.score(out["z"])}
            arrays.update({f"knn_k{k}": v for k, v in knn.score(out["z"]).items()})
            np.savez(
                self.out / "scores" / f"{split}.npz",
                ids=ids.astype(str),
                n_real_rows=n_rows,
                recon_raw=recon_raw,
                recon_defined=defined,
                **arrays,
            )
            counts[split] = {
                "windows": int(len(ids)),
                "complete": complete[split],
                "recon_filled_lt2_rows": int((~defined).sum()),
                "recon_filled_share": float((~defined).mean()) if len(ids) else None,
                f"n_le_{scoring.SHORT_MAX_ROWS}": int((n_rows <= scoring.SHORT_MAX_ROWS).sum()),
                f"n_ge_{scoring.SHORT_MAX_ROWS + 1}": int((n_rows > scoring.SHORT_MAX_ROWS).sum()),
            }
            _LOG.info("scored %s: %d windows (%d without hidden rows)", split, len(ids), int((~defined).sum()))
        fits = {
            "train_windows": n_train,
            "mahalanobis": maha.describe(),
            "knn": knn.describe(),
            "recon_undefined_fill": {
                "value": fill,
                "rule": "median of the defined benign val errors (label-free); windows with < 2 real rows",
            },
            "counts": counts,
        }
        return fits, complete

    @staticmethod
    def load_scores(path: Path) -> scoring.ScoredSplit:
        """A saved score file -> ScoredSplit (no labels)."""
        with np.load(path) as f:
            names = [k for k in f.files if k in ("recon", "mahalanobis") or k.startswith("knn_k")]
            return scoring.ScoredSplit(f["ids"], f["n_real_rows"], {k: f[k].astype(np.float64) for k in names})

    def _metrics(self, complete: dict[str, bool]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Reads labels (after every score file exists) and computes every metric."""
        scored = {s: self.load_scores(self.out / "scores" / f"{s}.npz") for s in SCORED_SPLITS}
        self.events["labels_first_read_utc"] = now_utc()
        reader = LabelReader(Path(self.s.data_dir), self.out / "scores", SCORED_SPLITS, self.s.val_attack_codes)
        labels, order = {}, {}
        for split in SCORED_SPLITS:
            labels[split], order[split] = reader.read(
                split, scored[split].ids, scored[split].n_real_rows, complete[split]
            )
        report = scoring.DetectionReport(scored["val"], self.s.quantiles)
        metrics = {
            split: report.compare(scored["test"], scored[split], labels["test"], labels[split])
            for split in ATTACK_CODES
        }
        alignment = {"info_shards_opened": len(reader.opened), "order_identical": order, "checks": "all passed"}
        return metrics, report.thresholds, alignment

    def run(self) -> dict[str, Any]:
        """Scores, saves, reads labels, writes summary.json and summary.md; returns the summary."""
        if self.out.exists():
            raise FileExistsError(f"output folder {self.out} exists; nothing was overwritten")
        metadata_check = MetadataCheck(
            Path(self.s.data_dir), self.run_dir, Path(self.s.cache_dir), ("train", *SCORED_SPLITS)
        ).run(self.s.allow_metadata_mismatch)
        model_settings, model, rescaler, ckpt_info = self._build()  # fails before any output is written
        splits, train_filter = self._load_all(self.access.make_firewall(), ckpt_info["min_train_rows"])
        self.out.mkdir(parents=True)
        model, rescaler = model.to(self.device), rescaler.to(self.device)  # after every split is in RAM
        scorer = FrozenScorer(model, ReconMasker(model_settings), rescaler, self.s, self.device)
        self.events["scoring_started_utc"] = now_utc()
        fits, complete = self._score_all(scorer, splits)
        fits["train_filter"] = train_filter
        self.events["scores_written_utc"] = now_utc()
        metrics, thresholds, alignment = self._metrics(complete)
        self.events["labels_read_after_scores"] = (
            self.events["labels_first_read_utc"] > self.events["scores_written_utc"]
        )
        config_path = self.run_dir / "config.json"
        summary = {
            "kind": "one-class evaluation" + (" (SMOKE: plumbing only, not results)" if self.s.smoke else ""),
            "run_dir": str(self.run_dir),
            "random_init": self.s.random_init,
            "smoke": self.s.smoke,
            "seed": self.s.seed,
            "eval_settings": dataclasses.asdict(self.s),
            "checkpoint": ckpt_info,
            "model_settings": dataclasses.asdict(model_settings),
            "run_config": json.loads(config_path.read_text()) if config_path.is_file() else None,
            "env": Provenance.environment(self.device),
            "events": self.events,
            "fits": fits,
            "thresholds": thresholds,
            "alignment": alignment,
            "metadata_check": metadata_check,
            "metrics": metrics,
            "protocol": {
                "benign_reference": "test (benign windows of the test vehicles)",
                "thresholds_from": "benign val (window scores; track max / mean of val tracks); "
                "alarm = score > threshold",
                "length_only": "score = -n_real_rows, direction fixed in advance (not chosen on test labels); the "
                "score is discrete, so a val quantile on a tie can leave no window above it (recall 0 is then exact)",
                "track": "window id without the crop index (one link); max and mean of its window scores",
                "recon": "masked MSE on hidden real rows, ReconMasker of the training code, seeded per (seed, split, "
                "pass, batch), averaged over recon_passes; windows sorted by real rows before batching",
                "recon_fill": "windows with < 2 real rows have nothing to hide: their recon score is the benign val "
                "median (filled, not measured), so every 'all' recon metric depends on that fill; the n_ge_2 subset "
                "holds the windows whose error is measured, and every score is reported on it for a like-for-like "
                "comparison",
                "subsets": "all, n_ge_2, n_le_3, n_ge_4 real rows; every score (recon, Mahalanobis, kNN, length-only) "
                "on the same windows with the same val thresholds",
                "comparable_to_baseline": "the like-for-like comparison with task/autoencoder_benign_t32 (same split "
                "and test windows) is GridSybil on the n_ge_4 subset; its other numbers use other subsets",
            },
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
        (self.out / "summary.md").write_text(SummaryTable.render(summary))
        _LOG.info("wrote %s", self.out / "summary.json")
        return summary


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


class SummaryTable:
    """The short markdown table of summary.md."""

    @staticmethod
    def _fmt(value: Any) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    @classmethod
    def render(cls, summary: dict[str, Any]) -> str:
        """Window AUROC (all, n >= 2, n >= 4, n <= 3), PR-AUC, recall / FPR at val p95, track AUROC per attack."""
        lo, hi = f"n_le_{scoring.SHORT_MAX_ROWS}", f"n_ge_{scoring.SHORT_MAX_ROWS + 1}"
        ge2 = f"n_ge_{scoring.MIN_RECON_ROWS}"
        maha = summary["fits"]["mahalanobis"]
        lines = [
            f"# One-class evaluation: {Path(summary['run_dir']).name}",
            "",
            f"{summary['kind']}; weights: {summary['checkpoint']['weights']}; checkpoint sha256 "
            f"`{summary['checkpoint']['sha256'][:12]}`; seed {summary['seed']}; device {summary['env']['device']}.",
            "Each attack split vs benign test; thresholds from benign val; labels read after every score was saved "
            f"({summary['events']['labels_read_after_scores']}).",
            "",
            "`recon (all)*` depends on the fill value of windows with < 2 real rows (benign val median, not measured);"
            " `n≥2` = the windows whose reconstruction error is measured, every score on the same windows. Mahalanobis:"
            f" cond {maha['cond']:.3g}, shrinkage {maha['shrinkage']:g}. Like-for-like with "
            "task/autoencoder_benign_t32: GridSybil, n≥4.",
            "",
        ]
        for split, m in summary["metrics"].items():
            filled = m["counts"]["recon_filled_share"]
            lines += [
                f"## {split}",
                "",
                f"recon filled (< 2 rows): benign test {cls._pct(filled['benign'])}, {split} "
                f"{cls._pct(filled['attack'])}.",
                "",
                "| score | AUROC all | AUROC n≥2 | AUROC n≥4 | AUROC n≤3 | PR-AUC all | recall@p95 | benign FPR@p95 "
                "| track max | track mean |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
            for name, w in m["window"].items():
                p95 = w["all"]["at_threshold"].get("p95", {})
                track = m["track"].get(name, {})
                cells = [
                    w["all"]["auroc"],
                    w[ge2]["auroc"],
                    w[hi]["auroc"],
                    w[lo]["auroc"],
                    w["all"]["pr_auc"],
                    p95.get("recall"),
                    p95.get("benign_fpr"),
                    track.get("max", {}).get("auroc"),
                    track.get("mean", {}).get("auroc"),
                ]
                label = cls._label(name, maha)
                lines.append(f"| {label} | " + " | ".join(cls._fmt(c) for c in cells) + " |")
            c = m["counts"]
            lines += [
                "",
                f"windows benign / attack: {c['all']['benign_windows']} / {c['all']['attack_windows']}; "
                f"n≥2: {c[ge2]['benign_windows']} / {c[ge2]['attack_windows']}; "
                f"n≥4: {c[hi]['benign_windows']} / {c[hi]['attack_windows']}; tracks {c['tracks']['benign']} / "
                f"{c['tracks']['attack']} (mixed-label tracks {c['tracks']['mixed_label_tracks']}).",
                "",
            ]
        return "\n".join(lines)

    @staticmethod
    def _pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:.1f}%"

    @staticmethod
    def _label(name: str, maha: dict[str, Any]) -> str:
        if name == "recon":
            return "recon (all)*"
        if name == "mahalanobis":
            return f"mahalanobis (cond {maha['cond']:.3g}, shrink {maha['shrinkage']:g})"
        return name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser(description="One-class evaluation of a pretrained TimesNet on all/T32_benign")
    p.add_argument("--run", required=True, help="pretraining run folder")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out", default=None, help=f"output folder (default {DEFAULT_EVAL_ROOT}/<run_id>[-random-init])")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--recon-passes", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="a few windows per split: plumbing check, not results")
    p.add_argument("--random-init", action="store_true", help="untrained seed-fixed model (no-pretraining control)")
    p.add_argument(
        "--allow-metadata-mismatch",
        action="store_true",
        help="continue although metadata.json differs from the run's / the cache's (recorded; off by default)",
    )
    p.add_argument(
        "--allow-val-attack",
        type=int,
        action="append",
        default=[],
        metavar="CODE",
        help="attack code allowed in val besides 0 (repeatable; mixed-val contamination runs; thresholds then come "
        "from the mixed val; off by default)",
    )
    return p.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> EvalSettings:
    """EvalSettings from the parsed options (default output folder named after the run)."""
    out = args.out
    if out is None:
        name = Path(args.run).name + ("-random-init" if args.random_init else "")
        if args.smoke:
            name += "-smoke-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = str(Path(DEFAULT_EVAL_ROOT) / name)
    return EvalSettings(
        run_dir=args.run,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        out_dir=out,
        device=args.device,
        batch_size=args.batch_size,
        recon_passes=args.recon_passes,
        seed=args.seed,
        smoke=args.smoke,
        random_init=args.random_init,
        allow_metadata_mismatch=args.allow_metadata_mismatch,
        val_attack_codes=tuple(args.allow_val_attack),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point: parse options, build the evaluation, run it."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = settings_from_args(parse_args(argv))
    summary = Evaluation(settings, DataAccess.from_package()).run()
    _LOG.info("summary: %s", Path(settings.out_dir) / "summary.md")
    if summary["smoke"]:
        _LOG.info("smoke run: plumbing only, the numbers are not results")


if __name__ == "__main__":
    main()
