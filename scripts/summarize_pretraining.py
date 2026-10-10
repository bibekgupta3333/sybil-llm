"""Summarise one self-supervised pretraining run into ``training_summary.json`` (plan task S1.4, training health).

Read-only on the run folder: it reads ``config.json``, ``env.json``, ``metrics.jsonl``, ``train.log`` and
``SHA256SUMS`` (and checks every listed file against its sha256), optionally the text outputs of
``pretrain_monitor.ipynb`` and the window lengths ``n`` of the train shards (label-free: the ``_info`` label files are
never opened). Nothing is written under the run folder, ``data/`` or ``src/data/``.

What it reports:

* hardware and settings (batch, LR, schedule, λ, epochs, seed), wall time, steps per epoch;
* skipped batches (no applicable loss: length-bucketed batches of 1-row windows) per epoch, bracketed from the
  logged cumulative counter, and the share of logged real updates that carried only the reconstruction loss
  (2–3-row windows) versus the full objective (≥ ``min_rows_aux`` rows);
* training curves per component (downsampled to ``--max-points`` by step-bin means), split by batch kind, and the
  eval metrics per epoch (init + each epoch);
* the LR the schedule actually reached, compared with the cosine the config asked for;
* one-line health verdicts with the numbers behind each (converged, plateau, reconstruction vs interpolation,
  physics heads, collapse, length in z, loss coverage, LR schedule, finite values).

These are pretraining-health numbers on the label-free check set, not detection results (RULE 1).

Usage:
    python scripts/summarize_pretraining.py --run src/runs/pretraining/all/T24/model-all
    python scripts/summarize_pretraining.py --run <run_dir> --out <file.json> --no-length-scan
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gzip
import hashlib
import json
import math
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RUN = "src/runs/pretraining/all/T24/model-all"
DEFAULT_NOTEBOOK = "src/model/benign_gridsybil/timesnet/pretrain_monitor.ipynb"
LOSS_KEYS = ("joint", "recon", "nce", "p1", "p2", "p3")
EVAL_KEYS = (
    "val_joint",
    "val_recon",
    "val_recon_interp_baseline",
    "val_nce",
    "val_p1",
    "val_p2",
    "val_p3",
    "auroc_p1",
    "auroc_p2",
    "auroc_p3",
    "z_effective_rank",
    "z_std_mean",
    "z_std_min",
    "length_probe_r2",
    "corr_znorm_length",
    "align",
    "uniformity",
    "cos_pos",
    "cos_neg",
)
PROTECTED = ("data", "src/data")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _round(value: Any, digits: int = 6) -> Any:
    return round(value, digits) if isinstance(value, float) and math.isfinite(value) else value


def _rel_change(new: float, old: float) -> float:
    return (new - old) / abs(old) if old else float("nan")


class RunFolder:
    """Read-only view of a pretraining run folder.

    Attributes:
        path: The run folder.
        config: config.json.
        env: env.json.
        records: Every metrics.jsonl line, in file order.
        log_text: train.log ('' if absent).
    """

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.config: dict[str, Any] = json.loads((path / "config.json").read_text())
        self.env: dict[str, Any] = json.loads((path / "env.json").read_text())
        self.records = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines() if line.strip()]
        log = path / "train.log"
        self.log_text = log.read_text(errors="replace") if log.exists() else ""

    @property
    def train(self) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("kind") == "train"]

    @property
    def evals(self) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("kind") == "eval"]

    @property
    def summary(self) -> dict[str, Any] | None:
        found = [r for r in self.records if r.get("kind") == "summary"]
        return found[-1] if found else None

    def checksums(self) -> dict[str, Any]:
        """Checks every file listed in SHA256SUMS; also lists run files SHA256SUMS does not cover."""
        sums = self.path / "SHA256SUMS"
        if not sums.exists():
            return {"present": False}
        files: dict[str, dict[str, Any]] = {}
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            target = self.path / name
            actual = self._sha256(target) if target.exists() else None
            files[name] = {
                "sha256": digest,
                "bytes": target.stat().st_size if target.exists() else None,
                "ok": actual == digest,
            }
        uncovered = sorted(
            p.name for p in self.path.iterdir() if p.is_file() and p.name not in set(files) | {"SHA256SUMS"}
        )
        return {"present": True, "all_ok": all(f["ok"] for f in files.values()), "files": files, "uncovered": uncovered}

    @staticmethod
    def _sha256(path: pathlib.Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def command(self) -> str | None:
        """The training command echoed at the top of train.log (npm prints it after '> ')."""
        lines = [ln[2:].strip() for ln in self.log_text.splitlines()[:4] if ln.startswith("> ")]
        return lines[-1] if lines else None


class CurveSampler:
    """Downsamples a (step, value) series to at most `max_points` by averaging inside equal step bins."""

    def __init__(self, max_points: int) -> None:
        self.max_points = max_points

    def __call__(self, steps: Sequence[float], values: Sequence[float]) -> dict[str, list]:
        pairs = [(s, v) for s, v in zip(steps, values) if _finite(v)]
        if not pairs:
            return {"step": [], "value": [], "n": []}
        s = np.array([p[0] for p in pairs], dtype=float)
        v = np.array([p[1] for p in pairs], dtype=float)
        if len(s) <= self.max_points:
            return {"step": s.astype(int).tolist(), "value": [_round(x) for x in v.tolist()], "n": [1] * len(s)}
        edges = np.linspace(s.min(), s.max(), self.max_points + 1)
        which = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, self.max_points - 1)
        out: dict[str, list] = {"step": [], "value": [], "n": []}
        for b in range(self.max_points):
            sel = which == b
            if sel.any():
                out["step"].append(int(round(s[sel].mean())))
                out["value"].append(_round(float(v[sel].mean())))
                out["n"].append(int(sel.sum()))
        return out


class SkipAccounting:
    """Skipped batches (no applicable loss) per epoch, from the cumulative ``skipped_steps`` of the logged records.

    Skipped steps are never logged, so the counter is known only at logged real updates. At an epoch boundary step b
    the count lies between the value at the last logged step ≤ b (lower bound; it can only grow) and
    min(that + (b − its step), the value at the first logged step > b) (upper bound). The midpoint is the estimate.
    """

    def __init__(self, train: list[dict[str, Any]], boundaries: list[int], final_step: int) -> None:
        self.points = sorted((int(r["step"]), int(r["skipped_steps"])) for r in train if "skipped_steps" in r)
        self.boundaries = boundaries
        self.final_step = final_step

    def bracket(self, step: int) -> tuple[int, int]:
        before = [p for p in self.points if p[0] <= step]
        after = [p for p in self.points if p[0] > step]
        lo_step, lo = before[-1] if before else (0, 0)
        hi = lo + (step - lo_step)
        if after:
            hi = min(hi, after[0][1])
        return lo, hi

    def per_epoch(self, steps_per_epoch: int) -> list[dict[str, Any]]:
        rows, prev = [], (0, 0)
        for epoch, b in enumerate(self.boundaries, start=1):
            lo, hi = self.bracket(b)
            n_batches = b - (self.boundaries[epoch - 2] if epoch > 1 else 0)
            est_prev = (prev[0] + prev[1]) / 2
            est = (lo + hi) / 2
            rows.append(
                {
                    "epoch": epoch,
                    "end_step": b,
                    "batches": n_batches,
                    "cumulative_skipped_bracket": [lo, hi],
                    "skipped_in_epoch_bracket": [max(0, lo - prev[1]), hi - prev[0]],
                    "skipped_in_epoch_est": round(est - est_prev, 1),
                    "skipped_share_est": round((est - est_prev) / n_batches, 4) if n_batches else None,
                }
            )
            prev = (lo, hi)
        return rows

    def total(self) -> dict[str, Any]:
        lo, hi = self.bracket(self.final_step)
        return {
            "batches": self.final_step,
            "skipped_bracket": [lo, hi],
            "real_updates_bracket": [self.final_step - hi, self.final_step - lo],
            "skipped_share_est": round((lo + hi) / 2 / self.final_step, 4) if self.final_step else None,
        }


class LengthScan:
    """Label-free histogram of real rows ``n`` per window over the train shards (reads ``part-*.json.gz`` only)."""

    def __init__(self, data_dir: pathlib.Path, seq_len: int) -> None:
        self.data_dir = data_dir
        self.seq_len = seq_len

    def __call__(self, min_rows_aux: int) -> dict[str, Any] | None:
        shards = sorted(
            p for p in (self.data_dir / "train").glob("part-*.json.gz") if not p.name.endswith("_info.json.gz")
        )
        if not shards:
            return None
        hist = np.zeros(self.seq_len + 1, dtype=np.int64)
        for shard in shards:
            with gzip.open(shard, "rt") as f:
                for w in json.load(f)["windows"]:
                    hist[min(int(w["n"]), self.seq_len)] += 1
        total = int(hist.sum())
        n1, n23, n4 = int(hist[1]), int(hist[2:min_rows_aux].sum()), int(hist[min_rows_aux:].sum())
        return {
            "scope": "train split as stored (pretraining train + its label-free check set, which is carved out by "
            "vehicle at load time); labels not read",
            "shards": len(shards),
            "windows": total,
            "hist_n": {str(n): int(c) for n, c in enumerate(hist) if c},
            "share_n1_no_loss": round(n1 / total, 4) if total else None,
            "share_n2_3_recon_only": round(n23 / total, 4) if total else None,
            f"share_n_ge_{min_rows_aux}_all_losses": round(n4 / total, 4) if total else None,
        }


class NotebookOutputs:
    """Text outputs (stream + text/plain, figures skipped) of the run monitor notebook."""

    def __init__(self, path: pathlib.Path, max_chars: int = 4000) -> None:
        self.path = path
        self.max_chars = max_chars

    def __call__(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        nb = json.loads(self.path.read_text())
        cells = []
        for i, cell in enumerate(nb.get("cells", [])):
            texts = []
            for out in cell.get("outputs", []):
                if out.get("output_type") == "stream":
                    texts.append("".join(out.get("text", [])))
                elif "text/plain" in out.get("data", {}) and not "".join(out["data"]["text/plain"]).startswith(
                    "<Figure"
                ):
                    texts.append("".join(out["data"]["text/plain"]))
                elif out.get("output_type") == "error":
                    texts.append(f"ERROR {out.get('ename')}: {out.get('evalue')}")
            if texts:
                cells.append({"cell": i, "text": "\n".join(texts)[: self.max_chars]})
        return {
            "path": str(self.path.relative_to(REPO_ROOT)) if self.path.is_relative_to(REPO_ROOT) else str(self.path),
            "cells": cells,
        }


@dataclasses.dataclass
class Verdict:
    """One health check: a short question, a one-line honest answer, and the numbers behind it."""

    check: str
    verdict: str
    status: str  # "ok" | "look" | "no"
    evidence: dict[str, Any]


class TrainingAnalysis:
    """Turns a RunFolder into the summary dictionary (curves, per-epoch metrics, skips, LR check, verdicts)."""

    def __init__(self, run: RunFolder, max_points: int = 400) -> None:
        self.run = run
        self.cfg = run.config
        self.sampler = CurveSampler(max_points)
        self.evals = run.evals
        self.train = run.train
        self.summary = run.summary or {}
        self.final_step = int(self.summary.get("step") or (self.train[-1]["step"] if self.train else 0))
        self.steps_per_epoch = self._steps_per_epoch()
        boundaries = [int(e["step"]) for e in self.evals if str(e.get("tag", "")).startswith("epoch")]
        self.skips = SkipAccounting(self.train, boundaries, self.final_step)

    def _steps_per_epoch(self) -> int | None:
        epochs = [e for e in self.evals if str(e.get("tag", "")).startswith("epoch")]
        if epochs:
            return int(epochs[0]["step"])
        rows = (self.run.env.get("rows") or {}).get("train")
        return math.ceil(rows / self.cfg["batch_size"]) if rows else None

    def batch_kinds(self) -> dict[str, Any]:
        full = set(LOSS_KEYS[1:])
        kinds = {"all_losses": 0, "recon_only": 0, "other": 0}
        for r in self.train:
            present = set(r.get("losses_present", []))
            key = "all_losses" if present == full else "recon_only" if present == {"recon"} else "other"
            kinds[key] += 1
        n = sum(kinds.values())
        tot = self.skips.total()
        skipped = tot["skipped_share_est"] or 0.0
        share = {k: round(v / n, 4) for k, v in kinds.items()} if n else {}
        return {
            "logged_real_updates": n,
            "counts": kinds,
            "share_of_real_updates": share,
            "share_of_all_batches_est": {
                "skipped_no_loss": skipped,
                **{k: round(v * (1 - skipped), 4) for k, v in share.items()},
            },
            "note": "logged records are the real updates at steps divisible by log_every (plus the first), an "
            "unbiased sample of real updates; batches are length-bucketed, so a batch's kind follows its length",
        }

    def curves(self) -> dict[str, Any]:
        out: dict[str, Any] = {"method": f"mean per equal step bin, ≤ {self.sampler.max_points} points", "train": {}}
        groups = {
            "all": self.train,
            "all_losses_batches": [r for r in self.train if "nce" in r.get("losses_present", [])],
            "recon_only_batches": [r for r in self.train if r.get("losses_present") == ["recon"]],
        }
        for gname, rows in groups.items():
            keys = LOSS_KEYS if gname != "recon_only_batches" else ("joint", "recon")
            out["train"][gname] = {k: self.sampler([r["step"] for r in rows], [r.get(k) for r in rows]) for k in keys}
        for k in ("lr", "grad_norm", "mem_gb", "step_seconds", "nce_masked_pairs", "skipped_steps"):
            out["train"][k] = self.sampler([r["step"] for r in self.train], [r.get(k) for r in self.train])
        return out

    def eval_table(self) -> list[dict[str, Any]]:
        rows = []
        for e in self.evals:
            row = {"tag": e.get("tag"), "epoch": e.get("epoch"), "step": e.get("step")}
            row.update({k: _round(e.get(k)) for k in EVAL_KEYS if k in e})
            if _finite(e.get("val_recon")) and _finite(e.get("val_recon_interp_baseline")):
                row["recon_over_interp"] = _round(e["val_recon"] / e["val_recon_interp_baseline"])
            rows.append(row)
        return rows

    def init_to_final(self) -> dict[str, Any]:
        if not self.evals:
            return {}
        first, last = self.evals[0], self.evals[-1]
        best = min(self.evals[1:] or self.evals, key=lambda e: e.get("val_joint", float("inf")))
        return {
            "init_tag": first.get("tag"),
            "final_tag": last.get("tag"),
            "best_val_joint_tag": best.get("tag"),
            "metrics": {
                k: {"init": _round(first.get(k)), "final": _round(last.get(k))} for k in EVAL_KEYS if k in last
            },
        }

    def per_feature_recon(self) -> list[dict[str, Any]] | None:
        last = self.evals[-1] if self.evals else {}
        model, base = last.get("val_recon_per_feature"), last.get("val_recon_interp_baseline_per_feature")
        init = self.evals[0].get("val_recon_per_feature") if self.evals else None
        names = self.run.env.get("features") or list((self.run.env.get("scaling") or {}).keys())
        if not model or not base:
            return None
        names = names if len(names) == len(model) else [f"f{i}" for i in range(len(model))]
        return [
            {
                "feature": n,
                "model_final": _round(m),
                "model_init": _round(init[i]) if init else None,
                "interp": _round(b),
                "ratio": _round(m / b) if b else None,
                "beats_interp": m < b,
            }
            for i, (n, m, b) in enumerate(zip(names, model, base))
        ]

    def lr_schedule(self) -> dict[str, Any]:
        """Compares the last logged LR with the cosine the config describes over all batches of max_epochs."""
        if not self.train:
            return {}
        last = self.train[-1]
        peak, warm = float(self.cfg["lr"]), int(self.cfg["warmup_steps"])
        total = int(self.cfg.get("max_steps") or 0) or (self.cfg["max_epochs"] * (self.steps_per_epoch or 0))

        def factor(step: float) -> float:
            if step < warm:
                return (step + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

        observed = float(last["lr"])
        real_updates = int(last["step"]) - int(last.get("skipped_steps", 0))
        frac = min(1.0, max(-1.0, 2 * observed / peak - 1))
        inferred = warm + math.acos(frac) / math.pi * max(1, total - warm)
        return {
            "peak_lr": peak,
            "warmup_steps": warm,
            "planned_total_steps": total,
            "last_logged_step": int(last["step"]),
            "last_logged_lr": observed,
            "planned_lr_at_that_step": _round(peak * factor(int(last["step"])), 9),
            "lr_if_schedule_counts_real_updates": _round(peak * factor(real_updates), 9),
            "real_updates_at_that_step": real_updates,
            "scheduler_steps_inferred_from_lr": round(inferred),
            "final_lr_over_peak": _round(observed / peak, 4),
            "note": "train.py steps the LR scheduler only on real updates while total_steps counts every batch "
            "(skipped batches included), so the cosine reaches only real_updates / total of its length",
        }

    def timing(self) -> dict[str, Any]:
        hours = self.summary.get("hours")
        epochs = self.summary.get("epoch") or len([e for e in self.evals if str(e.get("tag", "")).startswith("epoch")])
        by_kind: dict[str, list[float]] = {"all_losses": [], "recon_only": []}
        for r in self.train:
            kind = "all_losses" if "nce" in r.get("losses_present", []) else "recon_only"
            if _finite(r.get("step_seconds")):
                by_kind[kind].append(r["step_seconds"])
        return {
            "wall_hours": hours,
            "epochs": epochs,
            "mean_hours_per_epoch": round(hours / epochs, 3) if hours and epochs else None,
            "per_epoch_note": "no per-epoch timestamps are logged; mean = wall hours / epochs (includes 11 evals)",
            "load_seconds": self.run.env.get("load_seconds"),
            "eval_seconds_mean": (
                _round(float(np.mean([e["eval_seconds"] for e in self.evals])), 2) if self.evals else None
            ),
            "step_seconds_median": {k: _round(float(np.median(v)), 3) if v else None for k, v in by_kind.items()},
            "step_seconds_note": "forward + backward time of logged real updates (excludes data gathering)",
        }

    def verdicts(self, lengths: dict[str, Any] | None) -> list[Verdict]:
        ev = self.evals
        if len(ev) < 2:
            return []
        first, last = ev[0], ev[-1]
        epochs = [e for e in ev if str(e.get("tag", "")).startswith("epoch")]
        joint = [e["val_joint"] for e in epochs]
        out: list[Verdict] = []

        best_before = min(joint[:-1]) if len(joint) > 1 else joint[0]
        last_gain = -_rel_change(joint[-1], best_before)
        best_is_last = joint[-1] <= min(joint)
        lr = self.lr_schedule()
        out.append(
            Verdict(
                "converged?",
                (
                    f"No — val_joint still fell {last_gain:.1%} in the last epoch (best = last epoch) and the LR was "
                    f"still {lr.get('final_lr_over_peak', float('nan')):.0%} of peak; more epochs would likely help."
                    if best_is_last and last_gain > 0.01
                    else "Roughly — the last epoch changed val_joint by < 1% relative to the best before it."
                ),
                "look" if best_is_last and last_gain > 0.01 else "ok",
                {
                    "val_joint_per_epoch": [_round(j, 4) for j in joint],
                    "last_epoch_gain_vs_prior_best": _round(last_gain, 4),
                },
            )
        )

        def tail_change(key: str, k: int = 3) -> float | None:
            vals = [e.get(key) for e in epochs]
            if len(vals) <= k or not all(_finite(v) for v in vals):
                return None
            return _rel_change(vals[-1], vals[-1 - k])

        plateau = {
            k: _round(tail_change(k), 4) for k in ("val_joint", "val_recon", "val_nce", "val_p1", "val_p2", "val_p3")
        }
        flat = [k for k, v in plateau.items() if v is not None and abs(v) < 0.03]
        moving = [k for k, v in plateau.items() if v is not None and abs(v) >= 0.03]
        out.append(
            Verdict(
                "plateau?",
                f"Flat (< 3% change over the last 3 epochs): {', '.join(flat) or 'none'}; still moving: "
                f"{', '.join(moving) or 'none'}.",
                "ok",
                {"relative_change_last_3_epochs": plateau},
            )
        )

        ratio = last["val_recon"] / last["val_recon_interp_baseline"]
        feats = self.per_feature_recon() or []
        n_beat = sum(f["beats_interp"] for f in feats)
        out.append(
            Verdict(
                "reconstruction beats interpolation?",
                (
                    f"Yes — val_recon {last['val_recon']:.4f} vs linear interpolation {last['val_recon_interp_baseline']:.4f} "
                    f"(ratio {ratio:.2f}; init {first['val_recon']:.4f}); {n_beat}/{len(feats)} features beat it."
                    if ratio < 1
                    else f"No — val_recon {last['val_recon']:.4f} ≥ interpolation {last['val_recon_interp_baseline']:.4f}."
                ),
                "ok" if ratio < 1 else "no",
                {"ratio": _round(ratio, 4), "features_beating_interp": n_beat, "features": len(feats)},
            )
        )

        aurocs = {k: (_round(first.get(k), 4), _round(last.get(k), 4)) for k in ("auroc_p1", "auroc_p2", "auroc_p3")}
        learned = all(_finite(v[1]) and v[1] > 0.9 for v in aurocs.values())
        out.append(
            Verdict(
                "physics heads learn?",
                "Yes, on their pretext task — AUROC on injected violations (check set) "
                + ", ".join(f"{k[-2:].upper()} {a:.3f}→{b:.3f}" for k, (a, b) in aurocs.items())
                + ". This is detection of violations we inject, not of real attacks.",
                "ok" if learned else "look",
                {"init_final": aurocs},
            )
        )

        rank0, rank1 = first["z_effective_rank"], last["z_effective_rank"]
        zmin = [e.get("z_std_min") for e in ev]
        gap = last["cos_pos"] - last["cos_neg"]
        collapsed = rank1 < 5 or (last.get("z_std_min") or 0) < 0.01 or gap < 0.1
        out.append(
            Verdict(
                "embedding collapse?",
                (
                    f"No — effective rank of z {rank0:.1f}→{rank1:.1f} (of d = {self.cfg['d_model']}), cos_pos − cos_neg "
                    f"{gap:.2f}; the least-used dimension's std fell {zmin[0]:.3f}→{zmin[-1]:.3f} (watch, not collapse)."
                    if not collapsed
                    else f"Possible collapse — effective rank {rank1:.1f}, cos gap {gap:.2f}."
                ),
                "no" if collapsed else "ok",
                {
                    "z_effective_rank": [_round(rank0, 3), _round(rank1, 3)],
                    "z_std_min_per_eval": [_round(z, 4) for z in zmin],
                    "cos_pos_minus_cos_neg": _round(gap, 4),
                    "uniformity": [_round(first.get("uniformity"), 4), _round(last.get("uniformity"), 4)],
                },
            )
        )

        r2 = [e.get("length_probe_r2") for e in ev]
        out.append(
            Verdict(
                "length encoded in z?",
                f"Yes, strongly — a ridge probe recovers window length from z with R² {r2[0]:.3f} (init) → {r2[-1]:.3f} "
                f"(final); corr(|z|, length) {last.get('corr_znorm_length'):.2f}. Probes can exploit length (F11).",
                "look" if (r2[-1] or 0) > 0.5 else "ok",
                {
                    "length_probe_r2_per_eval": [_round(v, 4) for v in r2],
                    "corr_znorm_length_final": _round(last.get("corr_znorm_length"), 4),
                },
            )
        )

        kinds = self.batch_kinds()
        shares = kinds["share_of_all_batches_est"]
        verdict = (
            f"Thin — ≈{shares.get('skipped_no_loss', 0):.0%} of batches had no loss (1-row windows, skipped), "
            f"≈{shares.get('recon_only', 0):.0%} reconstruction only (2–3 rows), ≈{shares.get('all_losses', 0):.0%} "
            f"all five losses (≥ {self.cfg.get('min_rows_aux')} rows)."
        )
        if lengths:
            verdict += (
                f" Train shards: {lengths['share_n1_no_loss']:.1%} of windows have 1 row, "
                f"{lengths['share_n2_3_recon_only']:.1%} have 2–3."
            )
        out.append(
            Verdict("how much data trains each loss?", verdict, "look", {"batch_kinds": kinds, "lengths": lengths})
        )

        planned = lr.get("planned_lr_at_that_step")
        off_schedule = planned is not None and lr["last_logged_lr"] - planned > 0.1 * lr["peak_lr"]
        out.append(
            Verdict(
                "LR schedule as configured?",
                (
                    f"No — the cosine was meant to reach ≈{planned:.2e} by the last logged step but the LR was "
                    f"{lr['last_logged_lr']:.2e} ({lr['final_lr_over_peak']:.0%} of peak): the scheduler counts only real "
                    f"updates (≈{lr['real_updates_at_that_step']:,}) against a total of {lr['planned_total_steps']:,} batches."
                    if off_schedule
                    else "Yes — the logged LR follows the configured warm-up + cosine."
                ),
                "look" if off_schedule else "ok",
                lr,
            )
        )

        nonfinite = sum(1 for r in self.train for k, v in r.items() if isinstance(v, float) and not math.isfinite(v))
        out.append(
            Verdict(
                "training values finite?",
                f"{'Yes' if nonfinite == 0 else 'No'} — {nonfinite} non-finite values in {len(self.train)} logged train "
                "records (absent losses are omitted, not NaN; a table that pads them with NaN can misreport this).",
                "ok" if nonfinite == 0 else "no",
                {"nonfinite_values": nonfinite, "logged_train_records": len(self.train)},
            )
        )
        return out

    def settings(self) -> dict[str, Any]:
        keys = (
            "dataset",
            "data_dir",
            "seq_len",
            "n_features",
            "d_model",
            "d_ff",
            "n_blocks",
            "top_k",
            "n_kernels",
            "dropout",
            "proj_dim",
            "batch_size",
            "length_bucketed",
            "lr",
            "weight_decay",
            "warmup_steps",
            "grad_clip",
            "max_epochs",
            "max_steps",
            "max_hours",
            "patience",
            "lambda_recon",
            "lambda_nce",
            "lambda_p1",
            "lambda_p2",
            "lambda_p3",
            "recon_ratio",
            "block_ratio",
            "temperature",
            "inject_prob",
            "min_rows_aux",
            "robust_features",
            "mask_same_broadcast",
            "eval_windows",
            "holdout_frac",
            "seed",
            "log_every",
        )
        out = {k: self.cfg.get(k) for k in keys if k in self.cfg}
        out["schedule"] = "linear warm-up (warmup_steps) then cosine to 0 over max_epochs × steps_per_epoch (LambdaLR)"
        out["optimiser"] = "AdamW"
        return out

    def hardware(self) -> dict[str, Any]:
        env = self.run.env
        keys = (
            "device",
            "platform",
            "python",
            "torch",
            "numpy",
            "cpu_threads",
            "git_commit",
            "git_dirty",
            "created_utc",
        )
        out = {k: env.get(k) for k in keys}
        mem = [r["mem_gb"] for r in self.train if _finite(r.get("mem_gb"))]
        out["mem_gb_peak"] = max(mem) if mem else None
        out["mem_gb_note"] = "process RSS (host RAM); GPU memory is not logged on CUDA"
        log = self.run.log_text
        out["instance_note"] = (
            "g4dn.xlarge (Tesla T4) per agent.md; env.json does not record the GPU model"
            if ("cuda" in str(env.get("device")))
            else None
        )
        out["train_log_mentions_gpu"] = any(s in log for s in ("Tesla", "T4", "NVIDIA"))
        return out


class SummaryWriter:
    """Builds the summary for one run and writes it as JSON outside the run folder and the protected trees."""

    def __init__(
        self, run_dir: pathlib.Path, out: pathlib.Path, notebook: pathlib.Path | None, scan: bool, max_points: int
    ) -> None:
        self.run_dir, self.out, self.notebook, self.scan, self.max_points = run_dir, out, notebook, scan, max_points
        self._guard()

    def _guard(self) -> None:
        out = self.out.resolve()
        if out.is_relative_to(self.run_dir.resolve()):
            raise ValueError(f"refusing to write inside the run folder: {out}")
        for p in PROTECTED:
            if out.is_relative_to((REPO_ROOT / p).resolve()):
                raise ValueError(f"refusing to write under protected {p}/: {out}")

    def build(self) -> dict[str, Any]:
        run = RunFolder(self.run_dir)
        analysis = TrainingAnalysis(run, self.max_points)
        lengths = None
        if self.scan:
            data_dir = pathlib.Path(run.config["data_dir"])
            data_dir = data_dir if data_dir.is_absolute() else REPO_ROOT / data_dir
            lengths = LengthScan(data_dir, int(run.config["seq_len"]))(int(run.config.get("min_rows_aux", 4)))
        verdicts = analysis.verdicts(lengths)
        rel = self.run_dir.resolve()
        return {
            "kind": "training_summary",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "script": "scripts/summarize_pretraining.py",
            "run_dir": str(rel.relative_to(REPO_ROOT)) if rel.is_relative_to(REPO_ROOT) else str(rel),
            "scope_note": "pretraining health on the label-free check set (10% of train vehicles); no attack "
            "labels read; not a detection result",
            "command": run.command(),
            "checksums": run.checksums(),
            "hardware": analysis.hardware(),
            "settings": analysis.settings(),
            "data": {
                "rows": run.env.get("rows"),
                "splits_opened": run.env.get("splits_opened"),
                "label_files_opened": run.env.get("label_files_opened"),
                "check_set": run.env.get("check_set"),
                "dataset_note": run.env.get("dataset_note"),
                "length_scan": lengths,
            },
            "params": {k: (run.env.get("preflight") or {}).get(k) for k in ("params_encoder", "params_total")},
            "preflight": run.env.get("preflight"),
            "run_summary": run.summary,
            "steps": {
                "steps_per_epoch": analysis.steps_per_epoch,
                "total_batches": analysis.final_step,
                **analysis.skips.total(),
                "per_epoch": analysis.skips.per_epoch(analysis.steps_per_epoch or 0),
                "method": SkipAccounting.__doc__.split("\n\n")[1].strip().replace("\n    ", " "),
            },
            "batch_kinds": analysis.batch_kinds(),
            "timing": analysis.timing(),
            "lr_schedule": analysis.lr_schedule(),
            "eval_per_epoch": analysis.eval_table(),
            "eval_init_to_final": analysis.init_to_final(),
            "recon_per_feature_final": analysis.per_feature_recon(),
            "curves": analysis.curves(),
            "verdicts": [dataclasses.asdict(v) for v in verdicts],
            "notebook_text_outputs": NotebookOutputs(self.notebook)() if self.notebook else None,
        }

    def write(self) -> dict[str, Any]:
        summary = self.build()
        self.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=1, allow_nan=False, default=str))
        tmp.replace(self.out)
        return summary


def default_out(run_dir: pathlib.Path) -> pathlib.Path:
    """src/runs/pretraining/<...>/<run> -> src/runs/evaluation/<...>/<run>/training_summary.json."""
    parts = list(run_dir.parts)
    if "pretraining" in parts:
        parts[parts.index("pretraining")] = "evaluation"
        return pathlib.Path(*parts) / "training_summary.json"
    return run_dir.parent / f"{run_dir.name}-training_summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", default=DEFAULT_RUN, help=f"pretraining run folder (default {DEFAULT_RUN})")
    p.add_argument("--out", default=None, help="output JSON (default: the run path with pretraining -> evaluation)")
    p.add_argument(
        "--notebook", default=DEFAULT_NOTEBOOK, help="monitor notebook whose text outputs to include ('' = none)"
    )
    p.add_argument("--no-length-scan", action="store_true", help="skip the label-free n-histogram of the train shards")
    p.add_argument("--max-points", type=int, default=400, help="max points per downsampled curve")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = pathlib.Path(args.run)
    run_dir = run_dir if run_dir.is_absolute() else REPO_ROOT / run_dir
    out = pathlib.Path(args.out) if args.out else default_out(run_dir)
    out = out if out.is_absolute() else REPO_ROOT / out
    notebook = (REPO_ROOT / args.notebook) if args.notebook else None
    summary = SummaryWriter(run_dir, out, notebook, not args.no_length_scan, args.max_points).write()
    print(f"wrote {out}")
    for v in summary["verdicts"]:
        print(f"[{v['status']:>4}] {v['check']} {v['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
