"""Tests for `src/model/all_t32_benign/timesnet/evaluate.py` on a tiny synthetic data dir and a fake run (CPU).

The fake run holds a randomly initialised `PretrainingModel` (small widths) and a FeatureSpace state with one robust
feature, so the rescale path is exercised. The data contract (load_split, restore_space, EvalFirewall) is replaced by
small fakes that read the synthetic JSON shards, so these tests do not depend on the package's data.py.
"""

from __future__ import annotations

import dataclasses
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from src.model.all_t32_benign.timesnet import evaluate
from src.model.all_t32_benign.timesnet.base_config import PretrainConfig
from src.model.all_t32_benign.timesnet.base_data import FEATURES, FeatureSpace
from src.model.all_t32_benign.timesnet.heads import PretrainingModel

T = 32
SPLIT_LABELS = {"train": 0, "val": 0, "test": 0, **evaluate.ATTACK_CODES}
SPLIT_SIZES = {
    "train": 60,
    "val": 30,
    "test": 30,
    "test_gridsybil": 12,
    "test_datareplay": 10,
    "test_dosrandom": 10,
    "test_dosdisruptive": 8,
}


def _windows(split: str, n: int, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    """Feature windows and their info rows; every second link has two crops (tracks with 2 windows)."""
    feats, infos = [], []
    code = SPLIT_LABELS[split]
    short = split in ("test_datareplay", "test_dosrandom", "test_dosdisruptive")
    k = 0
    link = 0
    while k < n:
        crops = 2 if link % 2 else 1
        for crop in range(min(crops, n - k)):
            rows = int(rng.integers(1, 4)) if short else int(rng.integers(1, T + 1))
            x = np.zeros((T, 13))
            x[:rows] = rng.normal(size=(rows, 13)) + (1.5 if code else 0.0)
            mask = [1] * rows + [0] * (T - rows)
            receiver_file = f"Fam_0709/run1/traceJSON-{split}-{link}.json"
            info = {
                "receiver_file": receiver_file,
                "sender_pseudo": 100 + link,
                "sender": 7 + link,
                "link_window_index": crop,
                "n_messages": rows,
                "label": code,
                "label_name": "Benign" if code == 0 else "Attack",
                "is_attack": int(code != 0),
            }
            wid = f"{receiver_file}#{100 + link}-{7 + link}#{crop}"
            info["id"] = wid
            feats.append({"id": wid, "x": np.round(x, 4).tolist(), "mask": mask})
            infos.append(info)
            k += 1
        link += 1
    return feats, infos


def _metadata() -> dict:
    return {"features": list(FEATURES), "normalisation": {"mean": [0.0] * 13, "std": [1.0] * 13}}


@pytest.fixture()
def fake_setup(tmp_path: Path):
    """(data_dir, run_dir, access): synthetic shards, a fake run with best.pt, and a fake data contract."""
    rng = np.random.default_rng(0)
    data_dir = tmp_path / "data"
    for split, n in SPLIT_SIZES.items():
        folder = data_dir / split / "b32"
        folder.mkdir(parents=True)
        feats, infos = _windows(split, n, rng)
        (folder / "part-00000.json").write_text(json.dumps({"seq_len": T, "windows": feats}))
        (folder / "part-00000_info.json").write_text(json.dumps({"split": split, "windows": infos}))
    (data_dir / "metadata.json").write_text(json.dumps(_metadata()))
    settings = PretrainConfig(seq_len=T, d_model=16, d_ff=8, n_blocks=1, proj_dim=8)
    torch.manual_seed(1)
    model = PretrainingModel(settings)
    space = FeatureSpace(_metadata())
    state = {"mean": [0.0] * 13, "std": [1.0] * 13, "robust": [False] * 13, "soft_clip": 5.0, "report": {}}
    state["mean"][0], state["std"][0], state["robust"][0] = 0.2, 0.5, True  # one robust feature: rescale is real
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(settings)))
    sha = evaluate.file_sha256(data_dir / "metadata.json")
    (run_dir / "env.json").write_text(json.dumps({"encoder_input_metadata_sha256": sha}))
    torch.save(
        {
            "model": model.state_dict(),
            "encoder": model.encoder.state_dict(),
            "settings": dataclasses.asdict(settings),
            "space": state,
            "config_class": "T32BenignConfig",
            "step": 3,
            "epoch": 1,
            "best": 1.0,
        },
        run_dir / "best.pt",
    )
    del space

    def load_split(cache_dir, data_dir_, split, firewall, max_shards=None):
        firewall.check(split)
        data = json.loads((Path(data_dir_) / split / "b32" / "part-00000.json").read_text())["windows"]
        return types.SimpleNamespace(
            x=torch.tensor([w["x"] for w in data], dtype=torch.float32),
            mask=torch.tensor([w["mask"] for w in data], dtype=torch.bool),
            ids=[w["id"] for w in data],
            split=split,
        )

    def restore_space(metadata, st):
        sp = FeatureSpace(metadata)
        sp.mean, sp.std = torch.tensor(st["mean"]), torch.tensor(st["std"])
        sp.robust, sp.soft_clip = torch.tensor(st["robust"]), st["soft_clip"]
        return sp

    class Firewall:
        def check(self, split):
            if "_info" in split:
                raise PermissionError(split)

    return data_dir, run_dir, evaluate.DataAccess(load_split, restore_space, Firewall)


def _settings(data_dir: Path, run_dir: Path, out: Path, **kw) -> evaluate.EvalSettings:
    base = dict(
        run_dir=str(run_dir),
        data_dir=str(data_dir),
        cache_dir=str(data_dir / "cache"),
        out_dir=str(out),
        device="cpu",
        batch_size=16,
        recon_passes=2,
        max_bank=40,
    )
    base.update(kw)
    return evaluate.EvalSettings(**base)


def test_end_to_end_writes_scores_summary_and_orders_labels_after_scores(fake_setup, tmp_path, monkeypatch):
    data_dir, run_dir, access = fake_setup
    out = tmp_path / "eval"
    original = evaluate.LabelReader.read

    def guarded(self, split, ids, n_real_rows, complete):
        for s in evaluate.SCORED_SPLITS:  # every score file exists before the first label is read
            assert (out / "scores" / f"{s}.npz").is_file(), s
        return original(self, split, ids, n_real_rows, complete)

    monkeypatch.setattr(evaluate.LabelReader, "read", guarded)
    summary = evaluate.Evaluation(_settings(data_dir, run_dir, out), access).run()
    for s in evaluate.SCORED_SPLITS:
        with np.load(out / "scores" / f"{s}.npz") as f:
            assert len(f["ids"]) == SPLIT_SIZES[s]
            assert {
                "recon",
                "recon_raw",
                "recon_defined",
                "mahalanobis",
                "knn_k5",
                "knn_k10",
                "knn_k20",
                "n_real_rows",
            } <= set(f.files)
            assert np.isfinite(f["recon"]).all() and np.isfinite(f["mahalanobis"]).all()
            assert (np.isnan(f["recon_raw"]) == (f["n_real_rows"] < 2)).all()  # < 2 rows: nothing hidden, flagged
    assert (out / "summary.json").is_file() and (out / "summary.md").is_file()
    assert (out / "fit" / "mahalanobis.npz").is_file()
    assert summary["events"]["labels_read_after_scores"] is True
    assert set(summary["metrics"]) == set(evaluate.ATTACK_CODES)
    m = summary["metrics"]["test_gridsybil"]
    assert set(m["window"]) == {"recon", "mahalanobis", "knn_k5", "knn_k10", "knn_k20", "length_only"}
    assert set(m["window"]["recon"]) == {"all", "n_ge_2", "n_le_3", "n_ge_4"}
    assert set(m["window"]["knn_k5"]) == {"all", "n_ge_2", "n_le_3", "n_ge_4"}
    assert m["counts"]["recon_filled_share"]["attack"] is not None
    assert "n≥2" in (out / "summary.md").read_text() and "recon (all)*" in (out / "summary.md").read_text()
    assert "n_ge_4" in summary["protocol"]["comparable_to_baseline"]
    assert summary["metadata_check"]["problems"] == []
    assert summary["fits"]["train_filter"]["windows_used"] == SPLIT_SIZES["train"]
    assert m["window"]["recon"]["all"]["n_benign"] == SPLIT_SIZES["test"]
    assert set(summary["thresholds"]["window"]["recon"]) == {"p95", "p99"}
    assert len(summary["checkpoint"]["sha256"]) == 64
    assert summary["alignment"]["order_identical"]["test"] is True
    json.loads((out / "summary.json").read_text())  # valid JSON


def test_refuses_existing_output_folder(fake_setup, tmp_path):
    data_dir, run_dir, access = fake_setup
    out = tmp_path / "eval"
    out.mkdir()
    with pytest.raises(FileExistsError):
        evaluate.Evaluation(_settings(data_dir, run_dir, out), access).run()


def test_random_init_is_seed_fixed_and_differs_from_checkpoint(fake_setup, tmp_path):
    data_dir, run_dir, access = fake_setup
    runs = {}
    for name, kw in (("a", {"random_init": True}), ("b", {"random_init": True}), ("ckpt", {})):
        out = tmp_path / name
        summary = evaluate.Evaluation(_settings(data_dir, run_dir, out, **kw), access).run()
        runs[name] = np.load(out / "scores" / "test.npz")["mahalanobis"]
        assert summary["random_init"] is bool(kw)
    np.testing.assert_array_equal(runs["a"], runs["b"])
    assert not np.allclose(runs["a"], runs["ckpt"])


def test_reconstruction_masks_are_deterministic(fake_setup, tmp_path):
    data_dir, run_dir, access = fake_setup
    for name in ("a", "b"):
        evaluate.Evaluation(_settings(data_dir, run_dir, tmp_path / name), access).run()
    a, b = (np.load(tmp_path / n / "scores" / "val.npz")["recon"] for n in ("a", "b"))
    np.testing.assert_array_equal(a, b)


def test_id_misalignment_raises(fake_setup, tmp_path):
    data_dir, run_dir, access = fake_setup
    info_path = data_dir / "test" / "b32" / "part-00000_info.json"
    info = json.loads(info_path.read_text())
    info["windows"][3]["sender"] = 99999  # the id no longer matches its fields
    info_path.write_text(json.dumps(info))
    with pytest.raises(evaluate.AlignmentError):
        evaluate.Evaluation(_settings(data_dir, run_dir, tmp_path / "eval"), access).run()


def test_unexpected_attack_code_raises(fake_setup, tmp_path):
    data_dir, run_dir, access = fake_setup
    info_path = data_dir / "test_gridsybil" / "b32" / "part-00000_info.json"
    info = json.loads(info_path.read_text())
    info["windows"][0]["label"] = 17
    info_path.write_text(json.dumps(info))
    with pytest.raises(evaluate.AlignmentError):
        evaluate.Evaluation(_settings(data_dir, run_dir, tmp_path / "eval"), access).run()


def test_label_reader_refuses_before_scores_exist(tmp_path):
    reader = evaluate.LabelReader(tmp_path, tmp_path / "scores", ("val",))
    with pytest.raises(RuntimeError):
        reader.read("val", np.array(["a#1-2#0"]), np.array([3]), complete=True)


def test_rescaler_matches_feature_space_rescale():
    meta = _metadata()
    stored, trained = FeatureSpace(meta), FeatureSpace(meta)
    trained.mean[0], trained.std[0], trained.robust[0] = 0.2, 0.5, True
    x = torch.randn(4, T, 13) * 4
    mask = torch.zeros(4, T, dtype=torch.bool)
    mask[:, :10] = True
    x = x * mask.unsqueeze(-1)
    out = evaluate.Rescaler(stored, trained)(x, mask)
    expected = trained.norm(stored.raw(x.clone()), mask)
    torch.testing.assert_close(out, expected)
    assert float(out[~mask].abs().max()) == 0.0
    assert float((out[mask][:, 0] - x[mask][:, 0]).abs().max()) > 0  # the robust feature really changed


def test_metadata_mismatch_fails_unless_allowed(fake_setup, tmp_path):
    data_dir, run_dir, access = fake_setup
    (run_dir / "env.json").write_text(json.dumps({"encoder_input_metadata_sha256": "0" * 64}))
    with pytest.raises(evaluate.MetadataMismatchError):
        evaluate.Evaluation(_settings(data_dir, run_dir, tmp_path / "a"), access).run()
    assert not (tmp_path / "a").exists()  # nothing written
    summary = evaluate.Evaluation(
        _settings(data_dir, run_dir, tmp_path / "b", allow_metadata_mismatch=True), access
    ).run()
    assert summary["metadata_check"]["problems"] and summary["metadata_check"]["allowed"] is True


def test_cache_manifest_with_other_metadata_fails(fake_setup, tmp_path):
    data_dir, run_dir, access = fake_setup
    manifest = data_dir / "cache" / "val" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"metadata_sha256": "f" * 64}))
    with pytest.raises(evaluate.MetadataMismatchError):
        evaluate.Evaluation(_settings(data_dir, run_dir, tmp_path / "eval"), access).run()


def test_min_train_rows_filters_the_fit_windows(fake_setup, tmp_path):
    data_dir, run_dir, access = fake_setup
    ckpt = torch.load(run_dir / "best.pt", weights_only=False)
    ckpt["settings"]["min_train_rows"] = 8
    torch.save(ckpt, run_dir / "best.pt")
    feats = json.loads((data_dir / "train" / "b32" / "part-00000.json").read_text())["windows"]
    expected = sum(sum(w["mask"]) >= 8 for w in feats)
    summary = evaluate.Evaluation(_settings(data_dir, run_dir, tmp_path / "eval"), access).run()
    assert summary["fits"]["train_filter"]["min_train_rows"] == 8
    assert summary["fits"]["train_filter"]["windows_used"] == expected < SPLIT_SIZES["train"]
    assert summary["fits"]["train_windows"] == expected
