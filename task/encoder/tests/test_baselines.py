"""Tests of the shortcut baselines on a synthetic T64 folder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from task.encoder.baselines import BaselineConfig, BaselineRunner, LengthBaseline, ShardReader, window_stats

T, F = 64, 13


def write_split(root: Path, split: str, n_shards: int, per_shard: int, rng: np.random.Generator) -> list[dict]:
    """Writes synthetic shards in the T64 layout; GridSybil windows are shorter and shifted."""
    entries = []
    (root / split / "b64").mkdir(parents=True, exist_ok=True)
    for s in range(n_shards):
        windows, info = [], []
        for k in range(per_shard):
            label = int(rng.random() < 0.6)
            n = int(rng.integers(1, 20 if label else 64))
            x = np.zeros((T, F))
            x[:n] = rng.normal(0.5 * label, 1.0, size=(n, F))
            mask = [1] * n + [0] * (T - n)
            wid = f"{split}-{s}-{k}"
            windows.append({"id": wid, "x": x.tolist(), "mask": mask})
            info.append(
                {
                    "id": wid,
                    "sender": int(rng.integers(0, 30)),
                    "n_messages": n,
                    "is_attack": label,
                    "label": 16 * label,
                    "label_name": "GridSybil" if label else "Benign",
                }
            )
        name = f"{split}/b64/part-{s:05d}"
        (root / f"{name}.json").write_text(json.dumps({"windows": windows}))
        (root / f"{name}_info.json").write_text(json.dumps({"split": split, "windows": info}))
        entries.append({"split": split, "file": f"{name}.json", "info": f"{name}_info.json"})
    return entries


@pytest.fixture()
def t64_dir(tmp_path: Path) -> Path:
    """Synthetic T64 folder with 2 train shards and 1 test shard."""
    rng = np.random.default_rng(0)
    root = tmp_path / "T64"
    shards = write_split(root, "train", 2, 150, rng) + write_split(root, "test", 1, 120, rng)
    meta = {"features": [f"f{i}" for i in range(F)], "shards": shards}
    (root / "metadata.json").write_text(json.dumps(meta))
    return root


def test_window_stats_uses_real_rows_only():
    x = np.zeros((2, 4, 2))
    x[0, :2] = [[1.0, 10.0], [3.0, 30.0]]
    x[0, 2:] = 999.0  # padding values must be ignored
    mask = np.array([[1, 1, 0, 0], [0, 0, 0, 0]])
    s = window_stats(x, mask)
    assert s.shape == (2, 4 * 2 + 1)
    np.testing.assert_allclose(s[0], [2, 20, 1, 10, 1, 10, 3, 30, 2])
    np.testing.assert_allclose(s[1], 0.0)


def test_reader_checks_ids_and_lengths(t64_dir: Path):
    table = ShardReader(t64_dir).read("train")
    assert len(table.ids) == 300
    assert table.stats.shape == (300, 4 * F + 1)
    np.testing.assert_array_equal(table.stats[:, -1], table.n_messages)


def test_reader_rejects_id_mismatch(t64_dir: Path):
    info_path = t64_dir / "test/b64/part-00000_info.json"
    info = json.loads(info_path.read_text())
    info["windows"][0]["id"] = "other"
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="id mismatch"):
        ShardReader(t64_dir).read("test")


def test_length_direction():
    n = np.array([1, 2, 3, 10, 20, 30])
    y = np.array([1, 1, 1, 0, 0, 0])
    base = LengthBaseline()
    info = base.choose_direction(n, y)
    assert base.direction == -1
    assert info["auroc_minus_n"] == pytest.approx(1.0)
    assert np.all(base.scores(n) == -n)


def test_runner_writes_contract_files(t64_dir: Path, tmp_path: Path):
    out = tmp_path / "baselines"
    summary = BaselineRunner(BaselineConfig(data_dir=str(t64_dir), out_dir=str(out))).run()
    for name in ("scores_test.npz", "config.json", "env.json", "summary.json"):
        assert (out / name).exists()
    scores = np.load(out / "scores_test.npz")
    assert set(scores.files) == {"id", "label", "n_messages", "sender", "length_score", "stats_prob"}
    assert len(scores["id"]) == 120
    assert summary["stats_lr"]["test_auroc"] > 0.8  # synthetic classes are separable by design
    assert summary["length"]["direction"].startswith("-n")
