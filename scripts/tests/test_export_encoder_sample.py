"""Unit tests for the encoder-input sample exporter (small synthetic shards, no access to src/data/)."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import export_encoder_sample as exp  # noqa: E402

_FEATURES = [
    "claimed_pos_x",
    "claimed_pos_y",
    "rx_pos_x",
    "rx_pos_y",
    "claimed_vel_x",
    "claimed_vel_y",
    "rx_vel_x",
    "rx_vel_y",
    "claimed_acl_x",
    "claimed_acl_y",
    "range",
    "bearing",
    "log_dtau",
]
_RANGE = _FEATURES.index("range")
_RANGE_MEAN, _RANGE_STD = 100.0, 50.0


def _links() -> list[dict]:
    """Synthetic links: (split, scenario, receiver, sender, pseudo, label, lengths of crops)."""
    spec = []
    rng = np.random.default_rng(1)
    split_of_sender = {}
    for scenario in ("GridSybil_0709", "GridSybil_1416"):
        for s_idx in range(24):
            sender = 100 + s_idx
            split = exp.SPLITS[s_idx % 4]
            split_of_sender[(scenario, sender)] = split
            attack = s_idx % 3 == 0
            for receiver in (1, 2):
                pseudo = 1 if attack else 1000 + sender
                n_total = int(rng.integers(1, 150))
                lens = [64] * (n_total // 64) + ([n_total % 64] if n_total % 64 else [])
                spec.append(
                    dict(
                        split=split,
                        scenario=scenario,
                        receiver=receiver,
                        sender=sender,
                        pseudo=pseudo,
                        label=16 if attack else 0,
                        lens=lens,
                    )
                )
    # One long benign test link (5 crops) and one test window of n = 1.
    spec.append(
        dict(
            split="test",
            scenario="GridSybil_1416",
            receiver=3,
            sender=103,
            pseudo=2103,
            label=0,
            lens=[64, 64, 64, 64, 10],
        )
    )
    spec.append(dict(split="test", scenario="GridSybil_1416", receiver=3, sender=107, pseudo=2107, label=0, lens=[1]))
    return spec


def _write_source(root: pathlib.Path, far_range_z: float = 200.0) -> pathlib.Path:
    """Writes metadata.json + one shard per split, mirroring the encoder-input notebook layout."""
    src = root / "T64"
    rng = np.random.default_rng(2)
    per_split: dict[str, tuple[list, list]] = {s: ([], []) for s in exp.SPLITS}
    total_windows = 0
    far_done = False
    for link in _links():
        run = f"{link['scenario']}/VeReMi_run"
        rfile = f"{run}/traceJSON-{link['receiver']}-0-A0-0-7.json"
        for k, n in enumerate(link["lens"]):
            x = np.zeros((64, 13))
            x[:n] = np.round(rng.normal(size=(n, 13)), 4)
            if link["label"] == 16 and link["split"] == "val" and not far_done:
                x[0, _RANGE] = far_range_z
                far_done = True
            mask = [1] * n + [0] * (64 - n)
            wid = f"{rfile}#{link['pseudo']}-{link['sender']}#{k}"
            per_split[link["split"]][0].append({"id": wid, "x": x.tolist(), "mask": mask})
            per_split[link["split"]][1].append(
                {
                    "id": wid,
                    "scenario": link["scenario"],
                    "run": run,
                    "receiver_file": rfile,
                    "receiver": link["receiver"],
                    "sender": link["sender"],
                    "sender_pseudo": link["pseudo"],
                    "link_window_index": k,
                    "n_messages": n,
                    "bucket": 64,
                    "label": link["label"],
                    "label_name": "GridSybil" if link["label"] == 16 else "Benign",
                    "is_attack": int(link["label"] == 16),
                }
            )
            total_windows += 1
    shards = []
    for split, (wins, infos) in per_split.items():
        d = src / split / "b64"
        d.mkdir(parents=True)
        (d / "part-00000.json").write_text(
            json.dumps({"seq_len": 64, "max_seq_len": 64, "features": _FEATURES, "windows": wins})
        )
        (d / "part-00000_info.json").write_text(
            json.dumps({"split": split, "bucket": 64, "shard": "part-00000", "windows": infos})
        )
        shards.append(
            {
                "split": split,
                "bucket": 64,
                "file": f"{split}/b64/part-00000.json",
                "info": f"{split}/b64/part-00000_info.json",
                "windows": len(wins),
            }
        )
    mean = [0.0] * 13
    std = [1.0] * 13
    mean[_RANGE], std[_RANGE] = _RANGE_MEAN, _RANGE_STD
    (src / "metadata.json").write_text(
        json.dumps(
            {
                "features": _FEATURES,
                "max_seq_len": 64,
                "normalisation": {"features": _FEATURES, "mean": mean, "std": std, "fitted_on": "train real rows"},
                "counts": {"windows": total_windows},
                "shards": shards,
            }
        )
    )
    return src


def _config(tmp_path: pathlib.Path, **kw) -> exp.ExportConfig:
    src = _write_source(tmp_path)
    out = tmp_path / "out" / "encoder"
    base = dict(
        source_dir=src,
        out_dir=out,
        allowed_out_root=out,
        per_stratum=6,
        max_sender_windows=400,
        budget_bytes=10_000_000,
        target_step=2,
        min_per_stratum=1,
        workers=1,
    )
    base.update(kw)
    return exp.ExportConfig(**base)


def test_sampling_is_deterministic(tmp_path: pathlib.Path) -> None:
    cfg = _config(tmp_path)
    index = exp.DatasetIndex(exp.EncoderInput(cfg.source_dir).all_records())
    a, _ = exp.StratifiedSampler(index, 0, 400).sample(6)
    b, _ = exp.StratifiedSampler(index, 0, 400).sample(6)
    c, _ = exp.StratifiedSampler(index, 5, 400).sample(6)
    assert [r.id for r in a] == [r.id for r in b]
    assert {r.id for r in a} != {r.id for r in c}


def test_sampler_takes_whole_senders_and_skips_large_ones(tmp_path: pathlib.Path) -> None:
    cfg = _config(tmp_path)
    index = exp.DatasetIndex(exp.EncoderInput(cfg.source_dir).all_records())
    sample, stats = exp.StratifiedSampler(index, 0, 400).sample(6)
    picked = {(r.split, r.scenario, r.label_name, r.sender_key) for r in sample}
    for split, scenario, cls, sender in picked:
        full = [
            r
            for r in index.records
            if (r.split, r.scenario, r.label_name, r.sender_key) == (split, scenario, cls, sender)
        ]
        assert {r.id for r in full} <= {r.id for r in sample}
    _, stats_small = exp.StratifiedSampler(index, 0, 1).sample(6)
    assert sum(s["skipped_large_senders"] for s in stats_small.values()) > 0


def test_export_bundle_is_consistent(tmp_path: pathlib.Path) -> None:
    cfg = _config(tmp_path)
    manifest = exp.Exporter(cfg).run()
    out = cfg.out_dir
    x = np.fromfile(out / "x.f32", dtype="<f4").reshape(-1, 13)
    ws = manifest["windows"]
    assert sum(w["n"] for w in ws) == x.shape[0] == manifest["sample"]["n_rows"]
    assert all(w["row"] == sum(v["n"] for v in ws[:i]) for i, w in enumerate(ws))
    assert ws == sorted(
        ws,
        key=lambda w: (
            exp.SPLITS.index(w["split"]),
            w["scenario"],
            w["run"],
            w["receiver_file"],
            w["sender"],
            w["sender_pseudo"],
            w["k"],
        ),
    )
    # Every crop of each sampled link is present.
    by_link: dict[str, set[int]] = {}
    for w in ws:
        by_link.setdefault(w["link"], set()).add(w["k"])
        assert w["K"] >= 1 and 0 <= w["k"] < w["K"]
    for link, ks in by_link.items():
        assert ks == set(range(next(w["K"] for w in ws if w["link"] == link)))
    # Rows decode back to the shard values.
    shard = json.loads((cfg.source_dir / "test/b64/part-00000.json").read_text())
    src = {w["id"]: np.asarray(w["x"], dtype=np.float32) for w in shard["windows"]}
    for w in ws:
        if w["split"] == "test":
            assert np.array_equal(x[w["row"] : w["row"] + w["n"]], src[w["id"]][: w["n"]])
    assert json.loads((out / "predictions/index.json").read_text()) == {"models": []}
    assert manifest["units"][_RANGE] == "m"


def test_sender_splits_full(tmp_path: pathlib.Path) -> None:
    cfg = _config(tmp_path)
    manifest = exp.Exporter(cfg).run()
    assert manifest["full_dataset"]["senders_in_multiple_splits"] == 0
    assert all(w["sender_splits_full"] == [w["split"]] for w in manifest["windows"])
    records = exp.EncoderInput(cfg.source_dir).all_records()
    leaked = dataclasses.replace(
        records[0], split="val" if records[0].split != "val" else "test", id="leak#0#99", sender_pseudo=999999, k=0
    )
    index = exp.DatasetIndex(records + [leaked])
    assert index.senders_in_multiple_splits == 1
    assert len(index.splits_of_sender(records[0])) == 2


def test_curated_rules(tmp_path: pathlib.Path) -> None:
    cfg = _config(tmp_path)
    manifest = exp.Exporter(cfg).run()
    cur = {c["name"].split(" (")[0]: c for c in manifest["curated"]}
    by_id = {w["id"]: w for w in manifest["windows"]}
    far = by_id[cur["Farthest claimed position"]["window_id"]]
    assert far["split"] == "val" and far["label_name"] == "GridSybil"
    assert "10.1 km" in [c["name"] for c in manifest["curated"]][0]  # 200 * 50 + 100 m
    longest = by_id[cur["Longest link, cropped"]["window_id"]]
    assert longest["K"] == 5 and longest["k"] == 0
    assert sum(1 for w in manifest["windows"] if w["link"] == longest["link"]) == 5
    single = by_id[cur["Single-message window"]["window_id"]]
    assert single["n"] == 1 and single["split"] == "test"
    ghost = by_id[cur["Ghost under shared pseudonym 1"]["window_id"]]
    assert ghost["sender_pseudo"] == 1 and ghost["n"] >= 20 and ghost["label_name"] == "GridSybil"
    pair = [by_id[c["window_id"]] for c in manifest["curated"] if c["name"].startswith("Benign and GridSybil")]
    assert {w["label_name"] for w in pair} == {"Benign", "GridSybil"}
    assert len({w["receiver_file"] for w in pair}) == 1 and pair[0]["split"] == "test"


def test_budget_lowers_target(tmp_path: pathlib.Path) -> None:
    roomy = exp.Exporter(_config(tmp_path / "a")).run()
    full_size = len(exp.encode_manifest(roomy)) + roomy["sample"]["n_rows"] * 52
    tight = exp.Exporter(_config(tmp_path / "b", budget_bytes=int(full_size * 0.7))).run()
    assert tight["sample"]["per_stratum_target"] < roomy["sample"]["per_stratum_target"]
    out = tmp_path / "b" / "out" / "encoder"
    assert (out / "x.f32").stat().st_size + (out / "manifest.json").stat().st_size <= int(full_size * 0.7)
    with pytest.raises(exp.ExportError):
        exp.Exporter(_config(tmp_path / "c", budget_bytes=100)).run()


def test_writer_refuses_outside_root(tmp_path: pathlib.Path) -> None:
    with pytest.raises(exp.ExportError):
        exp.BundleWriter(tmp_path / "elsewhere", tmp_path / "out" / "encoder")
    with pytest.raises(exp.ExportError):
        exp.BundleWriter(tmp_path / "out" / "encoder" / ".." / ".." / "x", tmp_path / "out" / "encoder")
