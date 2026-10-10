"""Unit tests for the Hugging Face Hub transfer CLI (fake HfApi / downloads, no network, no token)."""

from __future__ import annotations

import json
import pathlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import hf_hub  # noqa: E402
from huggingface_hub.errors import LocalTokenNotFoundError  # noqa: E402


class FakeApi:
    """Records calls; stores uploaded files in memory as {repo_path: bytes}."""

    def __init__(self, logged_in: bool = True, existing: dict[str, bytes] | None = None) -> None:
        self.logged_in = logged_in
        self.files: dict[str, bytes] = dict(existing or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def whoami(self) -> dict[str, str]:
        if not self.logged_in:
            raise LocalTokenNotFoundError("no token")
        return {"name": "tester"}

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))

    def repo_info(self, repo_id: str, **kwargs: Any) -> dict:
        return {"id": repo_id}

    def list_repo_files(self, repo_id: str, **kwargs: Any) -> list[str]:
        return sorted(self.files)

    def upload_folder(self, **kwargs: Any) -> None:
        self.calls.append(("upload_folder", kwargs))
        root = pathlib.Path(kwargs["folder_path"])
        for path in hf_hub.iter_tree(root):
            self.files[path.relative_to(root).as_posix()] = path.read_bytes()

    def create_commit(self, **kwargs: Any) -> Any:
        self.calls.append(("create_commit", kwargs))
        for op in kwargs["operations"]:
            if not hasattr(op, "path_or_fileobj"):  # CommitOperationDelete
                self.files.pop(op.path_in_repo, None)
                continue
            src = op.path_or_fileobj
            data = src if isinstance(src, bytes) else pathlib.Path(src).read_bytes()
            self.files[op.path_in_repo] = data

        class Info:
            oid = "abc123"

        return Info()


def make_hub(api: FakeApi, snapshot: Any = None, single: Any = None) -> hf_hub.Hub:
    return hf_hub.Hub(
        api=api, snapshot_download=snapshot or (lambda *a, **k: ""), hf_hub_download=single or (lambda *a, **k: "")
    )


def fake_snapshot_into(api: FakeApi):
    """snapshot_download that writes the fake repo's files (honouring allow/ignore patterns used here)."""

    def _download(repo_id: str, *, local_dir: str | None = None, cache_dir: str | None = None, **kw: Any) -> str:
        root = pathlib.Path(local_dir or pathlib.Path(cache_dir) / "snap")
        ignore = set(kw.get("ignore_patterns") or [])
        allow = [p.rstrip("*") for p in kw.get("allow_patterns") or []]
        for rel, data in api.files.items():
            if rel in ignore or (allow and not any(rel.startswith(a) for a in allow)):
                continue
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (root / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        return str(root)

    return _download


def fake_single(api: FakeApi):
    def _download(repo_id: str, filename: str, *, local_dir: str, **kw: Any) -> str:
        target = pathlib.Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(api.files[filename])
        return str(target)

    return _download


@pytest.fixture
def veremi(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "VeReMi-Dataset"
    for scen in ("GridSybil_0709", "DoSRandomSybil_1416"):
        run = root / scen / "run1"
        run.mkdir(parents=True)
        (run / "traceJSON-1-0-A0-0-0.json").write_text(f'{{"scen": "{scen}"}}\n')
        (run / "traceGroundTruthJSON-7.json").write_text("[]\n")
        (root / scen / ".DS_Store").write_bytes(b"junk")
    (root / ".DS_Store").write_bytes(b"junk")
    return root


@pytest.fixture
def run_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    run = tmp_path / "runs" / "20261008-104514-joint"
    run.mkdir(parents=True)
    (run / "config.json").write_text('{"seed": 0}')
    (run / "env.json").write_text('{"torch": "2.14.1"}')
    (run / "metrics.jsonl").write_text('{"step": 1}\n')
    (run / "best.pt").write_bytes(b"best-weights")
    (run / "last.pt").write_bytes(b"last-weights")
    return run


# ---- repo-id resolution ----------------------------------------------------------------------------------------


def test_repo_ids_default() -> None:
    cfg = hf_hub.HubConfig.resolve(env={})
    assert (cfg.dataset_repo, cfg.model_repo) == (hf_hub.DEFAULT_DATASET_REPO, hf_hub.DEFAULT_MODEL_REPO)


def test_repo_ids_env_over_default() -> None:
    cfg = hf_hub.HubConfig.resolve(env={"HF_DATASET_REPO": "u/d", "HF_MODEL_REPO": "u/m"})
    assert (cfg.dataset_repo, cfg.model_repo) == ("u/d", "u/m")


def test_repo_ids_flag_over_env() -> None:
    cfg = hf_hub.HubConfig.resolve("f/d", "f/m", env={"HF_DATASET_REPO": "u/d", "HF_MODEL_REPO": "u/m"})
    assert (cfg.dataset_repo, cfg.model_repo) == ("f/d", "f/m")
    assert cfg.dataset_url() == "https://huggingface.co/datasets/f/d"
    assert cfg.model_url() == "https://huggingface.co/f/m"


# ---- manifest / sums -------------------------------------------------------------------------------------------


def test_manifest_skips_clutter_and_hashes(veremi: pathlib.Path) -> None:
    manifest = hf_hub.build_manifest(veremi)
    assert sorted(manifest) == [
        "DoSRandomSybil_1416/run1/traceGroundTruthJSON-7.json",
        "DoSRandomSybil_1416/run1/traceJSON-1-0-A0-0-0.json",
        "GridSybil_0709/run1/traceGroundTruthJSON-7.json",
        "GridSybil_0709/run1/traceJSON-1-0-A0-0-0.json",
    ]
    entry = manifest["GridSybil_0709/run1/traceGroundTruthJSON-7.json"]
    assert entry["size"] == 3 and len(entry["sha256"]) == 64
    assert hf_hub.verify_manifest(veremi, manifest) == []


def test_verify_manifest_reports_problems(veremi: pathlib.Path) -> None:
    manifest = hf_hub.build_manifest(veremi)
    target = veremi / "GridSybil_0709/run1/traceJSON-1-0-A0-0-0.json"
    target.write_text(target.read_text().replace("Grid", "Xrid"))  # same size, different content
    (veremi / "extra.json").write_text("{}")
    problems = hf_hub.verify_manifest(veremi, manifest)
    assert any("sha256 mismatch" in p for p in problems)
    assert any("unexpected" in p for p in problems)


def test_sums_round_trip() -> None:
    sums = {"best.pt": "a" * 64, "config.json": "b" * 64}
    text = hf_hub.format_sums(sums)
    assert text.splitlines()[0] == "a" * 64 + "  best.pt"
    assert hf_hub.parse_sums(text) == sums


# ---- dataset ---------------------------------------------------------------------------------------------------


def test_upload_dataset_private_manifest_and_card(veremi: pathlib.Path) -> None:
    api = FakeApi()
    before = sorted(p.relative_to(veremi) for p in veremi.rglob("*"))
    cfg = hf_hub.HubConfig.resolve(env={})
    url = hf_hub.DatasetTransfer(cfg, make_hub(api), log=lambda m: None).upload(veremi)
    assert url == f"https://huggingface.co/datasets/{hf_hub.DEFAULT_DATASET_REPO}"
    create = [kw for name, kw in api.calls if name == "create_repo"][0]
    assert create["private"] is True and create["exist_ok"] is True and create["repo_type"] == "dataset"
    data_upload = [kw for name, kw in api.calls if name == "upload_folder"][0]
    assert ".DS_Store" in data_upload["ignore_patterns"] and data_upload["repo_type"] == "dataset"
    manifest = json.loads(api.files[hf_hub.MANIFEST_NAME])
    assert manifest["n_files"] == 4 and set(manifest["files"]) == set(hf_hub.build_manifest(veremi))
    assert "VeReMi-Extension" in api.files[hf_hub.CARD_NAME].decode()
    assert sorted(p.relative_to(veremi) for p in veremi.rglob("*")) == before  # nothing written into data/


def test_upload_dataset_keeps_existing_card(veremi: pathlib.Path) -> None:
    api = FakeApi(existing={hf_hub.CARD_NAME: b"custom card"})
    hf_hub.DatasetTransfer(hf_hub.HubConfig(), make_hub(api), log=lambda m: None).upload(veremi)
    assert api.files[hf_hub.CARD_NAME] == b"custom card"


def test_download_dataset_refuses_non_empty(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "VeReMi-Dataset"
    target.mkdir()
    (target / "keep.json").write_text("{}")
    with pytest.raises(hf_hub.HubError, match="--force"):
        hf_hub.DatasetTransfer(hf_hub.HubConfig(), make_hub(FakeApi()), log=lambda m: None).download(target)
    assert (target / "keep.json").read_text() == "{}"


def test_download_dataset_same_tree_and_verified(veremi: pathlib.Path, tmp_path: pathlib.Path) -> None:
    api = FakeApi()
    hf_hub.DatasetTransfer(hf_hub.HubConfig(), make_hub(api), log=lambda m: None).upload(veremi)
    target = tmp_path / "fresh" / "VeReMi-Dataset"
    hub = make_hub(api, snapshot=fake_snapshot_into(api), single=fake_single(api))
    hf_hub.DatasetTransfer(hf_hub.HubConfig(), hub, log=lambda m: None).download(target)
    got = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())
    assert got == sorted(hf_hub.build_manifest(veremi))  # same tree, no README / MANIFEST / .cache left
    assert (target / "GridSybil_0709/run1/traceJSON-1-0-A0-0-0.json").read_text() == '{"scen": "GridSybil_0709"}\n'


def test_download_dataset_detects_corruption(veremi: pathlib.Path, tmp_path: pathlib.Path) -> None:
    api = FakeApi()
    hf_hub.DatasetTransfer(hf_hub.HubConfig(), make_hub(api), log=lambda m: None).upload(veremi)
    api.files["GridSybil_0709/run1/traceGroundTruthJSON-7.json"] = b"{}\n"  # same size, other bytes
    hub = make_hub(api, snapshot=fake_snapshot_into(api), single=fake_single(api))
    with pytest.raises(hf_hub.HubError, match="sha256 mismatch"):
        hf_hub.DatasetTransfer(hf_hub.HubConfig(), hub, log=lambda m: None).download(tmp_path / "out")


def test_not_logged_in_is_clear(veremi: pathlib.Path) -> None:
    with pytest.raises(hf_hub.HubError, match="hf auth login"):
        hf_hub.DatasetTransfer(hf_hub.HubConfig(), make_hub(FakeApi(logged_in=False)), log=lambda m: None).upload(
            veremi
        )


def test_missing_repo_is_clear(tmp_path: pathlib.Path) -> None:
    from huggingface_hub.errors import RepositoryNotFoundError

    api = FakeApi()

    def missing(repo_id: str, **kw: Any) -> None:
        response = SimpleNamespace(headers={}, status_code=404, request=None)
        raise RepositoryNotFoundError("nope", response=response)

    api.repo_info = missing  # type: ignore[method-assign]
    with pytest.raises(hf_hub.HubError, match="not found"):
        hf_hub.ModelTransfer(hf_hub.HubConfig(), make_hub(api), log=lambda m: None).list_runs()


# ---- model -----------------------------------------------------------------------------------------------------


def test_select_files_best_and_logs(run_dir: pathlib.Path) -> None:
    (run_dir / "train.log").write_text("log")
    files, note = hf_hub.ModelTransfer.select_files(run_dir)
    assert [p.name for p in files] == ["config.json", "env.json", "metrics.jsonl", "best.pt", "train.log"]
    assert note is None
    files, _ = hf_hub.ModelTransfer.select_files(run_dir, include_last=True)
    assert "last.pt" in [p.name for p in files]


def test_select_files_falls_back_to_last(run_dir: pathlib.Path) -> None:
    (run_dir / "best.pt").unlink()
    files, note = hf_hub.ModelTransfer.select_files(run_dir)
    assert "last.pt" in [p.name for p in files] and "best.pt is missing" in note


def test_select_files_refuses_without_checkpoint(run_dir: pathlib.Path) -> None:
    (run_dir / "best.pt").unlink()
    (run_dir / "last.pt").unlink()
    with pytest.raises(hf_hub.HubError, match="neither best.pt nor last.pt"):
        hf_hub.ModelTransfer.select_files(run_dir)


def test_upload_model_layout_and_sums(run_dir: pathlib.Path) -> None:
    api = FakeApi()
    url, commit = hf_hub.ModelTransfer(hf_hub.HubConfig(), make_hub(api), log=lambda m: None).upload(
        run_dir, include_last=True
    )
    assert commit == "abc123" and url.endswith(hf_hub.DEFAULT_MODEL_REPO)
    prefix = "runs/20261008-104514-joint/"
    assert {f for f in api.files if f.startswith(prefix)} == {
        prefix + n for n in ("config.json", "env.json", "metrics.jsonl", "best.pt", "last.pt", "SHA256SUMS")
    }
    sums = hf_hub.parse_sums(api.files[prefix + "SHA256SUMS"].decode())
    assert sums["best.pt"] == hf_hub.sha256_file(run_dir / "best.pt")
    assert hf_hub.CARD_NAME in api.files
    create = [kw for name, kw in api.calls if name == "create_repo"][0]
    assert create["private"] is True and create["repo_type"] == "model"


def test_upload_model_name_override(run_dir: pathlib.Path) -> None:
    api = FakeApi()
    hf_hub.ModelTransfer(hf_hub.HubConfig(), make_hub(api), log=lambda m: None).upload(run_dir, name="seed0")
    assert "runs/seed0/best.pt" in api.files and "runs/seed0/last.pt" not in api.files


def test_list_runs(run_dir: pathlib.Path) -> None:
    api = FakeApi()
    transfer = hf_hub.ModelTransfer(hf_hub.HubConfig(), make_hub(api), log=lambda m: None)
    transfer.upload(run_dir)
    transfer.upload(run_dir, name="b-run")
    assert transfer.list_runs() == ["20261008-104514-joint", "b-run"]


def test_download_model_round_trip(run_dir: pathlib.Path, tmp_path: pathlib.Path) -> None:
    api = FakeApi()
    hf_hub.ModelTransfer(hf_hub.HubConfig(), make_hub(api), log=lambda m: None).upload(run_dir, include_last=True)
    hub = make_hub(api, snapshot=fake_snapshot_into(api))
    dest = hf_hub.ModelTransfer(hf_hub.HubConfig(), hub, log=lambda m: None).download(
        "20261008-104514-joint", to=tmp_path / "local"
    )
    assert dest == tmp_path / "local" / "20261008-104514-joint"
    assert (dest / "best.pt").read_bytes() == b"best-weights"
    assert (dest / "last.pt").read_bytes() == b"last-weights"
    assert (dest / "config.json").is_file() and (dest / "SHA256SUMS").is_file()


def test_download_model_refuses_overwrite_then_force(run_dir: pathlib.Path, tmp_path: pathlib.Path) -> None:
    api = FakeApi()
    hf_hub.ModelTransfer(hf_hub.HubConfig(), make_hub(api), log=lambda m: None).upload(run_dir)
    hub = make_hub(api, snapshot=fake_snapshot_into(api))
    transfer = hf_hub.ModelTransfer(hf_hub.HubConfig(), hub, log=lambda m: None)
    dest = tmp_path / "local" / "20261008-104514-joint"
    dest.mkdir(parents=True)
    (dest / "best.pt").write_bytes(b"old")
    with pytest.raises(hf_hub.HubError, match="--force"):
        transfer.download("20261008-104514-joint", to=tmp_path / "local")
    assert (dest / "best.pt").read_bytes() == b"old"
    transfer.download("20261008-104514-joint", to=tmp_path / "local", force=True)
    assert (dest / "best.pt").read_bytes() == b"best-weights"


def test_download_model_checksum_mismatch(run_dir: pathlib.Path, tmp_path: pathlib.Path) -> None:
    api = FakeApi()
    hf_hub.ModelTransfer(hf_hub.HubConfig(), make_hub(api), log=lambda m: None).upload(run_dir)
    api.files["runs/20261008-104514-joint/best.pt"] = b"tampered"
    hub = make_hub(api, snapshot=fake_snapshot_into(api))
    with pytest.raises(hf_hub.HubError, match="sha256 mismatch for best.pt"):
        hf_hub.ModelTransfer(hf_hub.HubConfig(), hub, log=lambda m: None).download(
            "20261008-104514-joint", to=tmp_path / "local"
        )
    assert not (tmp_path / "local" / "20261008-104514-joint").exists()


def test_download_model_unknown_run(tmp_path: pathlib.Path) -> None:
    api = FakeApi()
    hub = make_hub(api, snapshot=fake_snapshot_into(api))
    with pytest.raises(hf_hub.HubError, match="not found"):
        hf_hub.ModelTransfer(hf_hub.HubConfig(), hub, log=lambda m: None).download("nope", to=tmp_path)


@pytest.mark.parametrize(
    "name, runs_dir",
    [("model-all", "src/runs/pretraining/all/T24"), ("model-grid", "src/runs/pretraining/benign_gridsybil/T64")],
)
def test_named_run_upload_download_cli_round_trip(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], name: str, runs_dir: str
) -> None:
    """The npm `model:upload:<dataset>` / `model:download:<dataset>` commands, end to end on a fake hub."""
    local = tmp_path / "src-side" / runs_dir / name
    local.mkdir(parents=True)
    (local / "config.json").write_text(json.dumps({"seed": 0, "dataset": name.split("-")[1]}))
    (local / "env.json").write_text('{"torch": "2.14.1"}')
    (local / "metrics.jsonl").write_text('{"step": 1}\n')
    (local / "best.pt").write_bytes(b"best-" + name.encode())
    (local / "last.pt").write_bytes(b"last-" + name.encode())
    api = FakeApi()
    hub = make_hub(api, snapshot=fake_snapshot_into(api))
    assert hf_hub.main(["upload-model", str(local), "--name", name, "--include-last"], hub=hub) == 0
    prefix = f"runs/{name}/"
    assert {f for f in api.files if f.startswith(prefix)} == {
        prefix + n for n in ("config.json", "env.json", "metrics.jsonl", "best.pt", "last.pt", "SHA256SUMS")
    }
    assert hf_hub.main(["list-models"], hub=hub) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == name
    target = tmp_path / "fresh" / runs_dir
    assert hf_hub.main(["download-model", name, "--to", str(target)], hub=hub) == 0
    dest = target / name
    for n in ("config.json", "env.json", "metrics.jsonl", "best.pt", "last.pt"):
        assert (dest / n).read_bytes() == (local / n).read_bytes()
    assert hf_hub.parse_sums((dest / "SHA256SUMS").read_text())["last.pt"] == hf_hub.sha256_file(local / "last.pt")


def test_reupload_same_name_replaces_folder(run_dir: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """A second upload under the same run id leaves exactly the new files (stale last.pt deleted, sums match)."""
    api = FakeApi()
    logs: list[str] = []
    transfer = hf_hub.ModelTransfer(hf_hub.HubConfig(), make_hub(api), log=logs.append)
    transfer.upload(run_dir, name="model-all", include_last=True)
    (run_dir / "best.pt").write_bytes(b"better-weights")
    transfer.upload(run_dir, name="model-all")
    prefix = "runs/model-all/"
    assert {f for f in api.files if f.startswith(prefix)} == {
        prefix + n for n in ("config.json", "env.json", "metrics.jsonl", "best.pt", "SHA256SUMS")
    }
    assert api.files[prefix + "best.pt"] == b"better-weights"
    assert any("replacing it (deleting 1 stale file(s))" in m for m in logs)
    commits = [kw for name, kw in api.calls if name == "create_commit"]
    assert len(commits) == 2  # the replacement is one atomic commit
    hub = make_hub(api, snapshot=fake_snapshot_into(api))
    dest = tmp_path / "local" / "model-all"
    dest.mkdir(parents=True)
    (dest / "last.pt").write_bytes(b"old-last")
    hf_hub.ModelTransfer(hf_hub.HubConfig(), hub, log=lambda m: None).download(
        "model-all", to=tmp_path / "local", force=True
    )
    assert (dest / "best.pt").read_bytes() == b"better-weights" and not (dest / "last.pt").exists()


def test_upload_model_includes_run_readme(run_dir: pathlib.Path) -> None:
    (run_dir / "README.md").write_text("F14 open")
    files, _ = hf_hub.ModelTransfer.select_files(run_dir)
    assert "README.md" in [p.name for p in files]


# ---- CLI -------------------------------------------------------------------------------------------------------


def test_cli_whoami_and_error_exit(capsys: pytest.CaptureFixture[str]) -> None:
    assert hf_hub.main(["whoami"], hub=make_hub(FakeApi())) == 0
    assert capsys.readouterr().out.strip() == "tester"
    assert hf_hub.main(["whoami"], hub=make_hub(FakeApi(logged_in=False))) == 1
    assert "hf auth login" in capsys.readouterr().err


def test_cli_defaults() -> None:
    args = hf_hub.parse_args(["download-model", "r1"])
    assert args.to == hf_hub.DEFAULT_RUNS_DIR and args.force is False
    args = hf_hub.parse_args(["--model-repo", "x/y", "upload-model", "some/dir", "--include-last"])
    assert args.model_repo == "x/y" and args.include_last and args.run_dir == pathlib.Path("some/dir")


# ---- encoder input ---------------------------------------------------------------------------------------------


@pytest.fixture
def encoder_input(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "encoder_input" / "benign_gridsybil" / "T64"
    for split in ("train", "test"):
        shard = root / split / "b64"
        shard.mkdir(parents=True)
        (shard / "part-00000.json").write_text(f'[{{"id": "{split}-0"}}]\n')
        (shard / "part-00000_info.json").write_text('{"labels": [0]}\n')
    (root / "metadata.json").write_text('{"T": 64}\n')
    (root / ".DS_Store").write_bytes(b"junk")
    return root


def input_transfer(api: FakeApi, download: bool = False) -> hf_hub.EncoderInputTransfer:
    hub = make_hub(api, snapshot=fake_snapshot_into(api), single=fake_single(api)) if download else make_hub(api)
    return hf_hub.EncoderInputTransfer(hf_hub.HubConfig(), hub, log=lambda m: None)


def tree(root: pathlib.Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_input_repo_id_resolution() -> None:
    assert hf_hub.HubConfig.resolve(env={}).input_repo == "bibekgupta3333/roadfm-lite-encoder-input"
    assert hf_hub.HubConfig.resolve(env={"HF_INPUT_REPO": "u/i"}).input_repo == "u/i"
    cfg = hf_hub.HubConfig.resolve(env={"HF_INPUT_REPO": "u/i"}, input_repo="f/i")
    assert cfg.input_repo == "f/i" and cfg.input_url() == "https://huggingface.co/datasets/f/i"


def test_upload_encoder_input_layout(encoder_input: pathlib.Path) -> None:
    api = FakeApi()
    before = tree(encoder_input)
    url = input_transfer(api).upload(encoder_input)
    assert url == f"https://huggingface.co/datasets/{hf_hub.DEFAULT_INPUT_REPO}"
    create = [kw for name, kw in api.calls if name == "create_repo"][0]
    assert create["private"] is True and create["exist_ok"] is True and create["repo_type"] == "dataset"
    assert create["repo_id"] == hf_hub.DEFAULT_INPUT_REPO
    data_upload = [kw for name, kw in api.calls if name == "upload_folder"][0]
    assert data_upload["path_in_repo"] == "benign_gridsybil/T64"
    manifest = json.loads(api.files["manifests/benign_gridsybil/T64.json"])
    assert set(manifest["files"]) == {
        "metadata.json",
        "train/b64/part-00000.json",
        "train/b64/part-00000_info.json",
        "test/b64/part-00000.json",
        "test/b64/part-00000_info.json",
    }
    assert "encoder input" in api.files[hf_hub.CARD_NAME].decode()
    assert tree(encoder_input) == before  # the local tree is only read


class PrefixApi(FakeApi):
    """FakeApi whose upload_folder honours `path_in_repo` (the encoder input sits under a prefix)."""

    def upload_folder(self, **kwargs: Any) -> None:
        self.calls.append(("upload_folder", kwargs))
        root = pathlib.Path(kwargs["folder_path"])
        prefix = kwargs.get("path_in_repo")
        for path in hf_hub.iter_tree(root):
            rel = path.relative_to(root).as_posix()
            self.files[f"{prefix}/{rel}" if prefix else rel] = path.read_bytes()


def test_download_encoder_input_restores_exactly(encoder_input: pathlib.Path, tmp_path: pathlib.Path) -> None:
    api = PrefixApi()
    input_transfer(api).upload(encoder_input)
    target = tmp_path / "fresh" / "encoder_input" / "benign_gridsybil" / "T64"
    assert input_transfer(api, download=True).download(target) == target
    expected = {k: v for k, v in tree(encoder_input).items() if not k.endswith(".DS_Store")}
    assert tree(target) == expected  # same tree: no manifest, card, .cache or staging left
    assert sorted(p.name for p in target.parent.iterdir()) == ["T64"]


def test_download_encoder_input_idempotent(encoder_input: pathlib.Path, tmp_path: pathlib.Path) -> None:
    api = PrefixApi()
    input_transfer(api).upload(encoder_input)
    target = tmp_path / "T64"
    input_transfer(api, download=True).download(target)
    calls: list[str] = []
    transfer = input_transfer(api, download=True)
    transfer.hub.snapshot_download = lambda *a, **k: calls.append("snapshot")  # type: ignore[method-assign]
    transfer.download(target)  # already matches: no download, no error
    assert calls == []


def test_download_encoder_input_refuses_differing_then_force(
    encoder_input: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    api = PrefixApi()
    input_transfer(api).upload(encoder_input)
    target = tmp_path / "T64"
    target.mkdir()
    (target / "metadata.json").write_text('{"T": 32}\n')
    (target / "stale.json").write_text("{}")
    with pytest.raises(hf_hub.HubError, match="--force"):
        input_transfer(api, download=True).download(target)
    assert (target / "metadata.json").read_text() == '{"T": 32}\n'
    input_transfer(api, download=True).download(target, force=True)
    assert not (target / "stale.json").exists()
    assert (target / "metadata.json").read_text() == '{"T": 64}\n'


def test_download_encoder_input_detects_corruption(encoder_input: pathlib.Path, tmp_path: pathlib.Path) -> None:
    api = PrefixApi()
    input_transfer(api).upload(encoder_input)
    api.files["benign_gridsybil/T64/metadata.json"] = b'{"T": 32}\n'  # same size, other bytes
    target = tmp_path / "T64"
    with pytest.raises(hf_hub.HubError, match="sha256 mismatch"):
        input_transfer(api, download=True).download(target)
    assert not target.exists()


def test_download_encoder_input_checks_file_count(encoder_input: pathlib.Path, tmp_path: pathlib.Path) -> None:
    api = PrefixApi()
    input_transfer(api).upload(encoder_input)
    del api.files["benign_gridsybil/T64/test/b64/part-00000.json"]
    with pytest.raises(hf_hub.HubError, match="manifest lists 5"):
        input_transfer(api, download=True).download(tmp_path / "T64")


def test_download_encoder_input_needs_manifest(tmp_path: pathlib.Path) -> None:
    api = PrefixApi(existing={"benign_gridsybil/T64/metadata.json": b"{}"})
    with pytest.raises(hf_hub.HubError, match="upload-encoder-input"):
        input_transfer(api, download=True).download(tmp_path / "T64")


def test_cli_encoder_input_defaults() -> None:
    args = hf_hub.parse_args(["download-encoder-input"])
    assert args.input == "benign_gridsybil/T64" and args.path is None and args.all is False and args.force is False
    args = hf_hub.parse_args(["--input-repo", "x/i", "upload-encoder-input", "--path", "some/dir"])
    assert args.input_repo == "x/i" and args.path == pathlib.Path("some/dir")
    assert hf_hub.parse_args(["upload-encoder-input", "--input", "all/T24"]).input == "all/T24"
    with pytest.raises(SystemExit):
        hf_hub.parse_args(["upload-encoder-input", "--all", "--input", "all/T24"])  # one or the other


@pytest.fixture
def input_root(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Two encoder-input trees (benign_gridsybil/T64, all/T24) under a temporary INPUT_ROOT."""
    root = tmp_path / "encoder_input"
    for prefix, name in (("benign_gridsybil/T64", "part-00000.json"), ("all/T24", "part-00000.json.gz")):
        shard = root / prefix / "train"
        shard.mkdir(parents=True)
        (shard / name).write_bytes(prefix.encode())
        (root / prefix / "metadata.json").write_text(f'{{"prefix": "{prefix}"}}\n')
    (root / "notes").mkdir()  # no metadata.json: not a tree
    monkeypatch.setattr(hf_hub, "INPUT_ROOT", root)
    return root


def test_local_input_trees(input_root: pathlib.Path) -> None:
    assert hf_hub.local_input_trees(input_root) == ["all/T24", "benign_gridsybil/T64"]
    assert hf_hub.local_input_trees(input_root / "missing") == []


def test_cli_encoder_input_all_round_trip(input_root: pathlib.Path, tmp_path: pathlib.Path) -> None:
    api = PrefixApi()
    hub = make_hub(api, snapshot=fake_snapshot_into(api), single=fake_single(api))
    expected = {prefix: tree(input_root / prefix) for prefix in ("all/T24", "benign_gridsybil/T64")}
    assert hf_hub.main(["upload-encoder-input", "--all"], hub=hub) == 0
    assert {"manifests/all/T24.json", "manifests/benign_gridsybil/T64.json"} <= set(api.files)
    assert hf_hub.EncoderInputTransfer(hf_hub.HubConfig(), hub).remote_prefixes() == ["all/T24", "benign_gridsybil/T64"]
    fresh = tmp_path / "fresh"
    hf_hub.INPUT_ROOT = fresh  # restored by monkeypatch
    assert hf_hub.main(["download-encoder-input", "--all"], hub=hub) == 0
    assert {prefix: tree(fresh / prefix) for prefix in expected} == expected
    assert hf_hub.main(["download-encoder-input", "--input", "all/T24"], hub=hub) == 0  # already matches: no-op


def test_cli_encoder_input_all_rejects_path(input_root: pathlib.Path) -> None:
    assert hf_hub.main(["upload-encoder-input", "--all", "--path", "x"], hub=make_hub(PrefixApi())) == 1
