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
