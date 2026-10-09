"""Hugging Face Hub transfers: the raw VeReMi-Extension mirror (dataset repo) and pretraining runs (model repo).

Run from the repo root with the training env (it pins `huggingface_hub` and `hf-xet`):

    .venv-train/bin/python scripts/hf_hub.py whoami
    .venv-train/bin/python scripts/hf_hub.py upload-dataset [--path data/VeReMi-Dataset]
    .venv-train/bin/python scripts/hf_hub.py download-dataset [--path data/VeReMi-Dataset] [--force]
    .venv-train/bin/python scripts/hf_hub.py upload-model <run_dir> [--name <run_id>] [--include-last]
    .venv-train/bin/python scripts/hf_hub.py download-model <run_id> [--to src/runs/pretraining/benign_gridsybil/T64]
    .venv-train/bin/python scripts/hf_hub.py list-models

Both repos are private. Repo ids resolve as flag > environment (`HF_DATASET_REPO`, `HF_MODEL_REPO`) > default.
The dataset upload never writes into `data/`: its MANIFEST.json (relative path -> size, sha256) and dataset card are
built in a temporary folder. Downloads refuse to write into a non-empty target unless `--force`, and verify what they
fetched (dataset: MANIFEST.json; model: the run's SHA256SUMS). The token is read by `huggingface_hub` itself and is
never printed.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

DEFAULT_DATASET_REPO = "bibekgupta3333/veremi-extension-raw"
DEFAULT_MODEL_REPO = "bibekgupta3333/roadfm-lite-timesnet"
DEFAULT_DATASET_PATH = Path("data/VeReMi-Dataset")
DEFAULT_RUNS_DIR = Path("src/runs/pretraining/benign_gridsybil/T64")

MANIFEST_NAME = "MANIFEST.json"
SUMS_NAME = "SHA256SUMS"
CARD_NAME = "README.md"
# Local clutter that never belongs in the mirror; `.cache` is where snapshot_download keeps its resume metadata.
IGNORED_NAMES = frozenset({".DS_Store", "Thumbs.db"})
IGNORED_DIRS = frozenset({".cache", ".git"})
UPLOAD_IGNORE_PATTERNS = [".DS_Store", "**/.DS_Store", "Thumbs.db", "**/Thumbs.db", ".cache/**", ".git/**"]
# Files that live at the dataset repo root but are not part of the VeReMi tree.
DATASET_META_FILES = (CARD_NAME, MANIFEST_NAME, ".gitattributes")
RUN_FILES = ("config.json", "env.json", "metrics.jsonl")
_CHUNK = 8 * 1024 * 1024


class HubError(RuntimeError):
    """A user-facing failure (not logged in, missing repo, overwrite refused, verification failed)."""


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in 8 MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def iter_tree(root: Path) -> Iterable[Path]:
    """Every regular file under `root` (sorted), skipping OS clutter and the Hub's local `.cache`."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in sorted(filenames):
            if name not in IGNORED_NAMES:
                yield Path(dirpath) / name


def is_non_empty_dir(path: Path) -> bool:
    """True if `path` exists and holds anything (a file counts as non-empty too)."""
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    return any(path.iterdir())


def build_manifest(root: Path, progress: Callable[[int, int], None] | None = None) -> dict[str, dict[str, Any]]:
    """Map relative POSIX path -> {"size", "sha256"} for every file under `root`."""
    files = list(iter_tree(root))
    manifest: dict[str, dict[str, Any]] = {}
    for i, path in enumerate(files, 1):
        manifest[path.relative_to(root).as_posix()] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
        if progress is not None:
            progress(i, len(files))
    return manifest


def verify_manifest(root: Path, manifest: Mapping[str, Mapping[str, Any]], check_hashes: bool = True) -> list[str]:
    """Problems found comparing the files under `root` with `manifest` (empty list = all good)."""
    problems: list[str] = []
    local = {p.relative_to(root).as_posix(): p for p in iter_tree(root)}
    missing = sorted(set(manifest) - set(local))
    extra = sorted(set(local) - set(manifest))
    if missing:
        problems.append(f"{len(missing)} file(s) missing, e.g. {missing[0]}")
    if extra:
        problems.append(f"{len(extra)} unexpected file(s), e.g. {extra[0]}")
    for rel, entry in manifest.items():
        path = local.get(rel)
        if path is None:
            continue
        if path.stat().st_size != entry["size"]:
            problems.append(f"size mismatch: {rel}")
        elif check_hashes and sha256_file(path) != entry["sha256"]:
            problems.append(f"sha256 mismatch: {rel}")
    return problems


def format_sums(sums: Mapping[str, str]) -> str:
    """`sha256sum`-compatible text: one "<hex>  <name>" line per file."""
    return "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))


def parse_sums(text: str) -> dict[str, str]:
    """Inverse of `format_sums`."""
    sums: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip().lstrip("*")] = digest
    return sums


@dataclasses.dataclass(frozen=True)
class HubConfig:
    """Which Hub repos to use.

    Attributes:
        dataset_repo: Private dataset repo holding the raw VeReMi-Extension tree.
        model_repo: Private model repo holding pretraining runs under `runs/<run_id>/`.
    """

    dataset_repo: str = DEFAULT_DATASET_REPO
    model_repo: str = DEFAULT_MODEL_REPO

    @classmethod
    def resolve(
        cls, dataset_repo: str | None = None, model_repo: str | None = None, env: Mapping[str, str] | None = None
    ) -> HubConfig:
        """Flag > environment (`HF_DATASET_REPO`, `HF_MODEL_REPO`) > default."""
        env = os.environ if env is None else env
        return cls(
            dataset_repo=dataset_repo or env.get("HF_DATASET_REPO") or DEFAULT_DATASET_REPO,
            model_repo=model_repo or env.get("HF_MODEL_REPO") or DEFAULT_MODEL_REPO,
        )

    def dataset_url(self) -> str:
        """Browser URL of the dataset repo."""
        return f"https://huggingface.co/datasets/{self.dataset_repo}"

    def model_url(self) -> str:
        """Browser URL of the model repo."""
        return f"https://huggingface.co/{self.model_repo}"


class Hub:
    """Thin wrapper over `HfApi` + download functions that turns Hub errors into clear `HubError`s.

    The download functions are injectable so tests never touch the network.
    """

    def __init__(
        self,
        api: Any | None = None,
        snapshot_download: Callable[..., str] | None = None,
        hf_hub_download: Callable[..., str] | None = None,
    ) -> None:
        if api is None or snapshot_download is None or hf_hub_download is None:
            import huggingface_hub

            api = api or huggingface_hub.HfApi()
            snapshot_download = snapshot_download or huggingface_hub.snapshot_download
            hf_hub_download = hf_hub_download or huggingface_hub.hf_hub_download
        self.api = api
        self.snapshot_download = snapshot_download
        self.hf_hub_download = hf_hub_download

    def whoami(self) -> str:
        """Logged-in user name; raises `HubError` when there is no valid token."""
        from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

        try:
            return self.api.whoami()["name"]
        except LocalTokenNotFoundError as err:
            raise HubError("not logged in to Hugging Face: run `hf auth login` (token with write access)") from err
        except HfHubHTTPError as err:
            raise HubError(f"Hugging Face rejected the token ({err}); run `hf auth login` again") from err

    def require_repo(self, repo_id: str, repo_type: str) -> None:
        """Raise `HubError` unless the repo exists and is visible to the logged-in user."""
        from huggingface_hub.errors import RepositoryNotFoundError

        try:
            self.api.repo_info(repo_id, repo_type=repo_type)
        except RepositoryNotFoundError as err:
            raise HubError(
                f"{repo_type} repo `{repo_id}` not found (or not visible to `{self.whoami()}`); "
                "check the id (flag / HF_DATASET_REPO / HF_MODEL_REPO) or upload first"
            ) from err

    def repo_files(self, repo_id: str, repo_type: str) -> list[str]:
        """All file paths in the repo."""
        return list(self.api.list_repo_files(repo_id, repo_type=repo_type))


class DatasetTransfer:
    """Mirror `data/VeReMi-Dataset` to a private dataset repo and back, verified by a manifest."""

    def __init__(self, config: HubConfig, hub: Hub, log: Callable[[str], None] = print) -> None:
        self.config = config
        self.hub = hub
        self.log = log
        self.repo_type = "dataset"

    def card(self, manifest: Mapping[str, Mapping[str, Any]]) -> str:
        """Dataset card (README.md) for the private mirror; it records no secrets."""
        top = sorted({rel.split("/", 1)[0] for rel in manifest if "/" in rel})
        total = sum(entry["size"] for entry in manifest.values())
        folders = "\n".join(f"- `{name}/`" for name in top)
        return f"""---
license: other
pretty_name: VeReMi-Extension (private mirror, Sybil subset)
tags: [vanet, misbehavior-detection, sybil, veremi]
---

# VeReMi-Extension — private mirror (Sybil subset)

Private mirror of the raw **VeReMi-Extension** simulation logs (Kamel et al., 2020; original distribution:
<https://github.com/josephkamel/VeReMi-Dataset>) used by a master's thesis (RoadFM-Lite, Florida Polytechnic
University). It exists only to move the data to training machines. It is not a new release; cite and license
under the original dataset's terms.

## Layout

{len(manifest):,} files, {total / 1e9:.1f} GB. The repo root is the `data/VeReMi-Dataset/` folder of the project:
4 Sybil attacks x 2 time windows (`_0709`, `_1416`), each with run folders of `traceJSON-*.json` logs (one per
vehicle) and ground truth.

{folders}

`{MANIFEST_NAME}` maps every relative path to its size and sha256.

## Download (from the project repo root)

```bash
.venv-train/bin/python scripts/hf_hub.py download-dataset   # -> data/VeReMi-Dataset/, verified against the manifest
```
"""

    def upload(self, path: Path = DEFAULT_DATASET_PATH) -> str:
        """Create the private repo if needed, upload the tree (resumable), then the manifest and card.

        Returns:
            The dataset repo URL.
        """
        path = Path(path)
        if not path.is_dir() or not is_non_empty_dir(path):
            raise HubError(f"dataset folder `{path}` is missing or empty")
        user = self.hub.whoami()
        repo = self.config.dataset_repo
        self.log(f"user {user}; dataset repo {repo} (private)")
        self.hub.api.create_repo(repo, repo_type=self.repo_type, private=True, exist_ok=True)
        self.log(f"hashing {path} for {MANIFEST_NAME} (read-only) ...")
        manifest = build_manifest(path, progress=self._progress)
        has_card = CARD_NAME in self.hub.repo_files(repo, self.repo_type)
        self.log(f"uploading {len(manifest):,} files (re-run the same command to resume) ...")
        self._upload_folder(
            repo_id=repo,
            folder_path=str(path),
            repo_type=self.repo_type,
            ignore_patterns=UPLOAD_IGNORE_PATTERNS,
            commit_message=f"Upload VeReMi-Extension tree ({len(manifest)} files)",
        )
        with tempfile.TemporaryDirectory(prefix="hf-dataset-meta-") as tmp:
            meta = Path(tmp)
            (meta / MANIFEST_NAME).write_text(json.dumps(self._manifest_doc(manifest), indent=1))
            if not has_card:
                (meta / CARD_NAME).write_text(self.card(manifest))
            self.hub.api.upload_folder(
                repo_id=repo, folder_path=str(meta), repo_type=self.repo_type, commit_message="Add manifest"
            )
        url = self.config.dataset_url()
        self.log(f"done: {url}")
        return url

    def download(self, path: Path = DEFAULT_DATASET_PATH, force: bool = False, check_hashes: bool = True) -> Path:
        """Download the mirror so files land at `path/<same tree>`, then verify it.

        Raises:
            HubError: The target is non-empty without `force`, the repo is missing, or verification fails.
        """
        path = Path(path)
        if is_non_empty_dir(path) and not force:
            raise HubError(f"`{path}` exists and is not empty; refusing to overwrite (pass --force to merge into it)")
        repo = self.config.dataset_repo
        self.hub.whoami()
        self.hub.require_repo(repo, self.repo_type)
        remote = self.hub.repo_files(repo, self.repo_type)
        data_files = sorted(f for f in remote if f not in DATASET_META_FILES)
        self.log(f"downloading {len(data_files):,} files from {repo} into {path} ...")
        path.mkdir(parents=True, exist_ok=True)
        self.hub.snapshot_download(
            repo, repo_type=self.repo_type, local_dir=str(path), ignore_patterns=list(DATASET_META_FILES)
        )
        if MANIFEST_NAME in remote:
            with tempfile.TemporaryDirectory(prefix="hf-dataset-meta-") as tmp:
                local = self.hub.hf_hub_download(repo, MANIFEST_NAME, repo_type=self.repo_type, local_dir=tmp)
                manifest = json.loads(Path(local).read_text())["files"]
            self.log(f"verifying {len(manifest):,} files against {MANIFEST_NAME} ...")
            problems = verify_manifest(path, manifest, check_hashes=check_hashes)
        else:
            self.log(f"no {MANIFEST_NAME} in the repo; checking the file count only")
            got = sum(1 for _ in iter_tree(path))
            problems = [] if got == len(data_files) else [f"file count {got} != {len(data_files)} in the repo"]
        if problems:
            raise HubError("download verification failed: " + "; ".join(problems[:5]))
        shutil.rmtree(path / ".cache", ignore_errors=True)
        self.log(f"verified: {path}")
        return path

    def _upload_folder(self, **kwargs: Any) -> Any:
        """`upload_large_folder` where this huggingface_hub has it; else `upload_folder`.

        In huggingface_hub 2.x `upload_large_folder` is gone: with `hf-xet` installed `upload_folder` streams the
        files, commits in batches and resumes when re-run.
        """
        large = getattr(self.hub.api, "upload_large_folder", None)
        if callable(large):
            kwargs.pop("commit_message", None)
            return large(**kwargs)
        return self.hub.api.upload_folder(**kwargs)

    def _progress(self, done: int, total: int) -> None:
        if done == total or done % 2000 == 0:
            self.log(f"  hashed {done:,}/{total:,}")

    @staticmethod
    def _manifest_doc(manifest: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "n_files": len(manifest),
            "total_bytes": sum(entry["size"] for entry in manifest.values()),
            "files": dict(manifest),
        }


class ModelTransfer:
    """Upload pretraining run folders to `runs/<run_id>/` of a private model repo and fetch them back."""

    def __init__(self, config: HubConfig, hub: Hub, log: Callable[[str], None] = print) -> None:
        self.config = config
        self.hub = hub
        self.log = log
        self.repo_type = "model"

    @staticmethod
    def select_files(run_dir: Path, include_last: bool = False) -> tuple[list[Path], str | None]:
        """Files of a run to upload and an optional note for the user.

        Returns metadata files that exist, `best.pt` (or `last.pt` when `best.pt` is missing), `last.pt` with
        `include_last`, and any `*.log`.

        Raises:
            HubError: Neither `best.pt` nor `last.pt` exists.
        """
        note = None
        files = [run_dir / name for name in RUN_FILES if (run_dir / name).is_file()]
        best, last = run_dir / "best.pt", run_dir / "last.pt"
        if best.is_file():
            files.append(best)
            if include_last and last.is_file():
                files.append(last)
        elif last.is_file():
            files.append(last)
            note = "best.pt is missing; uploading last.pt instead"
        else:
            raise HubError(f"`{run_dir}` has neither best.pt nor last.pt; nothing to upload")
        files.extend(sorted(run_dir.glob("*.log")))
        return files, note

    def card(self) -> str:
        """Model card (README.md) for the runs repo."""
        return """---
library_name: pytorch
tags: [timesnet, self-supervised, vanet, sybil-detection, veremi]
---

# RoadFM-Lite — TimesNet pretraining runs (private)

Checkpoints of a self-supervised TimesNet encoder (masked reconstruction + injected-violation physics heads +
contrastive) pretrained without attack labels on receiver-side VeReMi-Extension sequences (benign + GridSybil,
T = 64 x 13 features). Master's thesis research, Florida Polytechnic University. Not a released model.

Each run sits in `runs/<run_id>/` with the same layout as a local run folder: `config.json` (settings + seed),
`env.json` (environment), `metrics.jsonl`, `best.pt` (and `last.pt` when uploaded), `SHA256SUMS`.

```bash
.venv-train/bin/python scripts/hf_hub.py list-models
.venv-train/bin/python scripts/hf_hub.py download-model <run_id>   # -> src/runs/pretraining/benign_gridsybil/T64/<run_id>/
```
"""

    def upload(self, run_dir: Path, name: str | None = None, include_last: bool = False) -> tuple[str, str]:
        """Upload one run in a single commit, with a SHA256SUMS of everything uploaded.

        Returns:
            (repo URL, commit id).
        """
        from huggingface_hub import CommitOperationAdd

        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise HubError(f"run folder `{run_dir}` does not exist")
        run_id = name or run_dir.resolve().name
        files, note = self.select_files(run_dir, include_last)
        if note:
            self.log(note)
        user = self.hub.whoami()
        repo = self.config.model_repo
        self.log(f"user {user}; model repo {repo} (private); run {run_id}")
        self.hub.api.create_repo(repo, repo_type=self.repo_type, private=True, exist_ok=True)
        has_card = CARD_NAME in self.hub.repo_files(repo, self.repo_type)
        sums = {path.name: sha256_file(path) for path in files}
        prefix = f"runs/{run_id}"
        operations = [CommitOperationAdd(f"{prefix}/{path.name}", str(path)) for path in files]
        operations.append(CommitOperationAdd(f"{prefix}/{SUMS_NAME}", format_sums(sums).encode()))
        if not has_card:
            operations.append(CommitOperationAdd(CARD_NAME, self.card().encode()))
        self.log("uploading " + ", ".join(path.name for path in files) + f" -> {prefix}/")
        info = self.hub.api.create_commit(
            repo_id=repo, repo_type=self.repo_type, operations=operations, commit_message=f"Upload run {run_id}"
        )
        url = self.config.model_url()
        commit = getattr(info, "oid", None) or str(info)
        self.log(f"done: {url}/tree/main/{prefix}  commit {commit}")
        return url, commit

    def list_runs(self) -> list[str]:
        """Run ids present under `runs/` in the model repo."""
        repo = self.config.model_repo
        self.hub.whoami()
        self.hub.require_repo(repo, self.repo_type)
        runs = {f.split("/")[1] for f in self.hub.repo_files(repo, self.repo_type) if f.startswith("runs/")}
        return sorted(r for r in runs if r)

    def download(self, run_id: str, to: Path = DEFAULT_RUNS_DIR, force: bool = False) -> Path:
        """Fetch `runs/<run_id>/` into `to/<run_id>/` (local run layout, so `--resume` works) and verify sha256.

        Raises:
            HubError: The target is non-empty without `force`, the run is missing, or a checksum does not match.
        """
        dest = Path(to) / run_id
        if is_non_empty_dir(dest) and not force:
            raise HubError(f"`{dest}` exists and is not empty; refusing to overwrite (pass --force)")
        repo = self.config.model_repo
        self.hub.whoami()
        self.hub.require_repo(repo, self.repo_type)
        with tempfile.TemporaryDirectory(prefix="hf-model-") as tmp:
            snapshot = self.hub.snapshot_download(
                repo, repo_type=self.repo_type, allow_patterns=[f"runs/{run_id}/*"], cache_dir=tmp
            )
            source = Path(snapshot) / "runs" / run_id
            if not source.is_dir() or not any(source.iterdir()):
                raise HubError(f"run `{run_id}` not found in {repo}; see `list-models`")
            self.verify_run(source)
            dest.mkdir(parents=True, exist_ok=True)
            for path in sorted(source.iterdir()):
                if path.is_file():
                    shutil.copy2(path, dest / path.name)
        self.log(f"downloaded and verified: {dest}")
        return dest

    def verify_run(self, folder: Path) -> None:
        """Check every file listed in SHA256SUMS (all .pt files must be listed); raise `HubError` on a mismatch."""
        sums_path = folder / SUMS_NAME
        if not sums_path.is_file():
            raise HubError(f"{SUMS_NAME} missing in the downloaded run; cannot verify the checkpoints")
        sums = parse_sums(sums_path.read_text())
        unlisted = [p.name for p in folder.glob("*.pt") if p.name not in sums]
        if unlisted:
            raise HubError(f"checkpoint(s) not in {SUMS_NAME}: {', '.join(sorted(unlisted))}")
        for name, digest in sums.items():
            path = folder / name
            if not path.is_file():
                raise HubError(f"{name} is listed in {SUMS_NAME} but was not downloaded")
            if sha256_file(path) != digest:
                raise HubError(f"sha256 mismatch for {name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line."""
    p = argparse.ArgumentParser(description="Hugging Face Hub transfers for RoadFM-Lite (private repos)")
    p.add_argument("--dataset-repo", help=f"dataset repo id (env HF_DATASET_REPO; default {DEFAULT_DATASET_REPO})")
    p.add_argument("--model-repo", help=f"model repo id (env HF_MODEL_REPO; default {DEFAULT_MODEL_REPO})")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("whoami", help="logged-in Hugging Face user (no token printed)")
    up = sub.add_parser("upload-dataset", help="upload the raw VeReMi tree to the private dataset repo")
    up.add_argument("--path", type=Path, default=DEFAULT_DATASET_PATH)
    down = sub.add_parser("download-dataset", help="download the raw VeReMi tree and verify it")
    down.add_argument("--path", type=Path, default=DEFAULT_DATASET_PATH)
    down.add_argument("--force", action="store_true", help="download into a non-empty folder")
    down.add_argument("--no-hash", action="store_true", help="verify file list + sizes only (skip sha256)")
    upm = sub.add_parser("upload-model", help="upload one run folder to runs/<run_id>/ of the model repo")
    upm.add_argument("run_dir", type=Path)
    upm.add_argument("--name", help="run id in the repo (default: the folder name)")
    upm.add_argument("--include-last", action="store_true", help="also upload last.pt (needed to --resume)")
    dm = sub.add_parser("download-model", help="download runs/<run_id>/ into the local runs folder")
    dm.add_argument("run_id")
    dm.add_argument("--to", type=Path, default=DEFAULT_RUNS_DIR)
    dm.add_argument("--force", action="store_true", help="overwrite a non-empty local run folder")
    sub.add_parser("list-models", help="run ids in the model repo")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, hub: Hub | None = None) -> int:
    """Entry point; returns the exit code."""
    args = parse_args(argv)
    config = HubConfig.resolve(args.dataset_repo, args.model_repo)
    try:
        hub = hub or Hub()
        if args.command == "whoami":
            print(hub.whoami())
        elif args.command == "upload-dataset":
            DatasetTransfer(config, hub).upload(args.path)
        elif args.command == "download-dataset":
            DatasetTransfer(config, hub).download(args.path, force=args.force, check_hashes=not args.no_hash)
        elif args.command == "upload-model":
            ModelTransfer(config, hub).upload(args.run_dir, name=args.name, include_last=args.include_last)
        elif args.command == "download-model":
            ModelTransfer(config, hub).download(args.run_id, to=args.to, force=args.force)
        elif args.command == "list-models":
            print("\n".join(ModelTransfer(config, hub).list_runs()) or "(no runs)")
    except HubError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
