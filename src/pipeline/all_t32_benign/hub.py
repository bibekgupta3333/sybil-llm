"""Uploads a whole T32_benign TimesNet run (every file) and its evaluation folders to the private HF model repo.

`scripts/hf_hub.py upload-model` sends a fixed file list (config, env, metrics, checkpoints, `*.log`); this sends every
file of the run folder (history.jsonl, feature_history.jsonl, steps.jsonl, summary.json, train.log, best.pt, last.pt,
...) plus `src/runs/evaluation/all/T32_benign/<run_id>[-random-init]/` when present, each with a SHA256SUMS file.

Usage (repo root):
    .venv-train/bin/python -m src.pipeline.all_t32_benign.hub upload --seed 0
    .venv-train/bin/python -m src.pipeline.all_t32_benign.hub upload --run-id model-t32b-short-seed0
    .venv-train/bin/python -m src.pipeline.all_t32_benign.hub download --seed 0   (run + evaluations, sha256-checked)

Repo layout: `runs/<run_id>/...` and `evaluations/<eval_folder>/...` in `$HF_MODEL_REPO`
(default `bibekgupta3333/roadfm-lite-timesnet`, private).
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

RUNS_ROOT = Path("src/runs/pretraining/all/T32_benign")
EVAL_ROOT = Path("src/runs/evaluation/all/T32_benign")
DEFAULT_REPO = "bibekgupta3333/roadfm-lite-timesnet"
SUMS = "SHA256SUMS"
SKIP = {"PAUSE", SUMS}


class FolderUpload:
    """One local folder → one repo prefix, every regular file below it, with a SHA256SUMS manifest.

    Args:
        folder: local folder.
        prefix: path inside the repo.
    """

    def __init__(self, folder: Path, prefix: str) -> None:
        self.folder = folder
        self.prefix = prefix

    def files(self) -> list[Path]:
        """Every regular file below the folder except the pause flag and an old manifest."""
        return sorted(p for p in self.folder.rglob("*") if p.is_file() and p.name not in SKIP)

    def write_sums(self) -> int:
        """Writes SHA256SUMS (relative paths); returns the number of files listed."""
        lines = []
        for path in self.files():
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
            lines.append(f"{digest.hexdigest()}  {path.relative_to(self.folder).as_posix()}\n")
        (self.folder / SUMS).write_text("".join(lines))
        return len(lines)

    def upload(self, api: "HfApi", repo: str) -> None:  # noqa: F821 (lazy import)
        n = self.write_sums()
        print(f"uploading {self.folder} ({n} files + {SUMS}) -> {repo}/{self.prefix}", flush=True)
        api.upload_folder(
            repo_id=repo,
            repo_type="model",
            folder_path=str(self.folder),
            path_in_repo=self.prefix,
            ignore_patterns=["PAUSE"],
            delete_patterns=["*"],  # the repo folder mirrors the local one (stale files removed in the same commit)
            commit_message=f"Upload {self.prefix}",
        )


class RunUploader:
    """Uploads a run folder and the evaluation folders that belong to it.

    Args:
        run_id: run folder name under RUNS_ROOT.
        repo: HF model repo id (private).
    """

    def __init__(self, run_id: str, repo: str) -> None:
        self.run_id = run_id
        self.repo = repo

    def folders(self) -> list[FolderUpload]:
        run_dir = RUNS_ROOT / self.run_id
        if not (run_dir / "config.json").is_file():
            raise SystemExit(f"{run_dir} is not a run folder (no config.json)")
        if (run_dir / "PAUSE").exists():
            print("note: the run is paused or still pausing; uploading its current state", flush=True)
        out = [FolderUpload(run_dir, f"runs/{self.run_id}")]
        for name in (self.run_id, f"{self.run_id}-random-init"):
            if (EVAL_ROOT / name / "summary.json").is_file():
                out.append(FolderUpload(EVAL_ROOT / name, f"evaluations/{name}"))
        return out

    def run(self) -> None:
        from huggingface_hub import HfApi  # lazy: tests and --help work without the hub client

        api = HfApi()
        api.create_repo(self.repo, repo_type="model", private=True, exist_ok=True)
        for folder in self.folders():
            folder.upload(api, self.repo)
        print(f"done: https://huggingface.co/{self.repo}/tree/main/runs/{self.run_id}", flush=True)


def download(run_id: str, repo: str) -> None:
    """Downloads `runs/<run_id>/` and its evaluation folders back to their local places and checks each SHA256SUMS."""
    import shutil
    import tempfile

    from huggingface_hub import snapshot_download  # lazy, as in RunUploader.run

    targets = {
        f"runs/{run_id}": RUNS_ROOT / run_id,
        f"evaluations/{run_id}": EVAL_ROOT / run_id,
        f"evaluations/{run_id}-random-init": EVAL_ROOT / f"{run_id}-random-init",
    }
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(repo, repo_type="model", local_dir=tmp, allow_patterns=[f"{k}/*" for k in targets])
        for prefix, dest in targets.items():
            src = Path(tmp) / prefix
            if not (src / SUMS).is_file():
                print(f"skip {prefix}: not in {repo}", flush=True)
                continue
            for line in (src / SUMS).read_text().splitlines():
                digest, rel = line.split("  ", 1)
                h = hashlib.sha256()
                with (src / rel).open("rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        h.update(block)
                if h.hexdigest() != digest:
                    raise SystemExit(f"sha256 mismatch: {prefix}/{rel}")
            if dest.exists():
                raise SystemExit(f"{dest} exists; move it away first (nothing overwritten)")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest)
            print(f"{prefix} -> {dest} (sha256 OK)", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    up = sub.add_parser("upload", help="upload a run folder + its evaluation folders (every file)")
    up.add_argument("--seed", type=int, default=0, help="run model-t32b-seed<seed> (default 0)")
    up.add_argument("--run-id", help="run folder name (overrides --seed)")
    up.add_argument("--repo", default=os.environ.get("HF_MODEL_REPO", DEFAULT_REPO))
    down = sub.add_parser("download", help="download a run + its evaluation folders (sha256-checked)")
    for q in (down,):
        q.add_argument("--seed", type=int, default=0, help="run model-t32b-seed<seed> (default 0)")
        q.add_argument("--run-id", help="run folder name (overrides --seed)")
        q.add_argument("--repo", default=os.environ.get("HF_MODEL_REPO", DEFAULT_REPO))
    args = p.parse_args(argv)
    run_id = args.run_id or f"model-t32b-seed{args.seed}"
    if args.command == "download":
        download(run_id, args.repo)
    else:
        RunUploader(run_id, args.repo).run()


if __name__ == "__main__":
    main()
