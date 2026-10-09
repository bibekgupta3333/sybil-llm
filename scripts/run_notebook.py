"""Execute a notebook in place with its own kernel and the repo root as the working directory.

Works on the Mac (`.venv-train/bin/python`) and in the Docker image (`/opt/venv/bin/python`); the kernel is the one
named in the notebook's metadata (`python3` or `roadfm-train`), both registered in the image.

    python scripts/run_notebook.py src/pipeline/input_representation.ipynb [--timeout SECONDS] [--kernel NAME]

Outputs are written back into the notebook. A failing cell stops the run; the partly executed notebook is still saved
so the traceback is visible, and the exit code is non-zero.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parents[1]


class NotebookRunner:
    """Runs one notebook with nbclient from the repo root and saves it in place."""

    def __init__(self, path: Path, timeout: int | None, kernel: str | None) -> None:
        self.path = path if path.is_absolute() else (REPO_ROOT / path)
        self.timeout = timeout
        self.kernel = kernel

    def run(self) -> int:
        """Executes the notebook and returns a process exit code."""
        if not self.path.is_file():
            print(f"notebook not found: {self.path}", file=sys.stderr)
            return 2
        notebook = nbformat.read(self.path, as_version=4)
        kernel = self.kernel or notebook.metadata.get("kernelspec", {}).get("name", "python3")
        client = NotebookClient(
            notebook,
            timeout=self.timeout,
            kernel_name=kernel,
            resources={"metadata": {"path": str(REPO_ROOT)}},
        )
        print(f"running {self.path.relative_to(REPO_ROOT)} (kernel {kernel}, cwd {REPO_ROOT})", flush=True)
        started = time.monotonic()
        try:
            client.execute()
        except CellExecutionError as error:
            nbformat.write(notebook, self.path)
            print(f"failed after {time.monotonic() - started:.0f} s:\n{error}", file=sys.stderr)
            return 1
        nbformat.write(notebook, self.path)
        print(f"done in {time.monotonic() - started:.0f} s; outputs saved", flush=True)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebook", type=Path, help="notebook path (relative to the repo root or absolute)")
    parser.add_argument("--timeout", type=int, default=None, help="per-cell timeout in seconds (default: none)")
    parser.add_argument("--kernel", default=None, help="kernel name (default: the notebook's kernelspec)")
    args = parser.parse_args()
    return NotebookRunner(args.notebook, args.timeout, args.kernel).run()


if __name__ == "__main__":
    sys.exit(main())
