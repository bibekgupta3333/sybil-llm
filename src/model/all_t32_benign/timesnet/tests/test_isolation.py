"""The package is self-contained: nothing under it imports `src.model.benign_gridsybil` (static scan + a run with it blocked)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
ROOTS = (REPO / "src" / "model" / "all_t32_benign", REPO / "src" / "pipeline" / "all_t32_benign")
FORBIDDEN = "benign_gridsybil"
IMPORT_LINE = re.compile(r"^\s*(from\s+\S+\s+import\b|import\s+\S+)")
DYNAMIC_IMPORT = re.compile(r"(import_module|__import__|find_spec)\s*\(")


def _sources() -> list[tuple[Path, list[str]]]:
    """(path, source lines) of every .py file and every notebook code cell under the two package roots."""
    found = []
    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            found.append((path, path.read_text().splitlines()))
        for path in sorted(root.rglob("*.ipynb")):
            cells = json.loads(path.read_text())["cells"]
            lines = [
                line for cell in cells if cell["cell_type"] == "code" for line in "".join(cell["source"]).split("\n")
            ]
            found.append((path, lines))
    return found


def test_scan_covers_the_package() -> None:
    names = {path.name for path, _ in _sources()}
    assert {"encoder.py", "heads.py", "train.py", "cache.py", "timesnet_T32_benign.ipynb"} <= names


def test_no_import_statement_names_the_old_package() -> None:
    offending = [
        f"{path.relative_to(REPO)}:{i}: {line.strip()}"
        for path, lines in _sources()
        for i, line in enumerate(lines, 1)
        if FORBIDDEN in line and (IMPORT_LINE.match(line) or DYNAMIC_IMPORT.search(line))
    ]
    assert not offending, "imports of the old package:\n" + "\n".join(offending)


BLOCKED_RUN = textwrap.dedent("""
    import importlib
    import importlib.abc
    import sys

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "src.model.benign_gridsybil" or name.startswith("src.model.benign_gridsybil."):
                raise ImportError(f"blocked: {name}")
            return None

    sys.meta_path.insert(0, Block())

    import torch

    modules = [
        "src.model.all_t32_benign.timesnet." + name
        for name in (
            "encoder", "heads", "views", "losses", "monitors", "base_config", "base_data", "base_train",
            "config", "data", "history", "scoring", "train", "evaluate", "plots",
        )
    ] + ["src.pipeline.all_t32_benign.cache", "src.pipeline.all_t32_benign.hub"]
    for name in modules:
        importlib.import_module(name)

    from src.model.all_t32_benign.timesnet.base_data import FEATURES, FeatureSpace
    from src.model.all_t32_benign.timesnet.config import T32BenignConfig
    from src.model.all_t32_benign.timesnet.heads import PretrainingModel
    from src.model.all_t32_benign.timesnet.losses import LossScales, Objective

    settings = T32BenignConfig(d_model=16, d_ff=8, proj_dim=16, n_blocks=1, device="cpu", seed=0)
    settings.validate()
    torch.manual_seed(0)
    space = FeatureSpace({"features": FEATURES, "normalisation": {"mean": [0.0] * 13, "std": [1.0] * 13}})
    model = PretrainingModel(settings)
    objective = Objective(settings, space)
    gen = torch.Generator().manual_seed(1)
    lengths = torch.tensor([32, 30, 2, 1, 5, 17, 32, 8])
    mask = torch.arange(32)[None, :] < lengths[:, None]
    x = torch.randn(8, 32, 13, generator=gen) * mask.unsqueeze(-1)
    h, z = model.encoder(x, mask)
    assert h.shape == (8, 32, 16) and z.shape == (8, 16)
    losses, _ = objective(model, x, mask, torch.Generator().manual_seed(2), torch.arange(8) % 3)
    assert {"recon", "nce", "p1", "p2", "p3"} <= set(losses)
    scales = LossScales(settings.norm_warmup_steps)
    scales.update({k: float(v.detach()) for k, v in losses.items()}, 0)
    loss = objective.joint(losses, scales)
    opt = torch.optim.AdamW(model.parameters(), lr=settings.lr)
    before = [p.detach().clone() for p in model.parameters()]
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    leaked = sorted(m for m in sys.modules if "benign_gridsybil" in m)
    assert not leaked, leaked
    print("isolated ok")
    """)


def test_runs_with_the_old_package_blocked() -> None:
    result = subprocess.run(
        [sys.executable, "-c", BLOCKED_RUN], cwd=REPO, capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "isolated ok" in result.stdout
