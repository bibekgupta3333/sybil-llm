"""GPU check: what the host has (nvidia-smi, lspci, Apple chip) and what torch sees, plus a tiny matmul smoke test.

Run with the training env from the repo root (`npm run gpu:check`, `npm run gpu:require`):

    .venv-train/bin/python scripts/gpu_check.py [--require-gpu] [--json]

Never crashes when torch, nvidia-smi or lspci is missing: each missing piece is reported. Exit 0, or exit 1 with
`--require-gpu` when torch sees no CUDA device.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

NVIDIA_QUERY_FIELDS = ("name", "memory.total", "memory.used", "driver_version", "utilization.gpu")
SMOKE_SIZE = 1024
SMOKE_REPEATS = 5
SMOKE_TOLERANCE = 1e-2
# PCI subsystem ids (vendor 0x10de) of AWS vGPU profiles: 0x1733 = the fractional L4 of g6f / gr6f. These need NVIDIA's
# GRID guest driver; Ubuntu's nvidia drivers refuse them ("NVIDIA vGPU ... is not supported").
VGPU_SUBSYSTEMS = frozenset({"0x1733"})
PCI_DEVICES = pathlib.Path("/sys/bus/pci/devices")

# Runs argv and returns its stdout, or None when the command is missing, fails or times out.
CommandRunner = Callable[[Sequence[str]], "str | None"]


def run_command(argv: Sequence[str], timeout: float = 20.0) -> str | None:
    """Runs a command and returns stdout; None if the tool is missing, exits non-zero or hangs."""
    if shutil.which(argv[0]) is None:
        return None
    try:
        done = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _to_number(text: str) -> float | None:
    """'81559' -> 81559.0; '[N/A]' or '' -> None."""
    try:
        return float(text.strip())
    except ValueError:
        return None


@dataclasses.dataclass
class NvidiaGpu:
    """One row of `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`."""

    name: str
    memory_total_mib: float | None
    memory_used_mib: float | None
    driver_version: str
    utilization_pct: float | None


def parse_nvidia_query(text: str) -> list[NvidiaGpu]:
    """Parses the csv rows of the query in NVIDIA_QUERY_FIELDS order; malformed rows are skipped."""
    gpus = []
    for row in text.splitlines():
        cells = [cell.strip() for cell in row.split(",")]
        if len(cells) != len(NVIDIA_QUERY_FIELDS) or not cells[0]:
            continue
        gpus.append(
            NvidiaGpu(
                name=cells[0],
                memory_total_mib=_to_number(cells[1]),
                memory_used_mib=_to_number(cells[2]),
                driver_version=cells[3],
                utilization_pct=_to_number(cells[4]),
            )
        )
    return gpus


def parse_cuda_version(nvidia_smi_banner: str) -> str | None:
    """The highest CUDA version the driver supports, from the plain `nvidia-smi` banner ('CUDA Version: 12.8')."""
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)", nvidia_smi_banner)
    return match.group(1) if match else None


def parse_lspci_nvidia(text: str) -> list[str]:
    """The `lspci` lines that mention NVIDIA."""
    return [line.strip() for line in text.splitlines() if "nvidia" in line.lower()]


def find_vgpu_devices(pci_root: pathlib.Path = PCI_DEVICES) -> list[str]:
    """PCI addresses of NVIDIA devices whose subsystem id is a known AWS vGPU profile (empty when none / no sysfs)."""
    found = []
    try:
        devices = sorted(pci_root.iterdir())
    except OSError:
        return []
    for device in devices:
        try:
            vendor = (device / "vendor").read_text().strip()
            subsystem = (device / "subsystem_device").read_text().strip()
        except OSError:
            continue
        if vendor == "0x10de" and subsystem in VGPU_SUBSYSTEMS:
            found.append(device.name)
    return found


class HostProbe:
    """What the operating system reports about GPUs, independent of torch."""

    def __init__(
        self,
        runner: CommandRunner = run_command,
        system: str | None = None,
        machine: str | None = None,
        pci_root: pathlib.Path = PCI_DEVICES,
    ):
        self._run = runner
        self._pci_root = pci_root
        self._system = system or platform.system()
        self._machine = machine or platform.machine()

    def collect(self) -> dict[str, Any]:
        """Platform, nvidia-smi rows + CUDA version, lspci NVIDIA lines (Linux), Apple chip (macOS)."""
        info: dict[str, Any] = {
            "system": self._system,
            "machine": self._machine,
            "python": sys.version.split()[0],
            "nvidia_smi": None,
            "gpus": [],
            "cuda_driver_version": None,
            "lspci_nvidia": None,
            "vgpu_devices": [],
            "apple_chip": None,
        }
        query = self._run(
            ["nvidia-smi", f"--query-gpu={','.join(NVIDIA_QUERY_FIELDS)}", "--format=csv,noheader,nounits"]
        )
        if query is not None:
            info["nvidia_smi"] = "ok"
            info["gpus"] = [dataclasses.asdict(gpu) for gpu in parse_nvidia_query(query)]
            banner = self._run(["nvidia-smi"])
            info["cuda_driver_version"] = parse_cuda_version(banner) if banner else None
        else:
            info["nvidia_smi"] = "not found or failed"
        if self._system == "Linux":
            lspci = self._run(["lspci"])
            info["lspci_nvidia"] = parse_lspci_nvidia(lspci) if lspci is not None else None
            info["vgpu_devices"] = find_vgpu_devices(self._pci_root)
        if self._system == "Darwin":
            chip = self._run(["sysctl", "-n", "machdep.cpu.brand_string"])
            info["apple_chip"] = chip.strip() if chip else None
        return info


class TorchProbe:
    """What torch sees (CUDA / MPS) and a small matmul on the best device, checked against the CPU result."""

    def __init__(self, importer: Callable[[], Any] | None = None):
        self._importer = importer or self._import_torch

    @staticmethod
    def _import_torch() -> Any:
        import torch  # noqa: PLC0415 - optional dependency, imported only when probing

        return torch

    def collect(self) -> dict[str, Any]:
        """Torch version, CUDA build, devices, bf16, MPS and the smoke test; `available: False` without torch."""
        try:
            torch = self._importer()
        except Exception as exc:  # noqa: BLE001 - torch missing or broken: report it, do not crash
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        info: dict[str, Any] = {
            "available": True,
            "version": torch.__version__,
            "built_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": 0,
            "devices": [],
            "bf16_supported": None,
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        }
        if info["cuda_available"]:
            info["device_count"] = torch.cuda.device_count()
            for index in range(info["device_count"]):
                props = torch.cuda.get_device_properties(index)
                info["devices"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "capability": f"{props.major}.{props.minor}",
                        "memory_gib": round(props.total_memory / 1024**3, 2),
                    }
                )
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        info["smoke_test"] = self.smoke_test(torch, self.best_device(info))
        return info

    @staticmethod
    def best_device(info: dict[str, Any]) -> str:
        """cuda > mps > cpu."""
        if info.get("cuda_available"):
            return "cuda"
        if info.get("mps_available"):
            return "mps"
        return "cpu"

    @staticmethod
    def smoke_test(torch: Any, device: str, size: int = SMOKE_SIZE, repeats: int = SMOKE_REPEATS) -> dict[str, Any]:
        """Times `size x size` float32 matmuls on `device`; the result must match the CPU one (max relative error)."""
        result: dict[str, Any] = {"device": device, "size": size, "ok": False}
        try:
            generator = torch.Generator().manual_seed(0)
            a_cpu = torch.randn(size, size, generator=generator)
            b_cpu = torch.randn(size, size, generator=generator)
            reference = a_cpu @ b_cpu
            a, b = a_cpu.to(device), b_cpu.to(device)
            out = a @ b  # warm-up (kernel load, allocator)
            TorchProbe._sync(torch, device)
            start = time.perf_counter()
            for _ in range(repeats):
                out = a @ b
            TorchProbe._sync(torch, device)
            seconds = (time.perf_counter() - start) / repeats
            error = float(((out.cpu() - reference).abs().max() / reference.abs().max()).item())
            result.update(
                ms_per_matmul=round(seconds * 1e3, 3),
                gflops=round(2 * size**3 / seconds / 1e9, 1) if seconds > 0 else None,
                max_rel_error=error,
                ok=error < SMOKE_TOLERANCE,
            )
        except Exception as exc:  # noqa: BLE001 - a failing device is a finding, not a crash
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    @staticmethod
    def _sync(torch: Any, device: str) -> None:
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()


class GpuReport:
    """Collects host + torch info and renders it as text or JSON."""

    def __init__(self, host: HostProbe | None = None, torch_probe: TorchProbe | None = None):
        self.host = (host or HostProbe()).collect()
        self.torch = (torch_probe or TorchProbe()).collect()

    @property
    def has_cuda(self) -> bool:
        """True when torch sees at least one CUDA device."""
        return bool(self.torch.get("available") and self.torch.get("cuda_available"))

    def as_dict(self) -> dict[str, Any]:
        """Everything collected, plus the verdict."""
        return {"host": self.host, "torch": self.torch, "cuda_ready": self.has_cuda}

    def render(self) -> str:
        """Human-readable report in the repo's ✓ ✗ → ! style."""
        lines = ["== Host ==", f"→ {self.host['system']} {self.host['machine']}, Python {self.host['python']}"]
        if self.host["gpus"]:
            for gpu in self.host["gpus"]:
                lines.append(
                    f"✓ nvidia-smi: {gpu['name']}, {_fmt(gpu['memory_used_mib'])} / {_fmt(gpu['memory_total_mib'])} MiB"
                    f" used, util {_fmt(gpu['utilization_pct'])}%, driver {gpu['driver_version']},"
                    f" CUDA {self.host['cuda_driver_version'] or '?'}"
                )
        elif self.host["system"] != "Darwin":
            lines.append(f"! nvidia-smi: {self.host['nvidia_smi']}")
        if self.host["lspci_nvidia"]:
            lines.extend(f"→ lspci: {line}" for line in self.host["lspci_nvidia"])
        elif self.host["system"] == "Linux":
            lines.append("→ lspci: no NVIDIA device" if self.host["lspci_nvidia"] == [] else "→ lspci: not available")
        lines.extend(self.driver_hints())
        if self.host["apple_chip"]:
            lines.append(f"✓ Apple chip: {self.host['apple_chip']}")

        lines.append("== torch ==")
        if not self.torch.get("available"):
            lines.append(f"✗ torch not importable ({self.torch.get('error')})")
            return "\n".join(lines)
        lines.append(f"→ torch {self.torch['version']}, built CUDA {self.torch['built_cuda'] or 'none'}")
        if self.torch["cuda_available"]:
            for device in self.torch["devices"]:
                lines.append(
                    f"✓ cuda:{device['index']} {device['name']} (sm {device['capability']}, {device['memory_gib']} GiB)"
                )
            lines.append(f"→ bf16 supported: {self.torch['bf16_supported']}")
        else:
            lines.append("! torch.cuda.is_available() is False")
        if self.host["system"] == "Darwin":
            mark = "✓" if self.torch["mps_available"] else "!"
            lines.append(f"{mark} MPS available: {self.torch['mps_available']}")
        smoke = self.torch.get("smoke_test", {})
        if smoke.get("ok"):
            lines.append(
                f"✓ smoke test on {smoke['device']}: {smoke['size']}² matmul {smoke['ms_per_matmul']} ms"
                f" ({smoke['gflops']} GFLOP/s), max rel error {smoke['max_rel_error']:.1e}"
            )
        else:
            lines.append(f"✗ smoke test on {smoke.get('device')}: {smoke.get('error') or smoke}")
        lines.append("✓ CUDA ready" if self.has_cuda else f"! no CUDA device — training runs on {smoke.get('device')}")
        return "\n".join(lines)

    def driver_hints(self) -> list[str]:
        """What to do when the host has an NVIDIA device (lspci / vGPU scan) but nvidia-smi does not work."""
        vgpu = self.host.get("vgpu_devices") or []
        if self.host["gpus"] or not (self.host["lspci_nvidia"] or vgpu):
            return []
        if vgpu:
            return [
                f"! vGPU device ({', '.join(vgpu)}, fractional GPU as on g6f / gr6f) without a working driver:"
                " it needs NVIDIA's GRID guest driver, Ubuntu's nvidia drivers refuse it",
                "  fix: bash scripts/ec2_bootstrap.sh --install-driver   (installs AWS's GRID driver; docs/ec2-training.md)",
            ]
        name = self.host["lspci_nvidia"][0].split("NVIDIA Corporation", 1)[-1].strip()
        return [
            f"! full NVIDIA GPU ({name}) present but no working driver: it needs Ubuntu's server driver"
            " (open kernel modules on Turing and newer, e.g. the g4dn T4; proprietary before) and a reboot",
            "  fix: bash scripts/ec2_bootstrap.sh --install-driver, then sudo reboot"
            "   (driver already installed: just sudo reboot; docs/ec2-training.md)",
        ]


def _fmt(value: float | None) -> str:
    return "?" if value is None else f"{value:g}"


def main(argv: Sequence[str] | None = None, report: GpuReport | None = None) -> int:
    """CLI entry point; returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--require-gpu", action="store_true", help="exit 1 when torch sees no CUDA device")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    report = report or GpuReport()
    print(json.dumps(report.as_dict(), indent=2) if args.json else report.render())
    if args.require_gpu and not report.has_cuda:
        if not args.json:
            print("✗ --require-gpu: no CUDA device visible to torch", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
