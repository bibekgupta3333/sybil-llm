"""Unit tests for the GPU check (fake commands and a fake / missing torch; CPU only, no GPU needed)."""

from __future__ import annotations

import json
import pathlib
import sys
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gpu_check  # noqa: E402

QUERY_OUT = "NVIDIA A10G, 23028, 1, 570.86.15, 0\nNVIDIA A10G, 23028, 512, 570.86.15, 37\n"
BANNER_OUT = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.15              Driver Version: 570.86.15      CUDA Version: 12.8     |
+-----------------------------------------------------------------------------------------+
"""
NO_PCI = pathlib.Path("/nonexistent-pci-root")  # keeps the tests independent of the machine's own sysfs
LSPCI_L4 = "31:00.0 3D controller: NVIDIA Corporation AD104GL [L4] (rev a1)\n"
LSPCI_OUT = (
    "00:1e.0 3D controller: NVIDIA Corporation GA102GL [A10G] (rev a1)\n"
    "00:03.0 VGA compatible controller: Amazon.com, Inc. Device 1111\n"
)


class FakeRunner:
    """Answers commands from a table keyed by argv[0] (+ first flag); unknown commands are 'missing' (None)."""

    def __init__(self, answers: dict[str, str | None]):
        self.answers = answers
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> str | None:
        argv = list(argv)
        self.calls.append(argv)
        key = argv[0] if len(argv) == 1 else f"{argv[0]} {argv[1].split('=')[0]}"
        return self.answers.get(key)


class NoTorchProbe(gpu_check.TorchProbe):
    def __init__(self) -> None:
        def importer():
            raise ModuleNotFoundError("No module named 'torch'")

        super().__init__(importer=importer)


def cpu_torch():
    """The real torch with CUDA and MPS reported as unavailable (CPU path), or skip when torch is missing."""
    torch = pytest.importorskip("torch")
    fake = SimpleNamespace(
        __version__=torch.__version__,
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        Generator=torch.Generator,
        randn=torch.randn,
    )
    return fake


# --- parsing -------------------------------------------------------------------------------------------------------


def test_parse_nvidia_query_rows():
    gpus = gpu_check.parse_nvidia_query(QUERY_OUT)
    assert len(gpus) == 2
    assert gpus[0].name == "NVIDIA A10G"
    assert gpus[0].memory_total_mib == 23028
    assert gpus[1].memory_used_mib == 512
    assert gpus[1].driver_version == "570.86.15"
    assert gpus[1].utilization_pct == 37


def test_parse_nvidia_query_handles_na_and_junk():
    gpus = gpu_check.parse_nvidia_query("Tesla T4, [N/A], [N/A], 560.35.03, [N/A]\nnot a csv row\n\n")
    assert len(gpus) == 1
    assert gpus[0].memory_total_mib is None and gpus[0].utilization_pct is None
    assert gpus[0].driver_version == "560.35.03"


def test_parse_cuda_version():
    assert gpu_check.parse_cuda_version(BANNER_OUT) == "12.8"
    assert gpu_check.parse_cuda_version("no banner here") is None


def test_parse_lspci_nvidia():
    assert gpu_check.parse_lspci_nvidia(LSPCI_OUT) == [
        "00:1e.0 3D controller: NVIDIA Corporation GA102GL [A10G] (rev a1)"
    ]
    assert gpu_check.parse_lspci_nvidia("00:03.0 VGA: Amazon\n") == []


def test_run_command_missing_tool_returns_none():
    assert gpu_check.run_command(["definitely-not-a-real-command-xyz"]) is None


# --- host probe ----------------------------------------------------------------------------------------------------


def test_host_probe_linux_with_gpu():
    runner = FakeRunner({"nvidia-smi --query-gpu": QUERY_OUT, "nvidia-smi": BANNER_OUT, "lspci": LSPCI_OUT})
    info = gpu_check.HostProbe(runner=runner, system="Linux", machine="x86_64", pci_root=NO_PCI).collect()
    assert info["nvidia_smi"] == "ok"
    assert [g["name"] for g in info["gpus"]] == ["NVIDIA A10G", "NVIDIA A10G"]
    assert info["cuda_driver_version"] == "12.8"
    assert info["lspci_nvidia"] and "A10G" in info["lspci_nvidia"][0]
    assert info["apple_chip"] is None


def test_host_probe_linux_without_nvidia_smi_or_lspci():
    info = gpu_check.HostProbe(runner=FakeRunner({}), system="Linux", machine="x86_64", pci_root=NO_PCI).collect()
    assert info["nvidia_smi"] == "not found or failed"
    assert info["gpus"] == [] and info["cuda_driver_version"] is None
    assert info["lspci_nvidia"] is None


def test_host_probe_macos_reports_chip():
    runner = FakeRunner({"sysctl -n": "Apple M4 Pro\n"})
    info = gpu_check.HostProbe(runner=runner, system="Darwin", machine="arm64").collect()
    assert info["apple_chip"] == "Apple M4 Pro"
    assert info["lspci_nvidia"] is None
    assert not any(call[0] == "lspci" for call in runner.calls)


# --- torch probe + report ------------------------------------------------------------------------------------------


def test_no_torch_is_reported_not_raised():
    info = NoTorchProbe().collect()
    assert info["available"] is False
    assert "ModuleNotFoundError" in info["error"]


def test_cpu_path_smoke_test_passes():
    probe = gpu_check.TorchProbe(importer=cpu_torch)
    info = probe.collect()
    assert info["available"] and not info["cuda_available"] and not info["mps_available"]
    assert info["device_count"] == 0
    smoke = info["smoke_test"]
    assert smoke["device"] == "cpu" and smoke["ok"], smoke
    assert smoke["max_rel_error"] < gpu_check.SMOKE_TOLERANCE


def test_smoke_test_reports_device_errors():
    torch = pytest.importorskip("torch")
    result = gpu_check.TorchProbe.smoke_test(torch, "no-such-device", size=8, repeats=1)
    assert result["ok"] is False and "error" in result


@pytest.mark.parametrize(
    ("info", "device"),
    [({"cuda_available": True, "mps_available": True}, "cuda"), ({"mps_available": True}, "mps"), ({}, "cpu")],
)
def test_best_device_order(info, device):
    assert gpu_check.TorchProbe.best_device(info) == device


def linux_no_gpu_report(torch_probe: gpu_check.TorchProbe) -> gpu_check.GpuReport:
    host = gpu_check.HostProbe(
        runner=FakeRunner({"lspci": LSPCI_OUT.splitlines()[1]}), system="Linux", machine="x", pci_root=NO_PCI
    )
    return gpu_check.GpuReport(host=host, torch_probe=torch_probe)


def test_report_without_torch_renders_and_exits_zero(capsys):
    report = linux_no_gpu_report(NoTorchProbe())
    assert report.has_cuda is False
    assert gpu_check.main([], report=report) == 0
    out = capsys.readouterr().out
    assert "torch not importable" in out
    assert "lspci: no NVIDIA device" in out


def test_require_gpu_exits_one_without_cuda(capsys):
    report = linux_no_gpu_report(NoTorchProbe())
    assert gpu_check.main(["--require-gpu"], report=report) == 1
    assert "no CUDA device" in capsys.readouterr().err


def test_cpu_report_text_and_json(capsys):
    report = linux_no_gpu_report(gpu_check.TorchProbe(importer=cpu_torch))
    assert gpu_check.main([], report=report) == 0
    text = capsys.readouterr().out
    assert "smoke test on cpu" in text and "no CUDA device" in text
    assert gpu_check.main(["--json"], report=report) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["cuda_ready"] is False
    assert data["torch"]["smoke_test"]["device"] == "cpu"


def test_report_with_cuda_device(capsys):
    """A fake CUDA torch: devices, capability and bf16 are reported and --require-gpu passes."""

    class FakeProbe(gpu_check.TorchProbe):
        def collect(self):
            return {
                "available": True,
                "version": "2.14.1+cu126",
                "built_cuda": "12.6",
                "cuda_available": True,
                "device_count": 1,
                "devices": [{"index": 0, "name": "NVIDIA A10G", "capability": "8.6", "memory_gib": 22.3}],
                "bf16_supported": True,
                "mps_available": False,
                "smoke_test": {
                    "device": "cuda",
                    "size": 1024,
                    "ok": True,
                    "ms_per_matmul": 0.2,
                    "gflops": 10000.0,
                    "max_rel_error": 1e-4,
                },
            }

    runner = FakeRunner({"nvidia-smi --query-gpu": QUERY_OUT, "nvidia-smi": BANNER_OUT, "lspci": LSPCI_OUT})
    report = gpu_check.GpuReport(
        host=gpu_check.HostProbe(runner=runner, system="Linux", pci_root=NO_PCI), torch_probe=FakeProbe()
    )
    assert gpu_check.main(["--require-gpu"], report=report) == 0
    out = capsys.readouterr().out
    assert "cuda:0 NVIDIA A10G (sm 8.6, 22.3 GiB)" in out
    assert "driver 570.86.15" in out and "CUDA 12.8" in out
    assert "CUDA ready" in out


# --- vGPU (g6f / gr6f) and driver hints -----------------------------------------------------------------------------


def fake_pci(root: pathlib.Path, devices: dict[str, tuple[str, str]]) -> pathlib.Path:
    """A fake /sys/bus/pci/devices: {address: (vendor, subsystem_device)}."""
    for address, (vendor, subsystem) in devices.items():
        (root / address).mkdir(parents=True)
        (root / address / "vendor").write_text(vendor + "\n")
        (root / address / "subsystem_device").write_text(subsystem + "\n")
    return root


def test_find_vgpu_devices(tmp_path):
    root = fake_pci(
        tmp_path,
        {"0000:31:00.0": ("0x10de", "0x1733"), "0000:1e.0": ("0x10de", "0x1234"), "0000:00.3": ("0x1d0f", "0x1733")},
    )
    assert gpu_check.find_vgpu_devices(root) == ["0000:31:00.0"]
    assert gpu_check.find_vgpu_devices(tmp_path / "missing") == []


def test_vgpu_without_driver_hints_grid(tmp_path, capsys):
    root = fake_pci(tmp_path, {"0000:31:00.0": ("0x10de", "0x1733")})
    host = gpu_check.HostProbe(runner=FakeRunner({"lspci": LSPCI_L4}), system="Linux", machine="x86_64", pci_root=root)
    report = gpu_check.GpuReport(host=host, torch_probe=NoTorchProbe())
    assert report.host["vgpu_devices"] == ["0000:31:00.0"]
    assert gpu_check.main([], report=report) == 0
    out = capsys.readouterr().out
    assert "vGPU device (0000:31:00.0" in out and "GRID guest driver" in out
    assert "ec2_bootstrap.sh --install-driver" in out


def test_nvidia_without_driver_hints_install_driver(capsys):
    host = gpu_check.HostProbe(runner=FakeRunner({"lspci": LSPCI_L4}), system="Linux", machine="x", pci_root=NO_PCI)
    report = gpu_check.GpuReport(host=host, torch_probe=NoTorchProbe())
    assert gpu_check.main([], report=report) == 0
    out = capsys.readouterr().out
    assert "NVIDIA device present but no working driver" in out and "GRID" in out


def test_no_hint_when_driver_works(tmp_path, capsys):
    root = fake_pci(tmp_path, {"0000:31:00.0": ("0x10de", "0x1733")})
    runner = FakeRunner({"nvidia-smi --query-gpu": QUERY_OUT, "nvidia-smi": BANNER_OUT, "lspci": LSPCI_L4})
    report = gpu_check.GpuReport(
        host=gpu_check.HostProbe(runner=runner, system="Linux", pci_root=root), torch_probe=NoTorchProbe()
    )
    assert gpu_check.main([], report=report) == 0
    out = capsys.readouterr().out
    assert "nvidia-smi: NVIDIA A10G" in out and "fix:" not in out and "no working driver" not in out
