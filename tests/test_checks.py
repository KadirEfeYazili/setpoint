"""Doctor checks against machines we do not have.

setpoint is developed on one machine and will run on many. These tests drive every
check with snapshots that stand in for other hardware - no GPU, an unreadable driver, a
card far larger than the development one, a host that is not Windows - and assert the
rules that keep an answer honest there: a check that could not look reports that it
could not look, and a check that failed says how to fix it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from setpoint import doctor
from setpoint.backend import BackendDevice
from setpoint.doctor.checks import ALL_CHECKS
from setpoint.doctor.types import Outcome, Severity
from setpoint.hardware.types import (
    AdapterMemory,
    DriverInfo,
    GpuSample,
    GpuStatic,
    HardwareSnapshot,
    HostInfo,
    ProbeStatus,
)

GIB = 1024**3

# Findings from these checks describe the GPU. Without GPU data they must not conclude.
GPU_CHECK_PREFIXES = ("nvidia.", "driver.", "vram.", "gpu.")


@pytest.fixture(autouse=True)
def _no_backend_installed(monkeypatch):
    """Keep the backend out of the matrix, and keep the checks from running a subprocess."""
    monkeypatch.delenv("SETPOINT_LLAMA_BENCH", raising=False)
    monkeypatch.setattr("setpoint.doctor.checks.find_binary", lambda *a, **k: None)
    monkeypatch.setattr("setpoint.doctor.checks.shutil.which", lambda _: None)
    monkeypatch.setattr("setpoint.doctor.checks.backend_devices", lambda: None)


def with_backend(monkeypatch, *devices: BackendDevice, path: str = "/opt/llama-bench") -> None:
    """Pretend a backend is installed and offers these devices."""
    monkeypatch.setattr("setpoint.doctor.checks.find_binary", lambda *a, **k: Path(path))
    monkeypatch.setattr("setpoint.doctor.checks.backend_devices", lambda: devices or None)


IGPU = BackendDevice("Vulkan0", "Intel(R) Iris(R) Xe Graphics", 8064, 7404)
DGPU = BackendDevice("Vulkan1", "NVIDIA GeForce GTX 1650", 4176, 3581)
CUDA = BackendDevice("CUDA0", "NVIDIA GeForce RTX 4090", 24564, 23000)


def host(system: str = "Windows", ram: int | None = 16 * GIB) -> HostInfo:
    return HostInfo(system, "10.0", "AMD64", "3.12.0", total_ram_bytes=ram)


def machine(
    system: str = "Windows",
    driver: DriverInfo | None = None,
    gpus: tuple[GpuStatic, ...] = (),
    samples: tuple[GpuSample, ...] = (),
    adapters: tuple[AdapterMemory, ...] = (),
) -> HardwareSnapshot:
    return HardwareSnapshot(
        host=host(system),
        driver=driver or DriverInfo(ProbeStatus.UNAVAILABLE, detail="NVML not loaded"),
        gpus=gpus,
        samples=samples,
        adapters=adapters,
    )


def nvidia(
    name: str = "NVIDIA GeForce RTX 4090",
    vram: int = 24 * GIB,
    free: int | None = 23 * GIB,
    driver_version: str = "580.10",
    cuda: tuple[int, int] = (13, 0),
    throttle: tuple[str, ...] = (),
    pcie: tuple[int, int] | None = (16, 4),
) -> HardwareSnapshot:
    gpu = GpuStatic(
        index=0,
        name=name,
        uuid="GPU-0",
        vram_total_bytes=vram,
        compute_capability=(8, 9),
        max_pcie_gen=4,
        max_pcie_width=16,
    )
    sample = GpuSample(
        index=0,
        vram_free_bytes=free,
        vram_used_bytes=None if free is None else vram - free,
        temperature_c=45,
        throttle_reasons=throttle,
        pcie_width=pcie[0] if pcie else None,
        pcie_gen=pcie[1] if pcie else None,
    )
    return machine(
        driver=DriverInfo(
            ProbeStatus.OK,
            driver_version=driver_version,
            cuda_driver_major=cuda[0],
            cuda_driver_minor=cuda[1],
        ),
        gpus=(gpu,),
        samples=(sample,),
    )


# Machines other than the one this was written on.
MATRIX = {
    "no nvml at all": machine(),
    "nvml present, no nvidia card": machine(
        driver=DriverInfo(ProbeStatus.UNSUPPORTED, detail="no NVIDIA device")
    ),
    "linux host": machine(system="Linux"),
    "macos host": machine(system="Darwin"),
    "driver without a cuda version": machine(
        driver=DriverInfo(ProbeStatus.OK, driver_version="470.00"),
        gpus=(GpuStatic(0, "NVIDIA Tesla T4", None, 16 * GIB),),
        samples=(GpuSample(0),),
    ),
    "modern workstation card": nvidia(),
    "old driver, small card": nvidia(
        name="NVIDIA GeForce GTX 1650",
        vram=4 * GIB,
        free=3 * GIB,
        driver_version="512.89",
        cuda=(11, 6),
    ),
    "card with no readings at all": nvidia(free=None, pcie=None),
    "throttling card": nvidia(throttle=("hw_thermal_slowdown",)),
    "datacentre card, no pcie report": nvidia(name="NVIDIA H100 80GB", vram=80 * GIB, pcie=None),
}


class TestEveryMachine:
    @pytest.mark.parametrize("label", MATRIX)
    def test_no_check_raises(self, label):
        report = doctor.run(MATRIX[label])
        assert report.findings
        assert not any(f.title == "Check failed to run" for f in report.findings)

    @pytest.mark.parametrize("label", MATRIX)
    def test_a_failure_always_says_how_to_fix_it(self, label):
        # The promise is three things: what, why it costs throughput, how to fix it.
        for finding in doctor.run(MATRIX[label]).failures:
            assert finding.what, finding.check_id
            assert finding.why, finding.check_id
            assert finding.fix, finding.check_id

    @pytest.mark.parametrize("label", MATRIX)
    def test_the_exit_code_stays_inside_the_contract(self, label):
        assert doctor.run(MATRIX[label]).exit_code in (0, 1)

    @pytest.mark.parametrize("label", MATRIX)
    def test_check_ids_are_unique(self, label):
        ids = [f.check_id for f in doctor.run(MATRIX[label]).findings]
        assert len(ids) == len(set(ids))


class TestHardwareWeDoNotHave:
    @pytest.mark.parametrize(
        "label", ["no nvml at all", "nvml present, no nvidia card", "linux host", "macos host"]
    )
    def test_gpu_checks_never_conclude_without_a_gpu(self, label):
        # Not being able to look and having looked are different claims.
        for finding in doctor.run(MATRIX[label]).findings:
            if finding.check_id.startswith(GPU_CHECK_PREFIXES):
                assert finding.outcome is not Outcome.FAIL, finding.check_id
                assert finding.outcome is not Outcome.PASS, finding.check_id

    def test_a_machine_without_a_gpu_is_not_reported_as_broken(self):
        report = doctor.run(MATRIX["nvml present, no nvidia card"])
        assert not any(f.severity is Severity.CRITICAL for f in report.failures)

    def test_a_non_windows_host_says_the_spill_probe_is_windows_only(self):
        report = doctor.run(MATRIX["linux host"])
        spill = [f for f in report.findings if f.check_id.startswith("vram.")]
        assert spill
        assert all(f.outcome is Outcome.SKIP for f in spill)

    def test_an_unreadable_reading_is_skipped_rather_than_guessed(self):
        report = doctor.run(MATRIX["card with no readings at all"])
        skipped = {f.check_id for f in report.skipped}
        assert "gpu.pcie-link" in skipped


class TestHealthyMachine:
    def test_a_current_setup_raises_no_critical_finding(self):
        report = doctor.run(MATRIX["modern workstation card"])
        assert not [f for f in report.failures if f.severity is Severity.CRITICAL]

    def test_an_old_driver_is_the_one_that_gets_flagged(self):
        report = doctor.run(MATRIX["old driver, small card"])
        critical = [f.check_id for f in report.failures if f.severity is Severity.CRITICAL]
        assert "driver.cuda-support" in critical

    def test_thermal_throttling_is_reported_but_idle_is_not(self):
        throttling = doctor.run(MATRIX["throttling card"])
        quiet = doctor.run(MATRIX["modern workstation card"])
        assert any("throttl" in f.title.lower() for f in throttling.failures)
        assert not any("throttl" in f.title.lower() for f in quiet.failures)


class TestCheckContract:
    @pytest.mark.parametrize("check", ALL_CHECKS, ids=lambda c: c.__name__)
    def test_every_check_handles_an_empty_machine(self, check):
        findings = check(machine())
        assert isinstance(findings, list)
        assert all(f.check_id for f in findings)


class TestBackendChangesTheVerdict:
    """A driver too old for CUDA prebuilts is only a fault if CUDA is what you run."""

    def test_an_old_driver_with_a_vulkan_backend_is_not_a_fault(self, monkeypatch):
        with_backend(monkeypatch, DGPU)
        report = doctor.run(MATRIX["old driver, small card"])
        cuda = next(f for f in report.findings if f.check_id == "driver.cuda-support")
        assert cuda.outcome is Outcome.PASS
        assert not [f for f in report.failures if f.severity is Severity.CRITICAL]

    def test_an_old_driver_with_no_backend_stays_critical(self, monkeypatch):
        report = doctor.run(MATRIX["old driver, small card"])
        cuda = next(f for f in report.findings if f.check_id == "driver.cuda-support")
        assert cuda.outcome is Outcome.FAIL
        assert cuda.severity is Severity.CRITICAL

    def test_a_cuda_only_backend_does_not_excuse_the_old_driver(self, monkeypatch):
        # CUDA is exactly the thing the driver cannot run, so this is no reprieve.
        with_backend(monkeypatch, CUDA)
        report = doctor.run(MATRIX["old driver, small card"])
        cuda = next(f for f in report.findings if f.check_id == "driver.cuda-support")
        assert cuda.outcome is Outcome.FAIL


class TestDeviceChoice:
    def test_two_accelerators_are_flagged(self, monkeypatch):
        with_backend(monkeypatch, IGPU, DGPU)
        report = doctor.run(MATRIX["old driver, small card"])
        choice = next(f for f in report.findings if f.check_id == "backend.device-choice")
        assert choice.outcome is Outcome.FAIL
        assert choice.severity is Severity.WARNING
        assert "Iris" in choice.what and "GTX 1650" in choice.what
        assert choice.fix

    def test_one_accelerator_has_nothing_to_choose_wrongly(self, monkeypatch):
        with_backend(monkeypatch, DGPU)
        report = doctor.run(MATRIX["old driver, small card"])
        choice = next(f for f in report.findings if f.check_id == "backend.device-choice")
        assert choice.outcome is Outcome.PASS

    def test_without_a_backend_there_is_no_device_finding_at_all(self):
        ids = {f.check_id for f in doctor.run(MATRIX["old driver, small card"]).findings}
        assert "backend.device-choice" not in ids

    def test_a_backend_that_will_not_list_its_devices_is_not_an_error(self, monkeypatch):
        monkeypatch.setattr("setpoint.doctor.checks.find_binary", lambda *a, **k: Path("x"))
        monkeypatch.setattr("setpoint.doctor.checks.backend_devices", lambda: None)
        report = doctor.run(MATRIX["old driver, small card"])
        backend = next(f for f in report.findings if f.check_id == "backend.llama-cpp")
        assert backend.outcome is Outcome.PASS
        assert "could not be asked" in backend.what
