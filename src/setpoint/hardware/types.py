"""Hardware data types.

Plain frozen dataclasses with no behaviour: this is the boundary between the
platform-specific probes and the rest of setpoint, and it must stay portable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

MIB = 1024 * 1024


class ProbeStatus(str, Enum):
    """Why a probe produced (or failed to produce) data.

    `UNSUPPORTED` is not an error: it means we looked and this machine genuinely
    does not have the thing. Reporting nothing is better than reporting a guess.
    """

    OK = "ok"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass(frozen=True)
class DriverInfo:
    """Driver-level facts, independent of any single GPU."""

    status: ProbeStatus
    driver_version: str | None = None
    cuda_driver_major: int | None = None
    cuda_driver_minor: int | None = None
    nvml_version: str | None = None
    detail: str | None = None

    @property
    def cuda_driver_version(self) -> str | None:
        if self.cuda_driver_major is None:
            return None
        return f"{self.cuda_driver_major}.{self.cuda_driver_minor or 0}"

    @property
    def driver_tuple(self) -> tuple[int, ...] | None:
        """`"512.89"` -> `(512, 89)`, for ordered comparison against feature gates."""
        if not self.driver_version:
            return None
        parts: list[int] = []
        for chunk in self.driver_version.split("."):
            try:
                parts.append(int(chunk))
            except ValueError:
                return tuple(parts) or None
        return tuple(parts) or None


@dataclass(frozen=True)
class GpuStatic:
    """Facts about a GPU that do not change while the machine is running."""

    index: int
    name: str
    uuid: str | None
    vram_total_bytes: int
    compute_capability: tuple[int, int] | None = None
    pci_bus_id: str | None = None
    max_pcie_gen: int | None = None
    max_pcie_width: int | None = None
    power_limit_w: float | None = None

    @property
    def vram_total_mib(self) -> int:
        return self.vram_total_bytes // MIB


@dataclass(frozen=True)
class GpuSample:
    """A single point-in-time reading of a GPU."""

    index: int
    timestamp: float = field(default_factory=time.time)
    vram_used_bytes: int | None = None
    vram_free_bytes: int | None = None
    utilization_pct: int | None = None
    temperature_c: int | None = None
    power_w: float | None = None
    pcie_gen: int | None = None
    pcie_width: int | None = None
    throttle_reasons: tuple[str, ...] = ()

    @property
    def vram_used_mib(self) -> int | None:
        return None if self.vram_used_bytes is None else self.vram_used_bytes // MIB

    @property
    def vram_free_mib(self) -> int | None:
        return None if self.vram_free_bytes is None else self.vram_free_bytes // MIB


@dataclass(frozen=True)
class AdapterMemory:
    """Windows WDDM per-adapter memory accounting.

    This is the *operating system's* view, which is the only place a driver-level
    spill from dedicated VRAM into system RAM becomes visible. NVML does not show it.
    """

    instance: str
    dedicated_bytes: int | None = None
    shared_bytes: int | None = None
    committed_bytes: int | None = None


@dataclass(frozen=True)
class HostInfo:
    os: str
    os_release: str
    arch: str
    python_version: str
    total_ram_bytes: int | None = None
    available_ram_bytes: int | None = None
    cpu_count: int | None = None


@dataclass(frozen=True)
class HardwareSnapshot:
    """Everything setpoint knows about this machine at one instant."""

    host: HostInfo
    driver: DriverInfo
    gpus: tuple[GpuStatic, ...] = ()
    samples: tuple[GpuSample, ...] = ()
    adapters: tuple[AdapterMemory, ...] = ()
    notes: tuple[str, ...] = ()

    def sample_for(self, index: int) -> GpuSample | None:
        for s in self.samples:
            if s.index == index:
                return s
        return None

    @property
    def has_nvidia_gpu(self) -> bool:
        return bool(self.gpus)
