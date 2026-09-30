"""Hardware probing.

`probe()` is the single entry point: it gathers everything setpoint knows about the
machine into one `HardwareSnapshot`. Every sub-probe is allowed to fail; a failure
becomes a note on the snapshot, never an exception and never a fabricated value.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys

from .nvml import ACTIVE_THROTTLE_REASONS, NvmlProbe
from .types import (
    MIB,
    AdapterMemory,
    DriverInfo,
    GpuProcess,
    GpuSample,
    GpuStatic,
    HardwareSnapshot,
    HostInfo,
    ProbeStatus,
)
from .watch import DEFAULT_INTERVAL_S, GpuWatch, GpuWatcher
from .wddm import match_adapter
from .wddm import probe as probe_wddm

__all__ = [
    "ACTIVE_THROTTLE_REASONS",
    "MIB",
    "DEFAULT_INTERVAL_S",
    "AdapterMemory",
    "DriverInfo",
    "GpuProcess",
    "GpuSample",
    "GpuStatic",
    "GpuWatch",
    "GpuWatcher",
    "HardwareSnapshot",
    "HostInfo",
    "NvmlProbe",
    "ProbeStatus",
    "match_adapter",
    "probe",
    "probe_wddm",
    "available_ram_bytes",
    "total_ram_bytes",
]


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def total_ram_bytes() -> int | None:
    """Physical RAM, without pulling in psutil."""
    return _ram_bytes("total")


def available_ram_bytes() -> int | None:
    """RAM that could be handed out right now.

    This is the figure that matters for an offload plan: what stays on the CPU has to
    fit in what is free, not in what the machine was sold with. Reading the total
    instead is how a plan gets drawn that takes the machine down.
    """
    return _ram_bytes("available")


def _ram_bytes(which: str) -> int | None:
    system = platform.system()
    try:
        if system == "Windows":
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return int(status.ullTotalPhys if which == "total" else status.ullAvailPhys)
            return None
        if system == "Linux":
            wanted = "MemTotal:" if which == "total" else "MemAvailable:"
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith(wanted):
                        return int(line.split()[1]) * 1024
            return None
        if system == "Darwin":
            import subprocess

            if which != "total":
                return None
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
            )
            return int(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None
    return None


def host_info() -> HostInfo:
    return HostInfo(
        os=platform.system(),
        os_release=platform.version() or platform.release(),
        arch=platform.machine(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        total_ram_bytes=total_ram_bytes(),
        available_ram_bytes=available_ram_bytes(),
        cpu_count=os.cpu_count(),
    )


def probe(include_wddm: bool = True) -> HardwareSnapshot:
    """Collect a full hardware snapshot.

    `include_wddm` exists because the Windows counter read costs about a second; callers
    that only need NVML data can skip it.
    """
    notes: list[str] = []

    with NvmlProbe() as nvml:
        driver = nvml.driver_info()
        gpus = nvml.gpus() if nvml.ok else ()
        samples = nvml.sample() if nvml.ok else ()
        processes = nvml.processes() if nvml.ok else ()
        if not nvml.ok and nvml.detail:
            notes.append(f"nvml: {nvml.detail}")

    adapters: tuple[AdapterMemory, ...] = ()
    if include_wddm:
        status, adapters, detail = probe_wddm()
        if status is not ProbeStatus.OK and detail:
            notes.append(f"wddm: {detail}")

    return HardwareSnapshot(
        host=host_info(),
        driver=driver,
        gpus=gpus,
        samples=samples,
        adapters=adapters,
        processes=processes,
        notes=tuple(notes),
    )
