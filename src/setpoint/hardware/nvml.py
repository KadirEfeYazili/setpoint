"""NVML probe.

NVML is loaded lazily and every call is defensive, so a machine without an NVIDIA
GPU or with a renamed NVML symbol still produces a usable doctor run with less data.
A value that could not be read is reported as None, never guessed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from .types import MIB, DriverInfo, GpuProcess, GpuSample, GpuStatic, ProbeStatus

# Stable ABI constants, hardcoded because bindings disagree on the exported names.
_THROTTLE_BITS: tuple[tuple[int, str], ...] = (
    (0x0000000001, "gpu_idle"),
    (0x0000000002, "applications_clocks_setting"),
    (0x0000000004, "sw_power_cap"),
    (0x0000000008, "hw_slowdown"),
    (0x0000000010, "sync_boost"),
    (0x0000000020, "sw_thermal_slowdown"),
    (0x0000000040, "hw_thermal_slowdown"),
    (0x0000000080, "hw_power_brake_slowdown"),
    (0x0000000100, "display_clock_setting"),
)

# Reasons that mean the GPU is actually being held back. Idle and application clock
# settings are normal states.
ACTIVE_THROTTLE_REASONS = frozenset(
    {
        "sw_power_cap",
        "hw_slowdown",
        "sw_thermal_slowdown",
        "hw_thermal_slowdown",
        "hw_power_brake_slowdown",
    }
)


def _decode(value: Any) -> str | None:
    """NVML bindings return `bytes` on older versions and `str` on newer ones."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def decode_throttle_reasons(bitmask: int) -> tuple[str, ...]:
    return tuple(label for bit, label in _THROTTLE_BITS if bitmask & bit)


def _import_nvml() -> Any:
    """Import whichever NVML binding is installed, or return None."""
    try:
        import pynvml  # type: ignore[import-untyped]

        return pynvml
    except ImportError:
        pass
    try:
        import nvidia_ml_py as pynvml  # type: ignore[import-untyped, no-redef]

        return pynvml
    except ImportError:
        return None


class NvmlProbe:
    """Context manager around an initialised NVML session."""

    def __init__(self) -> None:
        self._nvml: Any = None
        self._initialised = False
        self.status: ProbeStatus = ProbeStatus.UNAVAILABLE
        self.detail: str | None = None

    def __enter__(self) -> NvmlProbe:
        nvml = _import_nvml()
        if nvml is None:
            self.status = ProbeStatus.UNAVAILABLE
            self.detail = "no NVML python binding installed (pip install nvidia-ml-py)"
            return self
        self._nvml = nvml
        try:
            nvml.nvmlInit()
        except Exception as exc:
            # Almost always means no NVIDIA driver present, which is an unsupported
            # platform rather than a failure on our side.
            self.status = ProbeStatus.UNSUPPORTED
            self.detail = f"nvmlInit failed: {exc}"
            return self
        self._initialised = True
        self.status = ProbeStatus.OK
        return self

    def __exit__(self, *_: object) -> None:
        if self._initialised and self._nvml is not None:
            with contextlib.suppress(Exception):
                self._nvml.nvmlShutdown()
        self._initialised = False

    @property
    def ok(self) -> bool:
        return self.status is ProbeStatus.OK

    def _try(self, fn_name: str, *args: Any, alt_names: tuple[str, ...] = ()) -> Any:
        """Call an NVML function by name, tolerating renames and NOT_SUPPORTED."""
        if not self.ok:
            return None
        for name in (fn_name, *alt_names):
            fn = getattr(self._nvml, name, None)
            if fn is None:
                continue
            try:
                return fn(*args)
            except Exception:
                # NOT_SUPPORTED is routine: laptop GPUs have no enforced power limit,
                # consumer cards have no ECC.
                return None
        return None

    def driver_info(self) -> DriverInfo:
        if not self.ok:
            return DriverInfo(status=self.status, detail=self.detail)

        driver = _decode(self._try("nvmlSystemGetDriverVersion"))
        nvml_ver = _decode(self._try("nvmlSystemGetNVMLVersion"))
        raw_cuda = self._try(
            "nvmlSystemGetCudaDriverVersion_v2", alt_names=("nvmlSystemGetCudaDriverVersion",)
        )

        cuda_major = cuda_minor = None
        if isinstance(raw_cuda, int) and raw_cuda > 0:
            # NVML encodes CUDA 12.4 as 12040.
            cuda_major = raw_cuda // 1000
            cuda_minor = (raw_cuda % 1000) // 10

        return DriverInfo(
            status=ProbeStatus.OK,
            driver_version=driver,
            cuda_driver_major=cuda_major,
            cuda_driver_minor=cuda_minor,
            nvml_version=nvml_ver,
        )

    def _handles(self) -> Iterator[tuple[int, Any]]:
        count = self._try("nvmlDeviceGetCount_v2", alt_names=("nvmlDeviceGetCount",))
        if not isinstance(count, int):
            return
        for i in range(count):
            handle = self._try(
                "nvmlDeviceGetHandleByIndex_v2", i, alt_names=("nvmlDeviceGetHandleByIndex",)
            )
            if handle is not None:
                yield i, handle

    def gpus(self) -> tuple[GpuStatic, ...]:
        out: list[GpuStatic] = []
        for index, handle in self._handles():
            mem = self._try("nvmlDeviceGetMemoryInfo", handle)
            total = int(getattr(mem, "total", 0)) if mem is not None else 0

            cc = self._try("nvmlDeviceGetCudaComputeCapability", handle)
            compute = (int(cc[0]), int(cc[1])) if isinstance(cc, tuple) and len(cc) == 2 else None

            pci = self._try("nvmlDeviceGetPciInfo_v3", handle, alt_names=("nvmlDeviceGetPciInfo",))
            bus_id = _decode(getattr(pci, "busId", None)) if pci is not None else None

            limit_mw = self._try("nvmlDeviceGetEnforcedPowerLimit", handle)
            power_limit = float(limit_mw) / 1000.0 if isinstance(limit_mw, int) else None

            out.append(
                GpuStatic(
                    index=index,
                    name=_decode(self._try("nvmlDeviceGetName", handle)) or f"GPU {index}",
                    uuid=_decode(self._try("nvmlDeviceGetUUID", handle)),
                    vram_total_bytes=total,
                    compute_capability=compute,
                    pci_bus_id=bus_id,
                    max_pcie_gen=self._try("nvmlDeviceGetMaxPcieLinkGeneration", handle),
                    max_pcie_width=self._try("nvmlDeviceGetMaxPcieLinkWidth", handle),
                    power_limit_w=power_limit,
                )
            )
        return tuple(out)

    def processes(self) -> tuple[GpuProcess, ...]:
        """Processes holding the card, with their memory where the driver attributes it.

        Both lists are asked for and merged: a llama.cpp build on Vulkan appears as a
        graphics client, a CUDA one as a compute client. Measured on this machine, the
        driver names every process and sizes none of them.
        """
        found: dict[int, GpuProcess] = {}
        for _, handle in self._handles():
            for call in (
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetGraphicsRunningProcesses_v3",
            ):
                alt = (call.replace("_v3", "_v2"), call.replace("_v3", ""))
                for info in self._try(call, handle, alt_names=alt) or ():
                    pid = int(getattr(info, "pid", 0))
                    if not pid:
                        continue
                    used = getattr(info, "usedGpuMemory", None)
                    found.setdefault(
                        pid,
                        GpuProcess(
                            pid=pid,
                            name=_decode(self._try("nvmlSystemGetProcessName", pid)),
                            vram_mib=used // MIB if used else None,
                        ),
                    )
        return tuple(sorted(found.values(), key=lambda p: p.pid))

    def sample(self) -> tuple[GpuSample, ...]:
        out: list[GpuSample] = []
        for index, handle in self._handles():
            mem = self._try("nvmlDeviceGetMemoryInfo", handle)
            used = int(getattr(mem, "used", 0)) if mem is not None else None
            free = int(getattr(mem, "free", 0)) if mem is not None else None

            util = self._try("nvmlDeviceGetUtilizationRates", handle)
            gpu_util = int(getattr(util, "gpu", 0)) if util is not None else None

            # 0 == NVML_TEMPERATURE_GPU
            temp = self._try("nvmlDeviceGetTemperature", handle, 0)

            power_mw = self._try("nvmlDeviceGetPowerUsage", handle)
            power = float(power_mw) / 1000.0 if isinstance(power_mw, int) else None

            mask = self._try(
                "nvmlDeviceGetCurrentClocksEventReasons",
                handle,
                alt_names=("nvmlDeviceGetCurrentClocksThrottleReasons",),
            )
            reasons = decode_throttle_reasons(mask) if isinstance(mask, int) else ()

            out.append(
                GpuSample(
                    index=index,
                    vram_used_bytes=used,
                    vram_free_bytes=free,
                    utilization_pct=gpu_util,
                    temperature_c=temp if isinstance(temp, int) else None,
                    power_w=power,
                    pcie_gen=self._try("nvmlDeviceGetCurrPcieLinkGeneration", handle),
                    pcie_width=self._try("nvmlDeviceGetCurrPcieLinkWidth", handle),
                    throttle_reasons=reasons,
                )
            )
        return tuple(out)
