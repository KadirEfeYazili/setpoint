"""Individual doctor checks.

Each check takes a hardware snapshot and returns findings, keeping observations
separate from inferences.

The recurring trap is the idle machine: at idle a GPU downclocks, narrows its PCIe
link and reports an idle throttle reason, none of which are faults. Checks that only
mean something under load say so instead of raising a false alarm.
"""

from __future__ import annotations

import shutil

from ..backend import BINARY_ENV_VAR, BackendDevice, BackendError, LlamaCppBackend, find_binary
from ..hardware import ACTIVE_THROTTLE_REASONS, MIB, HardwareSnapshot, ProbeStatus, match_adapter
from .types import Finding, Outcome, Severity

# NVIDIA added the "CUDA - Sysmem Fallback Policy" control in this Windows driver.
# Before it, a CUDA allocation that did not fit failed with out-of-memory; after it,
# the driver may silently back the allocation with system RAM instead.
SYSMEM_FALLBACK_DRIVER = (536, 40)

# Official llama.cpp prebuilt Windows CUDA binaries are built against CUDA 12.
MIN_CUDA_FOR_PREBUILT = (12, 0)

# Idle VRAM consumed by the driver, compositor and background apps. Above this share of
# total VRAM the budget available to a model gets meaningfully squeezed.
IDLE_OVERHEAD_WARN_RATIO = 0.15


def _gib(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / (1024**3):.2f} GiB"


def check_nvidia_present(snap: HardwareSnapshot) -> list[Finding]:
    if snap.driver.status is ProbeStatus.UNAVAILABLE:
        return [
            Finding(
                check_id="nvidia.present",
                title="NVIDIA telemetry",
                outcome=Outcome.SKIP,
                severity=Severity.INFO,
                what="NVML could not be loaded, so no GPU checks could run.",
                fix="Install the NVML binding: pip install nvidia-ml-py",
                evidence={"detail": snap.driver.detail},
            )
        ]

    if snap.driver.status is ProbeStatus.UNSUPPORTED or not snap.gpus:
        return [
            Finding(
                check_id="nvidia.present",
                title="NVIDIA telemetry",
                outcome=Outcome.SKIP,
                severity=Severity.INFO,
                what="No NVIDIA GPU was detected on this machine.",
                why="setpoint currently only understands NVIDIA hardware.",
                fix="AMD, Intel and Apple support are not implemented yet. Rather than "
                "guess at your hardware, setpoint reports nothing.",
                evidence={"detail": snap.driver.detail},
            )
        ]

    gpu = snap.gpus[0]
    return [
        Finding(
            check_id="nvidia.present",
            title="NVIDIA telemetry",
            outcome=Outcome.PASS,
            severity=Severity.INFO,
            what=f"{gpu.name}, {gpu.vram_total_mib} MiB VRAM, driver {snap.driver.driver_version}.",
            evidence={
                "gpu": gpu.name,
                "vram_total_mib": gpu.vram_total_mib,
                "driver": snap.driver.driver_version,
                "compute_capability": (
                    f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
                    if gpu.compute_capability
                    else None
                ),
            },
        )
    ]


def check_cuda_support(snap: HardwareSnapshot) -> list[Finding]:
    """Does this driver support the CUDA runtime that prebuilt backends need?

    The supported CUDA version is read from NVML rather than mapped from driver
    numbers, since the driver is the authority on what it supports.
    """
    if snap.driver.status is not ProbeStatus.OK or snap.driver.cuda_driver_major is None:
        return [
            Finding(
                check_id="driver.cuda-support",
                title="CUDA runtime support",
                outcome=Outcome.SKIP,
                severity=Severity.INFO,
                what="The driver did not report a supported CUDA version.",
            )
        ]

    supported = (snap.driver.cuda_driver_major, snap.driver.cuda_driver_minor or 0)
    evidence = {
        "driver": snap.driver.driver_version,
        "cuda_supported": snap.driver.cuda_driver_version,
        "cuda_required_by_prebuilt": f"{MIN_CUDA_FOR_PREBUILT[0]}.{MIN_CUDA_FOR_PREBUILT[1]}",
    }

    if supported < MIN_CUDA_FOR_PREBUILT:
        # A driver too old for CUDA prebuilts only matters if CUDA is what you run.
        # With a working non-CUDA backend installed, this is history, not a fault.
        installed = backend_devices()
        kinds = sorted({d.kind for d in installed}) if installed else []
        non_cuda = [k for k in kinds if k.lower() not in ("cuda", "cpu")]
        evidence["backend_devices"] = kinds

        if non_cuda:
            return [
                Finding(
                    check_id="driver.cuda-support",
                    title="CUDA runtime support",
                    outcome=Outcome.PASS,
                    severity=Severity.INFO,
                    what=f"Driver {snap.driver.driver_version} is too old for prebuilt "
                    f"CUDA {MIN_CUDA_FOR_PREBUILT[0]}.x binaries, but the installed "
                    f"backend runs on {', '.join(non_cuda)} and does not need them.",
                    why="Which backend a measurement came from is recorded in the "
                    "profile signature, so a profile measured here stays honest about "
                    "what produced it.",
                    evidence=evidence,
                )
            ]

        return [
            Finding(
                check_id="driver.cuda-support",
                title="CUDA runtime support",
                outcome=Outcome.FAIL,
                severity=Severity.CRITICAL,
                what=f"Driver {snap.driver.driver_version} supports up to CUDA "
                f"{snap.driver.cuda_driver_version}, but official llama.cpp prebuilt "
                f"CUDA binaries are built against CUDA {MIN_CUDA_FOR_PREBUILT[0]}.x.",
                why="Those binaries will refuse to start, or fall back to CPU-only "
                "execution, on this driver. CPU-only inference is typically 5-20x "
                "slower than GPU inference -- and the fallback is often silent.",
                fix="Update the NVIDIA driver, or use a llama.cpp build that does not "
                "need a recent CUDA runtime: the Vulkan release binaries run on this "
                "driver generation. Building against the CUDA toolkit the driver does "
                "support also works, and costs more time.",
                evidence=evidence,
            )
        ]

    return [
        Finding(
            check_id="driver.cuda-support",
            title="CUDA runtime support",
            outcome=Outcome.PASS,
            severity=Severity.INFO,
            what=f"Driver supports CUDA {snap.driver.cuda_driver_version}, "
            "new enough for prebuilt CUDA backends.",
            evidence=evidence,
        )
    ]


def check_sysmem_fallback(snap: HardwareSnapshot) -> list[Finding]:
    """The silent VRAM spill.

    Two signals reported separately: whether the driver has the sysmem fallback policy
    at all, which the driver version settles, and whether shared system memory is
    currently backing GPU allocations, which is inconclusive while the machine is idle.
    """
    findings: list[Finding] = []
    driver_tuple = snap.driver.driver_tuple

    if driver_tuple is None:
        findings.append(
            Finding(
                check_id="vram.sysmem-policy",
                title="CUDA sysmem fallback policy",
                outcome=Outcome.SKIP,
                severity=Severity.INFO,
                what="Driver version unavailable, so policy support could not be determined.",
            )
        )
    elif driver_tuple < SYSMEM_FALLBACK_DRIVER:
        findings.append(
            Finding(
                check_id="vram.sysmem-policy",
                title="CUDA sysmem fallback policy",
                outcome=Outcome.PASS,
                severity=Severity.INFO,
                what=f"Driver {snap.driver.driver_version} predates the sysmem fallback "
                f"policy (added in {SYSMEM_FALLBACK_DRIVER[0]}.{SYSMEM_FALLBACK_DRIVER[1]}).",
                why="On this driver a CUDA allocation that does not fit in VRAM fails "
                "with out-of-memory instead of silently spilling to system RAM. That is "
                "louder, and for tuning purposes safer, than the newer behaviour.",
                evidence={
                    "driver": snap.driver.driver_version,
                    "policy_available": False,
                },
            )
        )
    else:
        findings.append(
            Finding(
                check_id="vram.sysmem-policy",
                title="CUDA sysmem fallback policy",
                outcome=Outcome.SKIP,
                severity=Severity.WARNING,
                what=f"Driver {snap.driver.driver_version} has the sysmem fallback policy, "
                "but setpoint cannot yet read its current value.",
                why="If the policy is set to 'Prefer Sysmem Fallback', allocations that "
                "overflow VRAM are silently backed by system RAM. Throughput drops "
                "sharply and nothing reports an error.",
                fix="Check NVIDIA Control Panel > Manage 3D Settings > "
                "CUDA - Sysmem Fallback Policy. For tuning, 'Prefer No Sysmem Fallback' "
                "makes overflow fail loudly instead of silently.",
                evidence={
                    "driver": snap.driver.driver_version,
                    "policy_available": True,
                    "reader_implemented": False,
                },
            )
        )

    findings.append(_shared_memory_observation(snap))
    return findings


def _shared_memory_observation(snap: HardwareSnapshot) -> Finding:
    if not snap.adapters:
        return Finding(
            check_id="vram.shared-usage",
            title="Shared system memory in use by GPU",
            outcome=Outcome.SKIP,
            severity=Severity.INFO,
            what="WDDM adapter memory counters were not available.",
            evidence={"notes": list(snap.notes)},
        )

    sample = snap.sample_for(0)
    adapter, confident = match_adapter(snap.adapters, sample.vram_used_bytes if sample else None)
    if adapter is None:
        return Finding(
            check_id="vram.shared-usage",
            title="Shared system memory in use by GPU",
            outcome=Outcome.SKIP,
            severity=Severity.INFO,
            what="No usable WDDM adapter instance was found.",
        )

    shared = adapter.shared_bytes or 0
    dedicated = adapter.dedicated_bytes or 0
    attribution = (
        "matched to the NVIDIA GPU by memory usage"
        if confident
        else (
            "NOT confidently matched to the NVIDIA GPU -- this is the adapter using the most "
            "dedicated memory, which on a laptop may be the integrated GPU"
        )
    )

    return Finding(
        check_id="vram.shared-usage",
        title="Shared system memory in use by GPU",
        outcome=Outcome.PASS,
        severity=Severity.INFO,
        what=f"Adapter is currently using {_gib(dedicated)} dedicated and "
        f"{_gib(shared)} shared memory ({attribution}).",
        why="Shared memory is system RAM reached over PCIe. A model that spills into it "
        "keeps reporting as GPU-resident while running far slower. Note that a non-zero "
        "value at idle is normal -- the desktop compositor and browsers use it too.",
        fix="This reading is taken at idle and cannot by itself prove a spill. Run "
        "`setpoint bench` under load: a spill shows up as dedicated memory pinned at "
        "capacity while shared memory climbs.",
        evidence={
            "adapter": adapter.instance,
            "dedicated_bytes": dedicated,
            "shared_bytes": shared,
            "confident_match": confident,
            "measured_at": "idle",
        },
    )


def check_vram_headroom(snap: HardwareSnapshot) -> list[Finding]:
    if not snap.gpus:
        return []

    gpu = snap.gpus[0]
    sample = snap.sample_for(gpu.index)
    if sample is None or sample.vram_used_bytes is None or gpu.vram_total_bytes <= 0:
        return [
            Finding(
                check_id="vram.headroom",
                title="Usable VRAM budget",
                outcome=Outcome.SKIP,
                severity=Severity.INFO,
                what="VRAM usage could not be read.",
            )
        ]

    used = sample.vram_used_bytes
    ratio = used / gpu.vram_total_bytes
    evidence = {
        "vram_total_mib": gpu.vram_total_bytes // MIB,
        "vram_used_mib": used // MIB,
        "vram_free_mib": (sample.vram_free_bytes or 0) // MIB,
        "idle_overhead_ratio": round(ratio, 4),
    }

    if ratio > IDLE_OVERHEAD_WARN_RATIO:
        return [
            Finding(
                check_id="vram.headroom",
                title="Usable VRAM budget",
                outcome=Outcome.FAIL,
                severity=Severity.WARNING,
                what=f"{used // MIB} MiB of {gpu.vram_total_bytes // MIB} MiB VRAM "
                f"({ratio:.0%}) is already in use before any model is loaded.",
                why="Every megabyte held by the desktop, a browser or another process is "
                "a megabyte the model cannot use. On a small card this is the difference "
                "between fitting a layer and spilling it.",
                fix="Close GPU-accelerated applications (browsers are the usual culprit) "
                "before tuning, or pass --reserve to tell setpoint to plan around them.",
                evidence=evidence,
            )
        ]

    return [
        Finding(
            check_id="vram.headroom",
            title="Usable VRAM budget",
            outcome=Outcome.PASS,
            severity=Severity.INFO,
            what=f"{(sample.vram_free_bytes or 0) // MIB} MiB of "
            f"{gpu.vram_total_bytes // MIB} MiB VRAM is free "
            f"(idle overhead {ratio:.1%}).",
            evidence=evidence,
        )
    ]


def check_throttle(snap: HardwareSnapshot) -> list[Finding]:
    if not snap.gpus:
        return []

    sample = snap.sample_for(snap.gpus[0].index)
    if sample is None:
        return []

    active = tuple(r for r in sample.throttle_reasons if r in ACTIVE_THROTTLE_REASONS)
    evidence = {
        "throttle_reasons": list(sample.throttle_reasons),
        "temperature_c": sample.temperature_c,
        "power_w": sample.power_w,
        "measured_at": "idle",
    }

    if active:
        return [
            Finding(
                check_id="gpu.throttle",
                title="Clock throttling",
                outcome=Outcome.FAIL,
                severity=Severity.WARNING,
                what=f"The GPU is being throttled right now: {', '.join(active)}.",
                why="A throttled GPU produces lower and less repeatable throughput, "
                "which also makes any measurement taken now untrustworthy.",
                fix="Let the machine cool, check power settings, and re-run. setpoint "
                "will refuse to write a calibration profile from throttled measurements.",
                evidence=evidence,
            )
        ]

    return [
        Finding(
            check_id="gpu.throttle",
            title="Clock throttling",
            outcome=Outcome.PASS,
            severity=Severity.INFO,
            what="No active thermal or power throttling."
            + (f" GPU at {sample.temperature_c} C." if sample.temperature_c else ""),
            evidence=evidence,
        )
    ]


def check_pcie_link(snap: HardwareSnapshot) -> list[Finding]:
    """PCIe link width and generation.

    Reported as context, never as a failure: at idle the link deliberately drops to a
    narrow, slow state to save power, and flagging that would be a false alarm.
    """
    if not snap.gpus:
        return []

    gpu = snap.gpus[0]
    sample = snap.sample_for(gpu.index)
    if sample is None or sample.pcie_width is None or gpu.max_pcie_width is None:
        return [
            Finding(
                check_id="gpu.pcie-link",
                title="PCIe link",
                outcome=Outcome.SKIP,
                severity=Severity.INFO,
                what="PCIe link state could not be read.",
            )
        ]

    return [
        Finding(
            check_id="gpu.pcie-link",
            title="PCIe link",
            outcome=Outcome.PASS,
            severity=Severity.INFO,
            what=f"Link is x{sample.pcie_width} gen{sample.pcie_gen} at idle "
            f"(maximum x{gpu.max_pcie_width} gen{gpu.max_pcie_gen}).",
            why="PCIe bandwidth sets the cost of every byte moved between system RAM and "
            "VRAM, which is exactly what offloaded layers and expert caches do. Idle "
            "downshift is normal power saving, not a fault -- the link state under load "
            "is the one that matters.",
            evidence={
                "current_width": sample.pcie_width,
                "current_gen": sample.pcie_gen,
                "max_width": gpu.max_pcie_width,
                "max_gen": gpu.max_pcie_gen,
                "measured_at": "idle",
            },
        )
    ]


def backend_devices() -> tuple[BackendDevice, ...] | None:
    """What the installed backend can run on, or `None` if it could not be asked."""
    backend = LlamaCppBackend()
    if not backend.available:
        return None
    try:
        return backend.devices()
    except BackendError:
        return None


def check_backend(snap: HardwareSnapshot) -> list[Finding]:
    """Is there a llama.cpp we can actually measure with, and on what?"""
    binaries = {name: shutil.which(name) for name in ("llama-bench", "llama-server", "llama-cli")}
    found = {k: v for k, v in binaries.items() if v}
    ollama = shutil.which("ollama")

    # find_binary also honours the override, so a binary outside PATH still counts.
    bench = find_binary()
    evidence: dict[str, object] = {"llama_cpp": found, "ollama": ollama, "llama_bench": str(bench)}

    if bench is not None:
        devices = backend_devices()
        findings = [_backend_present(bench, devices, evidence)]
        if devices:
            findings.extend(_device_choice(devices))
        return findings

    what = "llama-bench was not found on PATH."
    if ollama:
        what += " Ollama is installed, but it does not expose llama-bench."

    return [
        Finding(
            check_id="backend.llama-cpp",
            title="llama.cpp backend",
            outcome=Outcome.FAIL,
            severity=Severity.WARNING,
            what=what,
            why="setpoint tunes by measuring, and llama-bench is how it measures. "
            "Without it, only `setpoint budget` (static estimation) can run.",
            fix="Install a llama.cpp release build and put its binaries on PATH, or "
            f"point setpoint at one with {BINARY_ENV_VAR}: "
            "https://github.com/ggml-org/llama.cpp/releases",
            evidence=evidence,
        )
    ]


def _backend_present(
    bench: object, devices: tuple[BackendDevice, ...] | None, evidence: dict[str, object]
) -> Finding:
    what = f"llama-bench found at {bench}."
    if devices:
        kinds = sorted({d.kind for d in devices})
        what += f" It offers {len(devices)} device(s) through {', '.join(kinds)}."
        evidence["devices"] = [
            {"id": d.id, "name": d.name, "total_mib": d.total_mib} for d in devices
        ]
    elif devices is None:
        what += " It could not be asked which devices it offers."
    return Finding(
        check_id="backend.llama-cpp",
        title="llama.cpp backend",
        outcome=Outcome.PASS,
        severity=Severity.INFO,
        what=what,
        evidence=evidence,
    )


def _device_choice(devices: tuple[BackendDevice, ...]) -> list[Finding]:
    """More than one accelerator means the backend has to choose, and it may choose badly.

    An integrated GPU accepts the work and returns a number. That number describes the
    integrated GPU, whatever the profile says it describes.
    """
    if len(devices) < 2:
        return [
            Finding(
                check_id="backend.device-choice",
                title="Accelerator selection",
                outcome=Outcome.PASS,
                severity=Severity.INFO,
                what=f"Only one accelerator is available ({devices[0].name}), "
                "so there is nothing to choose wrongly.",
                evidence={"devices": [d.id for d in devices]},
            )
        ]

    listed = ", ".join(f"{d.id} {d.name} ({d.total_mib} MiB)" for d in devices)
    return [
        Finding(
            check_id="backend.device-choice",
            title="Accelerator selection",
            outcome=Outcome.FAIL,
            severity=Severity.WARNING,
            what=f"The backend offers {len(devices)} accelerators: {listed}.",
            why="A run that does not name one lets the backend pick, and an integrated "
            "GPU will accept the work and report a throughput that describes itself "
            "rather than the card you meant to measure.",
            fix="setpoint names the device on every run it makes. When you invoke "
            "llama.cpp yourself, pass -dev explicitly.",
            evidence={
                "devices": [{"id": d.id, "name": d.name, "total_mib": d.total_mib} for d in devices]
            },
        )
    ]


ALL_CHECKS = (
    check_nvidia_present,
    check_cuda_support,
    check_sysmem_fallback,
    check_vram_headroom,
    check_throttle,
    check_pcie_link,
    check_backend,
)
