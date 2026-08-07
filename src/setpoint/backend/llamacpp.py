"""llama.cpp backend adapter, driven through llama-bench.

llama-bench is used rather than a timed llama-cli run because it already repeats each
test, discards a warmup pass, and reports every repetition individually. setpoint reads
those raw repetitions and derives its own median and spread; the mean and standard
deviation the tool prints are not used.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from ..measure import Statistic
from .types import (
    BackendBuild,
    BackendDevice,
    BackendError,
    BenchRun,
    BenchSample,
    MeasurementKind,
    RunSpec,
)

BINARY_NAME = "llama-bench"
SERVER_BINARY_NAME = "llama-server"

# Points setpoint at a binary that is not on PATH.
BINARY_ENV_VAR = "SETPOINT_LLAMA_BENCH"
SERVER_ENV_VAR = "SETPOINT_LLAMA_SERVER"

# A deep-context run repeats a long prefill several times, so the ceiling is generous.
DEFAULT_TIMEOUT_S = 900.0

# Listing devices loads every backend library but no model, so it is quick.
PROBE_TIMEOUT_S = 30.0

# "  Vulkan1: NVIDIA GeForce GTX 1650 (4176 MiB, 3581 MiB free)"
_DEVICE_LINE = re.compile(
    r"^ {2}(?P<id>\w+):\s+(?P<name>.+?)"
    r"(?:\s+\((?P<total>\d+)\s*MiB(?:,\s*(?P<free>\d+)\s*MiB free)?\))?\s*$"
)

# Flash attention took 0|1 before it took on|off|auto. Builds older than this are not
# driven; the flag is spelled the modern way.
MIN_SUPPORTED_BUILD = 6000


def find_binary(explicit: str | Path | None = None) -> Path | None:
    """Locate llama-bench: an explicit path, then the environment, then PATH."""
    return _locate(BINARY_NAME, BINARY_ENV_VAR, explicit)


def find_server_binary(explicit: str | Path | None = None) -> Path | None:
    """Locate llama-server, which is what `run` starts.

    Falls back to looking beside llama-bench: release archives ship both together, and
    a user who pointed setpoint at one has almost certainly got the other.
    """
    found = _locate(SERVER_BINARY_NAME, SERVER_ENV_VAR, explicit)
    if found is not None:
        return found
    bench = find_binary()
    if bench is None:
        return None
    sibling = bench.with_name(SERVER_BINARY_NAME + bench.suffix)
    return sibling if sibling.is_file() else None


def _locate(name: str, env_var: str, explicit: str | Path | None) -> Path | None:
    for candidate in (explicit, os.environ.get(env_var)):
        if candidate:
            path = Path(candidate).expanduser()
            return path if path.is_file() else None
    found = shutil.which(name)
    return Path(found) if found else None


class LlamaCppBackend:
    """Runs one configuration and reports what it measured."""

    name = "llama.cpp"

    def __init__(self, binary: str | Path | None = None) -> None:
        self.binary = find_binary(binary)

    @property
    def available(self) -> bool:
        return self.binary is not None

    def build_argv(self, spec: RunSpec) -> list[str]:
        """The exact command line for a configuration.

        Warmup is deliberately left on: the first pass of any configuration is not a
        measurement, and skipping it would break the discipline the profiles rest on.
        """
        if self.binary is None:
            raise BackendError(
                f"{BINARY_NAME} was not found. Put it on PATH or set {BINARY_ENV_VAR}."
            )
        argv = [
            str(self.binary),
            "-m",
            str(spec.model_path),
            "-o",
            "json",
            "-r",
            str(spec.repetitions),
            "-p",
            str(spec.n_prompt),
            "-n",
            str(spec.n_gen),
        ]
        if spec.n_depth:
            argv += ["-d", str(spec.n_depth)]
        if spec.n_gpu_layers is not None:
            argv += ["-ngl", str(spec.n_gpu_layers)]
        if spec.n_cpu_moe is not None:
            argv += ["-ncmoe", str(spec.n_cpu_moe)]
        if spec.batch_size is not None:
            argv += ["-b", str(spec.batch_size)]
        if spec.ubatch_size is not None:
            argv += ["-ub", str(spec.ubatch_size)]
        if spec.threads is not None:
            argv += ["-t", str(spec.threads)]
        argv += ["-ctk", spec.cache_type_k, "-ctv", spec.cache_type_v]
        if spec.flash_attn is not None:
            argv += ["-fa", "on" if spec.flash_attn else "off"]
        if spec.main_gpu is not None:
            argv += ["-mg", str(spec.main_gpu)]
        if spec.devices:
            argv += ["-dev", "/".join(spec.devices)]
        for override in spec.tensor_overrides:
            argv += ["-ot", override]
        return argv

    def devices(self, timeout_s: float = PROBE_TIMEOUT_S) -> tuple[BackendDevice, ...]:
        """Ask the backend what it can run on, without loading a model.

        Which accelerator a run lands on is not a detail. An integrated GPU sitting
        beside a discrete one will accept the work and return a number that says
        nothing about the card the profile is filed under.
        """
        if self.binary is None:
            raise BackendError(f"{BINARY_NAME} was not found")
        try:
            completed = subprocess.run(
                [str(self.binary), "--list-devices"],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise BackendError(f"{BINARY_NAME} could not list its devices: {exc}") from exc
        if completed.returncode != 0:
            raise BackendError(f"{BINARY_NAME} exited {completed.returncode} listing devices")
        return parse_devices(f"{completed.stdout}\n{completed.stderr}")

    def run(self, spec: RunSpec, timeout_s: float = DEFAULT_TIMEOUT_S) -> BenchRun:
        """Measure one configuration. Raises `BackendError` if the run did not produce data."""
        argv = self.build_argv(spec)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout_s, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"{BINARY_NAME} did not finish within {timeout_s:.0f}s") from exc
        except OSError as exc:
            raise BackendError(f"{BINARY_NAME} could not be started: {exc}") from exc
        elapsed = time.perf_counter() - started

        if completed.returncode != 0:
            raise BackendError(
                f"{BINARY_NAME} exited with code {completed.returncode}: {_tail(completed.stderr)}"
            )
        return parse_output(completed.stdout, spec, command=tuple(argv), duration_s=elapsed)


def parse_output(
    stdout: str,
    spec: RunSpec,
    command: tuple[str, ...] = (),
    duration_s: float | None = None,
) -> BenchRun:
    """Turn llama-bench JSON into a `BenchRun`."""
    records = _load_records(stdout)
    if not records:
        raise BackendError(f"{BINARY_NAME} produced no results")

    notes: list[str] = []
    samples = tuple(_sample(record, notes) for record in records)
    first = records[0]
    build = BackendBuild(
        name=LlamaCppBackend.name,
        commit=_str(first, "build_commit"),
        number=_int(first, "build_number"),
        accelerators=_str(first, "backends"),
    )
    used = _str(first, "devices")
    if spec.devices and used and used != "/".join(spec.devices):
        notes.append(
            f"The run was asked for {'/'.join(spec.devices)} but reports {used}; the "
            "measurement may not describe the hardware it is filed under."
        )
    if build.number is not None and build.number < MIN_SUPPORTED_BUILD:
        notes.append(
            f"Build {build.number} is older than the versions setpoint drives; flag "
            "spellings may differ and the result may not mean what it says."
        )

    return BenchRun(
        spec=spec,
        build=build,
        samples=samples,
        gpu_info=_str(first, "gpu_info"),
        cpu_info=_str(first, "cpu_info"),
        devices=used,
        command=command,
        duration_s=duration_s,
        notes=tuple(dict.fromkeys(notes)),
    )


def server_argv(
    binary: str | Path,
    model_path: str | Path,
    config: object,
    context: int,
    devices: tuple[str, ...] = (),
    extra: tuple[str, ...] = (),
) -> list[str]:
    """Command line for llama-server from a profile's configuration.

    Long flag names throughout: the short forms differ between llama.cpp's tools and
    have moved between releases, while the long ones have held still.
    """
    argv = [str(binary), "--model", str(model_path), "--ctx-size", str(context)]
    pairs = (
        ("--n-gpu-layers", getattr(config, "n_gpu_layers", None)),
        ("--n-cpu-moe", getattr(config, "n_cpu_moe", None)),
        ("--batch-size", getattr(config, "batch_size", None)),
        ("--ubatch-size", getattr(config, "ubatch_size", None)),
        ("--threads", getattr(config, "threads", None)),
        ("--spec-type", getattr(config, "spec_type", None)),
    )
    for flag, value in pairs:
        if value is not None:
            argv += [flag, str(value)]
    argv += [
        "--cache-type-k",
        str(getattr(config, "cache_type_k", "f16")),
        "--cache-type-v",
        str(getattr(config, "cache_type_v", "f16")),
    ]
    flash = getattr(config, "flash_attn", None)
    if flash is not None:
        argv += ["--flash-attn", "on" if flash else "off"]
    if devices:
        argv += ["--device", "/".join(devices)]
    for override in getattr(config, "tensor_overrides", ()) or ():
        argv += ["--override-tensor", str(override)]
    return argv + list(extra)


def select_device(devices: tuple[BackendDevice, ...], prefer: str | None) -> BackendDevice | None:
    """Resolve a device preference against what the backend offers right now.

    Device ids are positional and are not stable: a reboot can renumber them, so an id
    written down earlier may name different hardware today. Matching on the name is what
    keeps a measurement attached to the card it claims to describe.
    """
    if not devices:
        return None
    if not prefer:
        return devices[0]
    for device in devices:
        if device.id.lower() == prefer.lower():
            return device
    needle = prefer.lower()
    for device in devices:
        if needle in device.name.lower() or device.name.lower() in needle:
            return device
    return None


def parse_devices(text: str) -> tuple[BackendDevice, ...]:
    """Read the device listing. Everything the backend prints before it is noise."""
    devices: list[BackendDevice] = []
    listing = False
    for line in text.splitlines():
        if not listing:
            listing = line.strip().lower().startswith("available devices")
            continue
        match = _DEVICE_LINE.match(line.rstrip())
        if match is None:
            if line.strip():
                break
            continue
        devices.append(
            BackendDevice(
                id=match["id"],
                name=match["name"].strip(),
                total_mib=int(match["total"]) if match["total"] else None,
                free_mib=int(match["free"]) if match["free"] else None,
            )
        )
    return tuple(devices)


def _load_records(stdout: str) -> list[dict[str, object]]:
    """Pull the JSON array out of stdout, ignoring anything printed around it."""
    start, end = stdout.find("["), stdout.rfind("]")
    if start < 0 or end < start:
        raise BackendError(f"{BINARY_NAME} output held no JSON array")
    try:
        records = json.loads(stdout[start : end + 1])
    except json.JSONDecodeError as exc:
        raise BackendError(f"{BINARY_NAME} output was not valid JSON: {exc}") from exc
    if not isinstance(records, list):
        raise BackendError(f"{BINARY_NAME} output was not a list of results")
    return [r for r in records if isinstance(r, dict)]


def _sample(record: dict[str, object], notes: list[str]) -> BenchSample:
    n_prompt = _int(record, "n_prompt") or 0
    n_gen = _int(record, "n_gen") or 0
    kind = MeasurementKind.DECODE if n_gen else MeasurementKind.PREFILL

    throughput = _series(record, "samples_ts", "avg_ts", notes)
    duration = _series(record, "samples_ns", "avg_ns", notes)
    return BenchSample(
        kind=kind,
        n_prompt=n_prompt,
        n_gen=n_gen,
        n_depth=_int(record, "n_depth") or 0,
        throughput=throughput,
        duration_ns=duration,
        reported_mean_ts=_float(record, "avg_ts"),
    )


def _series(record: dict[str, object], key: str, mean_key: str, notes: list[str]) -> Statistic:
    """Prefer the raw repetitions; fall back to the mean and say so."""
    values = record.get(key)
    if isinstance(values, list) and values:
        return Statistic(tuple(float(v) for v in values))
    mean = _float(record, mean_key)
    if mean is None:
        return Statistic(())
    notes.append(
        "This build reported no per-repetition samples, so the spread could not be "
        "checked and the result cannot back a profile."
    )
    return Statistic((mean,))


# Lines worth surfacing when a run fails. Everything else the backend prints on the way
# up is library loading, which explains nothing about why it stopped.
_ERROR_MARKERS = ("error", "failed", "out of memory", "cannot", "unable")


def _tail(text: str, lines: int = 3) -> str:
    """The part of the backend's output that says why it stopped."""
    stripped = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not stripped:
        return "no output"
    interesting = [
        line for line in stripped if any(marker in line.lower() for marker in _ERROR_MARKERS)
    ]
    chosen = (interesting or stripped)[-lines:]
    return " / ".join(chosen)


def _str(record: dict[str, object], key: str) -> str | None:
    value = record.get(key)
    return value if isinstance(value, str) and value else None


def _int(record: dict[str, object], key: str) -> int | None:
    value = record.get(key)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _float(record: dict[str, object], key: str) -> float | None:
    value = record.get(key)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
