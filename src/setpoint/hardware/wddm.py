"""Windows WDDM adapter memory probe.

When a GPU allocation no longer fits in dedicated VRAM, Windows can back it with
system RAM. NVML still reports it as GPU memory, so the only visible signal is the
performance counter for shared adapter memory.

Read through PowerShell Get-Counter to avoid an extra dependency. It costs about a
second, so it belongs at run boundaries rather than inside a sampling loop.
"""

from __future__ import annotations

import json
import platform
import subprocess

from .types import AdapterMemory, ProbeStatus

_COUNTER_PATHS = (
    r"\GPU Adapter Memory(*)\Dedicated Usage",
    r"\GPU Adapter Memory(*)\Shared Usage",
    r"\GPU Adapter Memory(*)\Total Committed",
)

# ErrorAction Stop matters: a missing counter set must fail loudly rather than return
# empty output, which would read as an absence of spill.
_PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$paths = @({paths})
$samples = (Get-Counter -Counter $paths).CounterSamples
$out = foreach ($s in $samples) {
    [pscustomobject]@{
        path     = $s.Path
        instance = $s.InstanceName
        value    = [double]$s.CookedValue
    }
}
$out | ConvertTo-Json -Compress -Depth 3
"""


def is_supported() -> bool:
    return platform.system() == "Windows"


def _counter_kind(path: str) -> str | None:
    lowered = path.lower()
    if "dedicated usage" in lowered:
        return "dedicated"
    if "shared usage" in lowered:
        return "shared"
    if "total committed" in lowered:
        return "committed"
    return None


def probe(timeout: float = 20.0) -> tuple[ProbeStatus, tuple[AdapterMemory, ...], str | None]:
    """Read per-adapter WDDM memory usage.

    Returns (status, adapters, detail). On failure the adapter tuple is empty and
    detail explains why. A zero is never substituted, since zero shared usage is a
    claim that would wrongly clear the machine of spilling.
    """
    if not is_supported():
        return (
            ProbeStatus.UNSUPPORTED,
            (),
            f"WDDM counters are Windows-only (this is {platform.system()})",
        )

    quoted = ",".join(f"'{p}'" for p in _COUNTER_PATHS)
    script = _PS_SCRIPT.replace("{paths}", quoted)

    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return ProbeStatus.UNAVAILABLE, (), "powershell not found on PATH"
    except subprocess.TimeoutExpired:
        return ProbeStatus.UNAVAILABLE, (), f"Get-Counter timed out after {timeout:g}s"

    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        return ProbeStatus.ERROR, (), detail[0] if detail else f"exit code {proc.returncode}"

    raw = (proc.stdout or "").strip()
    if not raw:
        return ProbeStatus.ERROR, (), "Get-Counter returned no output"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ProbeStatus.ERROR, (), f"could not parse Get-Counter output: {exc}"

    # ConvertTo-Json collapses a single-element array into a bare object.
    rows = parsed if isinstance(parsed, list) else [parsed]

    merged: dict[str, dict[str, int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = _counter_kind(str(row.get("path", "")))
        instance = str(row.get("instance", "")).strip()
        if kind is None or not instance:
            continue
        try:
            value = int(float(row.get("value", 0)))
        except (TypeError, ValueError):
            continue
        merged.setdefault(instance, {})[kind] = value

    adapters = tuple(
        AdapterMemory(
            instance=instance,
            dedicated_bytes=fields.get("dedicated"),
            shared_bytes=fields.get("shared"),
            committed_bytes=fields.get("committed"),
        )
        for instance, fields in sorted(merged.items())
    )
    if not adapters:
        return ProbeStatus.ERROR, (), "no GPU adapter instances found in counter output"
    return ProbeStatus.OK, adapters, None


def match_adapter(
    adapters: tuple[AdapterMemory, ...],
    nvml_used_bytes: int | None,
    tolerance_bytes: int = 256 * 1024 * 1024,
) -> tuple[AdapterMemory | None, bool]:
    """Guess which WDDM adapter corresponds to the GPU NVML reported on.

    Counters are keyed by LUID, which NVML does not expose, so the match is made on
    dedicated memory usage. Returns (adapter, confident); when confident is False the
    caller must present the result as a guess rather than a measurement.
    """
    usable = [a for a in adapters if a.dedicated_bytes is not None]
    if not usable:
        return None, False

    if nvml_used_bytes is not None:
        best = min(usable, key=lambda a: abs((a.dedicated_bytes or 0) - nvml_used_bytes))
        if abs((best.dedicated_bytes or 0) - nvml_used_bytes) <= tolerance_bytes:
            return best, True

    # Fall back to the adapter holding the most dedicated memory. Usually correct on a
    # laptop with an integrated and a discrete GPU, but it remains a guess.
    return max(usable, key=lambda a: a.dedicated_bytes or 0), False
