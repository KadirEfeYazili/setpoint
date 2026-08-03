"""Watching the machine while something else is using it.

The signal this exists for is a spill: a model that no longer fits keeps reporting as
GPU-resident while the driver quietly backs the overflow with system RAM. Throughput
collapses and nothing says so. It is visible in exactly one place - dedicated memory
pinned at capacity while shared memory climbs - and only under load.

The two readings come from different places at very different costs: NVML answers in
tens of milliseconds, the Windows adapter counters take nearly two seconds. They are
sampled on their own schedules and every reading carries its age, because a shared
figure from two seconds ago is a different claim from one taken now.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

from .hardware import GpuSample, match_adapter
from .hardware.nvml import NvmlProbe
from .hardware.wddm import probe as probe_wddm

MIB = 1024 * 1024

# Dedicated memory this close to capacity is the first half of a spill.
SPILL_DEDICATED_SHARE = 0.95

# Shared memory growing by this much over the window is the second half. Below it the
# desktop and the browser account for the movement on their own.
SPILL_SHARED_GROWTH_BYTES = 64 * MIB

# Below this utilisation nothing is really running, and a spill only shows under load.
LOAD_UTILISATION_PCT = 20

# How far back the spill check looks, in readings.
WINDOW = 8

# Readings kept for the display.
HISTORY = 240


class Verdict(str, Enum):
    """What the readings support saying.

    `UNKNOWN` is not a failure: the shared-memory counter may be unavailable, or there
    may not be enough history yet. Not being able to tell and having looked are
    different claims.
    """

    UNKNOWN = "unknown"
    IDLE = "idle"
    HEALTHY = "healthy"
    SPILLING = "spilling"


@dataclass(frozen=True)
class Reading:
    """One moment, from both sources."""

    at: float
    vram_used_bytes: int | None = None
    vram_total_bytes: int | None = None
    shared_bytes: int | None = None
    shared_at: float | None = None
    shared_confident: bool = True
    utilization_pct: int | None = None
    temperature_c: int | None = None
    power_w: float | None = None
    throttle_reasons: tuple[str, ...] = ()

    @property
    def dedicated_share(self) -> float | None:
        if not self.vram_used_bytes or not self.vram_total_bytes:
            return None
        return self.vram_used_bytes / self.vram_total_bytes

    @property
    def shared_age_s(self) -> float | None:
        return None if self.shared_at is None else self.at - self.shared_at

    @property
    def under_load(self) -> bool:
        return (self.utilization_pct or 0) >= LOAD_UTILISATION_PCT


@dataclass(frozen=True)
class SpillCheck:
    verdict: Verdict
    detail: str
    dedicated_share: float | None = None
    shared_growth_bytes: int | None = None


def check_spill(history: list[Reading]) -> SpillCheck:
    """Read the last few samples for the spill signature.

    Both halves have to hold. Shared memory alone climbs whenever a browser opens a tab;
    dedicated memory alone sits near capacity on any card doing its job.
    """
    window = [r for r in history[-WINDOW:] if r.shared_bytes is not None]
    if len(window) < 2:
        return SpillCheck(Verdict.UNKNOWN, "not enough readings with a shared-memory figure")

    latest = window[-1]
    share = latest.dedicated_share
    growth = latest.shared_bytes - min(r.shared_bytes for r in window)

    if share is None:
        return SpillCheck(Verdict.UNKNOWN, "dedicated memory could not be read")

    if not any(r.under_load for r in window):
        return SpillCheck(
            Verdict.IDLE,
            "nothing is loading the GPU; a spill only shows under load",
            share,
            growth,
        )

    if share >= SPILL_DEDICATED_SHARE and growth >= SPILL_SHARED_GROWTH_BYTES:
        return SpillCheck(
            Verdict.SPILLING,
            f"dedicated memory is at {share:.0%} and shared has grown "
            f"{growth / MIB:.0f} MiB; the overflow is being served from system RAM",
            share,
            growth,
        )

    return SpillCheck(
        Verdict.HEALTHY,
        f"dedicated memory at {share:.0%}, shared steady within {growth / MIB:.0f} MiB",
        share,
        growth,
    )


class Monitor:
    """Samples both sources on their own schedules and keeps the recent history.

    The adapter counters are read on a background thread because one read costs about
    as long as two display frames; blocking the loop on them would make the fast
    numbers as slow as the slow one.
    """

    def __init__(self, gpu_index: int | None = None, history: int = HISTORY) -> None:
        self.gpu_index = gpu_index
        self.readings: deque[Reading] = deque(maxlen=history)
        self._shared: tuple[int, float, bool] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml: NvmlProbe | None = None
        self.notes: list[str] = []

    def __enter__(self) -> Monitor:
        self.notes = []
        self._stop.clear()
        self._nvml = NvmlProbe().__enter__()
        if not self._nvml.ok:
            self.notes.append(self._nvml.detail or "NVML was unavailable")
        self._thread = threading.Thread(target=self._shared_loop, name="setpoint-wddm", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._nvml is not None:
            self._nvml.__exit__()
            self._nvml = None

    def sample(self) -> Reading:
        """One reading now, combining the live NVML numbers with the latest shared figure."""
        now = time.time()
        gpu = self._gpu_sample()
        shared, shared_at, confident = self._shared or (None, None, True)
        return Reading(
            at=now,
            vram_used_bytes=gpu.vram_used_bytes if gpu else None,
            vram_total_bytes=self._total_bytes(),
            shared_bytes=shared,
            shared_at=shared_at,
            shared_confident=confident,
            utilization_pct=gpu.utilization_pct if gpu else None,
            temperature_c=gpu.temperature_c if gpu else None,
            power_w=gpu.power_w if gpu else None,
            throttle_reasons=gpu.throttle_reasons if gpu else (),
        )

    def tick(self) -> Reading:
        """Sample and remember."""
        reading = self.sample()
        self.readings.append(reading)
        return reading

    @property
    def spill(self) -> SpillCheck:
        return check_spill(list(self.readings))

    def _gpu_sample(self) -> GpuSample | None:
        if self._nvml is None or not self._nvml.ok:
            return None
        for sample in self._nvml.sample():
            if self.gpu_index is None or sample.index == self.gpu_index:
                return sample
        return None

    def _total_bytes(self) -> int | None:
        if self._nvml is None or not self._nvml.ok:
            return None
        for gpu in self._nvml.gpus():
            if self.gpu_index is None or gpu.index == self.gpu_index:
                return gpu.vram_total_bytes
        return None

    def _shared_loop(self) -> None:
        """Read the adapter counters as fast as they will answer, which is not fast."""
        while not self._stop.is_set():
            try:
                _, adapters, detail = probe_wddm()
                sample = self._gpu_sample()
                adapter, confident = match_adapter(
                    adapters, sample.vram_used_bytes if sample else None
                )
                if adapter is not None and adapter.shared_bytes is not None:
                    self._shared = (adapter.shared_bytes, time.time(), confident)
                elif detail and detail not in self.notes:
                    self.notes.append(detail)
            except Exception as exc:  # a stalled counter must not take the display down
                message = f"adapter counters stopped: {exc!r}"
                if message not in self.notes:
                    self.notes.append(message)
                return
            self._stop.wait(0.1)
