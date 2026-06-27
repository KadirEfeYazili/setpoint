"""Sampling the GPU while something else runs.

A benchmark reports throughput but says nothing about what the GPU was doing while it
produced it. Peak VRAM is the number that decides whether a configuration actually fit,
and it is only visible from outside the process that allocated it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .nvml import ACTIVE_THROTTLE_REASONS, NvmlProbe
from .types import GpuSample

# Fast enough to catch an allocation peak, slow enough not to disturb the run.
DEFAULT_INTERVAL_S = 0.25


@dataclass(frozen=True)
class GpuWatch:
    """What the GPU did during one run."""

    samples: int = 0
    peak_vram_bytes: int | None = None
    average_power_w: float | None = None
    peak_temperature_c: int | None = None
    throttle_reasons: tuple[str, ...] = field(default_factory=tuple)
    detail: str | None = None

    @property
    def peak_vram_mib(self) -> int | None:
        return None if self.peak_vram_bytes is None else self.peak_vram_bytes // (1024 * 1024)

    @property
    def throttled(self) -> bool:
        return bool(self.throttle_reasons)


class GpuWatcher:
    """Polls NVML on a background thread for as long as the `with` block runs.

    A failure to sample is recorded rather than raised: losing the peak VRAM reading is
    not a reason to lose the measurement it was taken alongside.
    """

    def __init__(
        self, gpu_index: int | None = None, interval_s: float = DEFAULT_INTERVAL_S
    ) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[GpuSample] = []
        self._detail: str | None = None

    def __enter__(self) -> GpuWatcher:
        self._stop.clear()
        self._samples = []
        self._detail = None
        self._thread = threading.Thread(target=self._loop, name="setpoint-gpu-watch", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 4)
            self._thread = None

    def _loop(self) -> None:
        try:
            with NvmlProbe() as probe:
                if not probe.ok:
                    self._detail = probe.detail or "NVML was unavailable"
                    return
                while not self._stop.is_set():
                    self._collect(probe.sample())
                    self._stop.wait(self.interval_s)
                # One last look, so a peak reached just before the end is not missed.
                self._collect(probe.sample())
        except Exception as exc:  # sampling must never take the run down with it
            self._detail = f"GPU sampling stopped: {exc!r}"

    def _collect(self, samples: tuple[GpuSample, ...]) -> None:
        for sample in samples:
            if self.gpu_index is None or sample.index == self.gpu_index:
                self._samples.append(sample)

    @property
    def result(self) -> GpuWatch:
        """The summary. Reading it before the block ends gives what has been seen so far."""
        if not self._samples:
            return GpuWatch(detail=self._detail or "no GPU samples were taken")

        vram = [s.vram_used_bytes for s in self._samples if s.vram_used_bytes is not None]
        power = [s.power_w for s in self._samples if s.power_w is not None]
        temps = [s.temperature_c for s in self._samples if s.temperature_c is not None]
        reasons = {
            reason
            for sample in self._samples
            for reason in sample.throttle_reasons
            if reason in ACTIVE_THROTTLE_REASONS
        }

        return GpuWatch(
            samples=len(self._samples),
            peak_vram_bytes=max(vram) if vram else None,
            average_power_w=sum(power) / len(power) if power else None,
            peak_temperature_c=max(temps) if temps else None,
            throttle_reasons=tuple(sorted(reasons)),
            detail=self._detail,
        )
