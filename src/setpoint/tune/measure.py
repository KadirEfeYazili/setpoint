"""Turning a configuration into a measured trial.

This is where the search meets the machine. Three things happen around every run that
the backend does not do for us: the thermal state is read before it starts, the GPU is
sampled while it runs, and the result is scored against the objective the user asked
for. A run taken while the card was throttling is labelled, not discarded, because the
label is what lets a reader distinguish a bad configuration from a bad moment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ..backend import BackendError, BenchRun, LlamaCppBackend, MeasurementKind, RunSpec
from ..hardware import ACTIVE_THROTTLE_REASONS, GpuWatcher, NvmlProbe
from ..profile import Config, Objective
from .types import Effort, Trial

# How long to wait for a throttling card to settle before measuring anyway.
SETTLE_TIMEOUT_S = 30.0
SETTLE_POLL_S = 2.0


@dataclass
class BackendMeasure:
    """Measures one configuration with a real backend, and scores it.

    `vram_ceiling_bytes` only means anything for the headroom objective, where staying
    inside the budget is the point rather than a nice-to-have.
    """

    backend: LlamaCppBackend
    model_path: Path
    objective: Objective = Objective.SPEED
    devices: tuple[str, ...] = ()
    gpu_index: int | None = None
    vram_ceiling_bytes: int | None = None
    wait_for_throttle: bool = True
    timeout_s: float | None = None
    runs: list[BenchRun] = field(default_factory=list)

    def __call__(self, config: Config, effort: Effort) -> Trial:
        before = wait_until_settled(
            self.gpu_index, timeout_s=SETTLE_TIMEOUT_S if self.wait_for_throttle else 0.0
        )
        spec = run_spec(config, effort, self.model_path, self.devices)

        watcher = GpuWatcher(self.gpu_index)
        try:
            with watcher:
                extra = {"timeout_s": self.timeout_s} if self.timeout_s else {}
                run = self.backend.run(spec, **extra)
        except BackendError as exc:
            return Trial(config=config, detail=str(exc))

        self.runs.append(run)
        watch = watcher.result
        notes = list(run.notes)
        if before:
            notes.append(f"The card was throttling before the run started: {', '.join(before)}.")
        if watch.throttled:
            notes.append(f"The card throttled during the run: {', '.join(watch.throttle_reasons)}.")
        if watch.detail:
            notes.append(watch.detail)

        score, refusal = self.score(run, watch.peak_vram_bytes, watch.average_power_w)
        if refusal:
            notes.append(refusal)

        sample = run.sample_of(_measured_side(self.objective))
        return Trial(
            config=config,
            score=score,
            spread=sample.throughput.spread if sample else None,
            reliable=run.reliable and not watch.throttled and score is not None,
            detail=" ".join(notes),
            run=run,
            watch=watch,
        )

    def score(
        self, run: BenchRun, peak_vram_bytes: int | None, average_power_w: float | None
    ) -> tuple[float | None, str | None]:
        """The number the search maximises, and why there is none when there is none."""
        decode = run.decode_tokens_per_second
        prefill = run.prefill_tokens_per_second

        if self.objective is Objective.LATENCY:
            return (prefill, None) if prefill else (None, "no prefill measurement")

        if decode is None:
            return None, "no decode measurement"

        if self.objective is Objective.EFFICIENCY:
            if not average_power_w:
                return None, "power could not be read, so efficiency cannot be scored"
            return decode / average_power_w, None

        if self.objective is Objective.HEADROOM:
            if self.vram_ceiling_bytes is None:
                return decode, "no VRAM ceiling was given, so headroom was not enforced"
            if peak_vram_bytes is None:
                return None, "peak VRAM could not be read, so headroom cannot be judged"
            if peak_vram_bytes > self.vram_ceiling_bytes:
                over = (peak_vram_bytes - self.vram_ceiling_bytes) / (1024 * 1024)
                return None, f"peak VRAM went {over:.0f} MiB past the ceiling"

        return decode, None


def run_spec(
    config: Config, effort: Effort, model_path: Path, devices: tuple[str, ...] = ()
) -> RunSpec:
    """Map a backend-neutral configuration onto one backend invocation."""
    return RunSpec(
        model_path=model_path,
        n_gpu_layers=config.n_gpu_layers,
        n_cpu_moe=config.n_cpu_moe,
        n_depth=effort.n_depth,
        batch_size=config.batch_size,
        ubatch_size=config.ubatch_size,
        threads=config.threads,
        cache_type_k=config.cache_type_k,
        cache_type_v=config.cache_type_v,
        flash_attn=config.flash_attn,
        devices=devices,
        tensor_overrides=config.tensor_overrides,
        repetitions=effort.repetitions,
    )


def read_throttle(gpu_index: int | None = None) -> tuple[str, ...]:
    """Throttle reasons the card reports right now. Empty also means "could not read"."""
    try:
        with NvmlProbe() as probe:
            if not probe.ok:
                return ()
            for sample in probe.sample():
                if gpu_index is None or sample.index == gpu_index:
                    return tuple(r for r in sample.throttle_reasons if r in ACTIVE_THROTTLE_REASONS)
    except Exception:
        return ()
    return ()


def wait_until_settled(
    gpu_index: int | None = None, timeout_s: float = SETTLE_TIMEOUT_S
) -> tuple[str, ...]:
    """Give a throttling card a chance to cool. Returns what was still wrong at the end.

    Measuring a hot card is not forbidden, but it has to be recorded: the discipline is
    that every measurement says what the machine was doing when it was taken.
    """
    reasons = read_throttle(gpu_index)
    if not reasons or timeout_s <= 0:
        return reasons
    deadline = time.monotonic() + timeout_s
    while reasons and time.monotonic() < deadline:
        time.sleep(SETTLE_POLL_S)
        reasons = read_throttle(gpu_index)
    return reasons


def _measured_side(objective: Objective) -> MeasurementKind:
    """Which half of the run the objective is scored on."""
    if objective is Objective.LATENCY:
        return MeasurementKind.PREFILL
    return MeasurementKind.DECODE
