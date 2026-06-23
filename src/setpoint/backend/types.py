"""Backend adapter contract.

A backend turns a configuration into a measurement. The three parts are kept separate
on purpose - build the command line, run it, parse what comes back - so that the first
and last are pure functions that can be tested without the backend installed, and so
that porting the adapter to another language stays a mechanical job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..measure import Statistic


class BackendError(Exception):
    """The backend could not be found, could not run, or returned something unreadable."""


class MeasurementKind(str, Enum):
    """Which half of a run a measurement describes.

    llama-bench reports prompt processing and token generation as separate results of
    the same invocation, and they answer different questions: prefill decides how long
    the first token takes, decode decides how fast the answer streams.
    """

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class RunSpec:
    """One configuration to measure.

    `n_depth` is what makes the measurement honest about context: it pre-fills the KV
    cache, so the run allocates and traverses the cache the target context implies
    instead of measuring an empty one.
    """

    model_path: Path
    n_gpu_layers: int | None = None
    n_cpu_moe: int | None = None
    n_depth: int = 0
    n_prompt: int = 512
    n_gen: int = 128
    batch_size: int | None = None
    ubatch_size: int | None = None
    threads: int | None = None
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    flash_attn: bool | None = None
    main_gpu: int | None = None
    tensor_overrides: tuple[str, ...] = ()
    repetitions: int = 5

    @property
    def context(self) -> int:
        """Context the run actually allocates, which is what has to fit in VRAM."""
        return self.n_depth + max(self.n_prompt, self.n_gen)


@dataclass(frozen=True)
class BackendBuild:
    """Which build produced a measurement. Part of a profile signature."""

    name: str
    commit: str | None = None
    number: int | None = None
    accelerators: str | None = None

    def __str__(self) -> str:
        parts = [self.name]
        if self.number is not None:
            parts.append(f"b{self.number}")
        if self.commit:
            parts.append(self.commit)
        if self.accelerators:
            parts.append(f"({self.accelerators})")
        return " ".join(parts)


@dataclass(frozen=True)
class BenchSample:
    """One measured test, with every repetition kept.

    The backend reports a mean and a standard deviation; setpoint keeps the raw
    repetitions instead and derives its own median and spread from them.
    """

    kind: MeasurementKind
    n_prompt: int
    n_gen: int
    n_depth: int
    throughput: Statistic
    duration_ns: Statistic
    reported_mean_ts: float | None = None

    @property
    def tokens_per_second(self) -> float | None:
        return self.throughput.median

    @property
    def reliable(self) -> bool:
        return self.throughput.reliable


@dataclass(frozen=True)
class BenchRun:
    """Everything one backend invocation produced."""

    spec: RunSpec
    build: BackendBuild
    samples: tuple[BenchSample, ...]
    gpu_info: str | None = None
    cpu_info: str | None = None
    command: tuple[str, ...] = ()
    duration_s: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def sample_of(self, kind: MeasurementKind) -> BenchSample | None:
        for sample in self.samples:
            if sample.kind is kind:
                return sample
        return None

    @property
    def decode_tokens_per_second(self) -> float | None:
        sample = self.sample_of(MeasurementKind.DECODE)
        return sample.tokens_per_second if sample else None

    @property
    def prefill_tokens_per_second(self) -> float | None:
        sample = self.sample_of(MeasurementKind.PREFILL)
        return sample.tokens_per_second if sample else None

    @property
    def reliable(self) -> bool:
        """True only when every measurement in the run held together."""
        return bool(self.samples) and all(s.reliable for s in self.samples)

    def to_dict(self) -> dict[str, object]:
        return {
            "build": {
                "name": self.build.name,
                "commit": self.build.commit,
                "number": self.build.number,
                "accelerators": self.build.accelerators,
            },
            "gpu_info": self.gpu_info,
            "cpu_info": self.cpu_info,
            "command": list(self.command),
            "duration_s": self.duration_s,
            "reliable": self.reliable,
            "samples": [
                {
                    "kind": s.kind.value,
                    "n_prompt": s.n_prompt,
                    "n_gen": s.n_gen,
                    "n_depth": s.n_depth,
                    "tokens_per_second": s.throughput.to_dict(),
                }
                for s in self.samples
            ],
            "notes": list(self.notes),
        }
