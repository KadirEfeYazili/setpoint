"""Profile data types.

A profile is a claim with evidence behind it: this configuration, on this hardware, for
this target, and here is the measurement that justifies it. The schema these types
serialise is specified in `spec-profile-v1.md`, which is the authority; this module
implements it and does not extend it.

A stored measurement keeps the summary rather than the raw repetitions, so `Summary`
rather than `Statistic` is what crosses the file boundary and what round-trips exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..measure import MAX_RELIABLE_SPREAD, MIN_RELIABLE_RUNS, Statistic

SCHEMA = "setpoint/v1"

# Digest kinds, and what each one covers. See spec section 4.1.
DIGEST_HEADER = "header"
DIGEST_CONTENT = "content"


class ProfileError(Exception):
    """A profile could not be read, or may not be written."""


class Objective(str, Enum):
    """What the tuner was asked to maximise."""

    SPEED = "speed"
    LATENCY = "latency"
    EFFICIENCY = "efficiency"
    HEADROOM = "headroom"


@dataclass(frozen=True)
class Summary:
    """One measured quantity as a profile stores it. Mean and stddev are not kept."""

    median: float
    iqr: float
    spread: float

    @classmethod
    def of(cls, statistic: Statistic) -> Summary | None:
        median, iqr, spread = statistic.median, statistic.iqr, statistic.spread
        if median is None or iqr is None or spread is None:
            return None
        return cls(median=median, iqr=iqr, spread=spread)


@dataclass(frozen=True)
class Signature:
    """The world a profile is valid in. Every field must match, or it does not apply."""

    model_digest: str
    model_digest_kind: str
    model_size_bytes: int
    gpu: str
    vram_total_mb: int
    driver: str
    backend: str
    platform: str

    def matches(self, other: Signature) -> bool:
        return self == other


@dataclass(frozen=True)
class Target:
    context: int
    optimize: Objective = Objective.SPEED


@dataclass(frozen=True)
class Config:
    """Backend-neutral settings. `None` means leave the backend default alone."""

    n_gpu_layers: int | None = None
    n_cpu_moe: int | None = None
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    flash_attn: bool | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    threads: int | None = None
    tensor_overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class Measurement:
    """The evidence. Without it there is no profile."""

    runs: int
    decode_tok_s: Summary
    measured_at: str
    prefill_tok_s: Summary | None = None
    peak_vram_mb: int | None = None
    avg_watt: float | None = None

    @classmethod
    def from_statistics(
        cls,
        decode: Statistic,
        measured_at: str,
        prefill: Statistic | None = None,
        peak_vram_mb: int | None = None,
        avg_watt: float | None = None,
    ) -> Measurement:
        summary = Summary.of(decode)
        if summary is None:
            raise ProfileError("the decode measurement produced no usable median")
        return cls(
            runs=decode.runs,
            decode_tok_s=summary,
            measured_at=measured_at,
            prefill_tok_s=Summary.of(prefill) if prefill is not None else None,
            peak_vram_mb=peak_vram_mb,
            avg_watt=avg_watt,
        )

    @property
    def summaries(self) -> tuple[Summary, ...]:
        parts = [self.decode_tok_s]
        if self.prefill_tok_s is not None:
            parts.append(self.prefill_tok_s)
        return tuple(parts)

    @property
    def reliable(self) -> bool:
        """Derived, never read from the file: a stored flag can be edited, this cannot."""
        return self.runs >= MIN_RELIABLE_RUNS and all(
            s.spread <= MAX_RELIABLE_SPREAD for s in self.summaries
        )

    def why_unreliable(self) -> str | None:
        if self.reliable:
            return None
        if self.runs < MIN_RELIABLE_RUNS:
            return f"only {self.runs} runs were counted, and {MIN_RELIABLE_RUNS} are required"
        worst = max(s.spread for s in self.summaries)
        return (
            f"the spread was {worst:.1%}, above the {MAX_RELIABLE_SPREAD:.0%} a profile may carry"
        )


@dataclass(frozen=True)
class Baseline:
    """What the configuration was compared against, and by how much it won."""

    label: str
    config: Config
    decode_tok_s: float
    speedup: float

    @property
    def improved(self) -> bool:
        return self.speedup > 1.0


@dataclass(frozen=True)
class Profile:
    signature: Signature
    target: Target
    config: Config
    measurement: Measurement
    baseline: Baseline
    created: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def writable(self) -> bool:
        """An unreliable measurement is a refusal, not a warning. See spec section 7.1."""
        return self.measurement.reliable

    def why_not_writable(self) -> str | None:
        return self.measurement.why_unreliable()
