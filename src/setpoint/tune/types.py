"""Autotuner data types.

The search never measures anything itself. It calls a `Measure` function and works with
what comes back, which keeps the algorithm testable against a known landscape and keeps
the choice of objective - speed, latency, efficiency, headroom - outside the search.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from ..backend import BenchRun
from ..hardware import GpuWatch
from ..profile import Config


class Stage(str, Enum):
    """Which part of the search a step belongs to."""

    BASELINE = "baseline"
    SCREEN = "screen"
    DESCEND = "descend"
    CONFIRM = "confirm"


class Verdict(str, Enum):
    KEPT = "kept"
    DROPPED = "dropped"
    IMPROVED = "improved"
    NO_BETTER = "no better"
    FAILED = "failed"


@dataclass(frozen=True)
class Effort:
    """How much measurement one round spends per configuration.

    Screening rounds run short and cheap; they rank configurations and nothing more.
    The winner is always re-measured at full effort before it is reported, so no
    number that reaches a profile came from a screening run.
    """

    repetitions: int
    n_depth: int
    label: str = ""


@dataclass(frozen=True)
class SearchSpace:
    """Bounds for the moves coordinate descent is allowed to make.

    Context is deliberately absent. It is the user's constraint, not something to
    optimise: a different context is a different question and a different profile.
    """

    max_gpu_layers: int
    max_threads: int
    moe: bool = False
    batch_sizes: tuple[int, ...] = (512, 1024, 2048)
    ubatch_sizes: tuple[int, ...] = (128, 256, 512)
    tune_flash_attn: bool = True


@dataclass(frozen=True)
class Trial:
    """One configuration, measured. `score` is `None` when the run did not produce one."""

    config: Config
    score: float | None = None
    spread: float | None = None
    reliable: bool = False
    detail: str = ""
    run: BenchRun | None = None
    watch: GpuWatch | None = None

    @property
    def usable(self) -> bool:
        return self.score is not None

    @property
    def peak_vram_mib(self) -> int | None:
        return self.watch.peak_vram_mib if self.watch else None

    @property
    def average_power_w(self) -> float | None:
        return self.watch.average_power_w if self.watch else None


@dataclass(frozen=True)
class Step:
    """One decision the search made, in the order it made it."""

    stage: Stage
    config: Config
    score: float | None
    verdict: Verdict
    note: str = ""


@dataclass(frozen=True)
class SearchResult:
    best: Trial | None
    baseline: Trial | None
    steps: tuple[Step, ...] = field(default_factory=tuple)
    measurements: int = 0
    interrupted: bool = False
    reason: str = ""

    @property
    def speedup(self) -> float | None:
        """How much the winner beat the baseline by. Below 1.0 is reported, not hidden."""
        if not self.best or not self.baseline:
            return None
        if not self.best.score or not self.baseline.score:
            return None
        return self.best.score / self.baseline.score


# Measures one configuration at one effort level. Supplied by the caller so that the
# search can be driven by a real backend or by a test landscape.
Measure = Callable[[Config, Effort], Trial]
