"""Autotuning: start from the budgeter's estimate, then measure until it stops improving."""

from __future__ import annotations

from .measure import (
    DRIFT_NOTICE_BYTES,
    SETTLE_TIMEOUT_S,
    BackendMeasure,
    free_vram,
    read_throttle,
    run_spec,
    wait_until_settled,
)
from .search import (
    DEFAULT_MEASUREMENT_BUDGET,
    MAX_MOVES,
    MIN_IMPROVEMENT,
    is_improvement,
    neighbours,
    search,
)
from .types import (
    Effort,
    Measure,
    SearchResult,
    SearchSpace,
    Stage,
    Step,
    Trial,
    Verdict,
)

__all__ = [
    "DEFAULT_MEASUREMENT_BUDGET",
    "MAX_MOVES",
    "DRIFT_NOTICE_BYTES",
    "SETTLE_TIMEOUT_S",
    "BackendMeasure",
    "MIN_IMPROVEMENT",
    "Effort",
    "Measure",
    "SearchResult",
    "SearchSpace",
    "Stage",
    "Step",
    "Trial",
    "Verdict",
    "free_vram",
    "is_improvement",
    "neighbours",
    "read_throttle",
    "run_spec",
    "search",
    "wait_until_settled",
]
