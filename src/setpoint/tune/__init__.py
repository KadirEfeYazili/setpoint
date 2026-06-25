"""Autotuning: start from the budgeter's estimate, then measure until it stops improving."""

from __future__ import annotations

from .search import MAX_PASSES, MIN_IMPROVEMENT, is_improvement, neighbours, search
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
    "MAX_PASSES",
    "MIN_IMPROVEMENT",
    "Effort",
    "Measure",
    "SearchResult",
    "SearchSpace",
    "Stage",
    "Step",
    "Trial",
    "Verdict",
    "is_improvement",
    "neighbours",
    "search",
]
