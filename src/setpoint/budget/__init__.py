"""Budgeting: what a model needs, what the card has, and where the difference lands."""

from __future__ import annotations

from .kv import CACHE_TYPES
from .kv import estimate as estimate_kv
from .planner import fit, max_context, plan, seed_candidates
from .types import (
    GIB,
    MIB,
    Alternative,
    BudgetPlan,
    Candidate,
    KvEstimate,
    OffloadPlan,
    VramBudget,
)
from .vram import (
    DEFAULT_FRAGMENTATION_PCT,
    DEFAULT_RUNTIME_ALLOWANCE_BYTES,
    assumed,
    from_snapshot,
)

__all__ = [
    "CACHE_TYPES",
    "DEFAULT_FRAGMENTATION_PCT",
    "DEFAULT_RUNTIME_ALLOWANCE_BYTES",
    "GIB",
    "MIB",
    "Alternative",
    "BudgetPlan",
    "Candidate",
    "KvEstimate",
    "OffloadPlan",
    "VramBudget",
    "assumed",
    "estimate_kv",
    "fit",
    "from_snapshot",
    "max_context",
    "plan",
    "seed_candidates",
]
