"""Budgeting: what a model needs, what the card has, and where the difference lands."""

from __future__ import annotations

from .kv import CACHE_TYPES
from .kv import estimate as estimate_kv
from .overhead import (
    CALIBRATION,
    DEFAULT_UBATCH,
    RuntimeAllowance,
    runtime_allowance,
)
from .planner import fit, max_context, plan, seed_candidates
from .promptcache import DEFAULT_CACHE_RAM_MIB, PromptCacheEstimate
from .promptcache import estimate as estimate_prompt_cache
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
    "CALIBRATION",
    "DEFAULT_CACHE_RAM_MIB",
    "DEFAULT_FRAGMENTATION_PCT",
    "DEFAULT_RUNTIME_ALLOWANCE_BYTES",
    "DEFAULT_UBATCH",
    "GIB",
    "MIB",
    "Alternative",
    "BudgetPlan",
    "Candidate",
    "KvEstimate",
    "OffloadPlan",
    "PromptCacheEstimate",
    "RuntimeAllowance",
    "VramBudget",
    "assumed",
    "estimate_kv",
    "estimate_prompt_cache",
    "fit",
    "from_snapshot",
    "max_context",
    "plan",
    "runtime_allowance",
    "seed_candidates",
]
