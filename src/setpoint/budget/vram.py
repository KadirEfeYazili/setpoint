"""Turn a hardware snapshot into a spendable VRAM budget.

The starting point is free memory, not total: the driver, the desktop compositor and
any other process already hold their share, and NVML reports what is left. From there
setpoint subtracts three things it cannot see in that number.
"""

from __future__ import annotations

from ..hardware import GpuStatic, HardwareSnapshot
from .types import MIB, VramBudget

# Allocators do not pack perfectly, and the last block that "just fits" is the one that
# fails. Held back as a fraction of free memory.
#
# Measured, not guessed. Two models were probed for the largest -ngl that actually runs
# and compared against what each margin predicts:
#
#            8B (real 28)        7B (real 25)
#   3%       30, over by 2       26, over by 1
#   5%       29, over by 1       25, exact
#   8%       28, exact           24, one block spare
#
# 8% is the smallest margin safe on both. Under-reserving means the model does not
# start; over-reserving costs a block. The refusal that set the boundary was an
# allocation of 1.7 MiB, so this is allocator exhaustion rather than a missing term.
DEFAULT_FRAGMENTATION_PCT = 8.0

# The inference process has not started yet, so its CUDA context and compute buffers
# are absent from the free-memory reading. This is an allowance, not a measurement;
# `setpoint tune` replaces it with observed peak VRAM.
DEFAULT_RUNTIME_ALLOWANCE_BYTES = 192 * MIB


def from_snapshot(
    snapshot: HardwareSnapshot,
    gpu_index: int | None = None,
    reserve_bytes: int = 0,
    fragmentation_pct: float = DEFAULT_FRAGMENTATION_PCT,
    runtime_allowance_bytes: int = DEFAULT_RUNTIME_ALLOWANCE_BYTES,
) -> VramBudget | None:
    """Build a budget for one GPU, or `None` if the snapshot holds no usable GPU."""
    gpu = _select(snapshot, gpu_index)
    if gpu is None:
        return None

    sample = snapshot.sample_for(gpu.index)
    free = sample.vram_free_bytes if sample else None
    detail = None
    if free is None:
        free = gpu.vram_total_bytes
        detail = "free memory could not be read; assuming the card is empty"

    return VramBudget(
        total_bytes=gpu.vram_total_bytes,
        free_bytes=free,
        fragmentation_bytes=int(free * fragmentation_pct / 100),
        reserve_bytes=reserve_bytes,
        runtime_allowance_bytes=runtime_allowance_bytes,
        measured=detail is None,
        gpu_index=gpu.index,
        gpu_name=gpu.name,
        detail=detail,
    )


def assumed(
    total_bytes: int,
    reserve_bytes: int = 0,
    fragmentation_pct: float = DEFAULT_FRAGMENTATION_PCT,
    runtime_allowance_bytes: int = DEFAULT_RUNTIME_ALLOWANCE_BYTES,
) -> VramBudget:
    """Budget for a card setpoint cannot see, for planning against other hardware."""
    return VramBudget(
        total_bytes=total_bytes,
        free_bytes=total_bytes,
        fragmentation_bytes=int(total_bytes * fragmentation_pct / 100),
        reserve_bytes=reserve_bytes,
        runtime_allowance_bytes=runtime_allowance_bytes,
        measured=False,
        detail="hypothetical card; no driver reading was taken",
    )


def _select(snapshot: HardwareSnapshot, gpu_index: int | None) -> GpuStatic | None:
    if not snapshot.gpus:
        return None
    if gpu_index is None:
        return max(snapshot.gpus, key=lambda g: g.vram_total_bytes)
    for gpu in snapshot.gpus:
        if gpu.index == gpu_index:
            return gpu
    return None
