"""Decide what fits on the GPU, and say what a different request would buy.

llama.cpp offloads the last `n_gpu_layers` blocks, keeps the token embedding on the
CPU, and moves the output head to the GPU only once every block already fits. The KV
cache follows its block, so a block left on the CPU keeps its cache in system RAM.
"""

from __future__ import annotations

from ..model import ModelInfo
from . import kv as kv_module
from .types import Alternative, BudgetPlan, Candidate, KvEstimate, OffloadPlan, VramBudget

# Contexts worth suggesting when the request does not fit.
_CONTEXT_LADDER = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)

# Cache quantization to suggest as the first thing to try.
_FALLBACK_CACHE_TYPE = "q8_0"

# Above this share of *available* RAM, what stays on the CPU starts to cost paging on
# top of the slower compute. Judging against total RAM instead is how a plan gets drawn
# that a machine cannot actually hold.
#
# Half rather than most: a partial offload reads the whole model file, so the page cache
# it generates competes for the same RAM as the weights it leaves resident. A measurement
# run at 58% of 3.6 GiB free took the development machine down.
_RAM_PRESSURE_FRACTION = 0.5


def plan(
    model: ModelInfo,
    context: int,
    vram: VramBudget,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
    host_ram_bytes: int | None = None,
) -> BudgetPlan:
    """Build the full budget for one model, context and card.

    `host_ram_bytes` should be the RAM available now, not the RAM installed.
    """
    estimate = kv_module.estimate(model, context, cache_type_k, cache_type_v)
    offload = fit(model, estimate, vram.ceiling_bytes)

    notes = list(model.notes)
    if estimate is None:
        notes.append(
            "setpoint has no KV cache model for this architecture, so the plan covers "
            "weights only and understates what the GPU has to hold."
        )
    notes.extend(estimate.notes if estimate else ())
    if host_ram_bytes and offload.cpu_bytes > host_ram_bytes * _RAM_PRESSURE_FRACTION:
        notes.append(
            f"What stays on the CPU ({offload.cpu_bytes / (1024**3):.2f} GiB) is a large "
            f"share of the {host_ram_bytes / (1024**3):.2f} GiB of RAM free right now, and "
            f"reading the model will cache up to {model.file_bytes / (1024**3):.2f} GiB more. "
            "Close something before measuring."
        )

    return BudgetPlan(
        model=model,
        context=context,
        vram=vram,
        offload=offload,
        kv=estimate,
        alternatives=alternatives(model, vram, offload, context, cache_type_k, cache_type_v),
        candidates=seed_candidates(model, offload, cache_type_k, cache_type_v),
        host_ram_bytes=host_ram_bytes,
        notes=tuple(notes),
    )


def fit(model: ModelInfo, estimate: KvEstimate | None, ceiling_bytes: int) -> OffloadPlan:
    """Place as many trailing blocks on the GPU as the ceiling allows."""
    weights = model.weights
    kv_bytes = estimate.bytes_per_block if estimate else (0,) * model.block_count

    used = 0
    placed = 0
    next_block = None
    for index in reversed(range(model.block_count)):
        cost = weights.block_bytes[index] + kv_bytes[index]
        if used + cost > ceiling_bytes:
            next_block = cost
            break
        used += cost
        placed += 1

    output_on_gpu = placed == model.block_count and used + weights.output_bytes <= ceiling_bytes
    first_on_gpu = model.block_count - placed

    gpu_weights = sum(weights.block_bytes[first_on_gpu:])
    if output_on_gpu:
        gpu_weights += weights.output_bytes
    gpu_kv = sum(kv_bytes[first_on_gpu:])
    total = weights.total_bytes + sum(kv_bytes)

    return OffloadPlan(
        n_gpu_layers=placed + 1 if output_on_gpu else placed,
        block_count=model.block_count,
        output_on_gpu=output_on_gpu,
        gpu_bytes=gpu_weights + gpu_kv,
        cpu_bytes=total - gpu_weights - gpu_kv,
        weights_on_gpu_bytes=gpu_weights,
        kv_on_gpu_bytes=gpu_kv,
        next_block_bytes=next_block,
    )


def max_context(model: ModelInfo, ceiling_bytes: int, cache_type_k: str, cache_type_v: str) -> int:
    """Largest context whose weights and cache both fit entirely on the GPU."""
    weights = model.weights
    room = ceiling_bytes - weights.block_total_bytes - weights.output_bytes
    if room <= 0:
        return 0
    probe = kv_module.estimate(model, 1024, cache_type_k, cache_type_v)
    if probe is None or not probe.bytes_per_token:
        return 0
    limit = int(room / probe.bytes_per_token)
    if model.train_context:
        limit = min(limit, model.train_context)
    return max(0, limit)


def alternatives(
    model: ModelInfo,
    vram: VramBudget,
    offload: OffloadPlan,
    context: int,
    cache_type_k: str,
    cache_type_v: str,
) -> tuple[Alternative, ...]:
    """Concrete changes the user can make, each with the blocks it wins back."""
    ceiling = vram.ceiling_bytes
    baseline = kv_module.estimate(model, context, cache_type_k, cache_type_v)
    if baseline is None:
        return ()

    if offload.fits_fully:
        headroom = max_context(model, ceiling, cache_type_k, cache_type_v)
        if headroom <= context:
            return ()
        return (
            Alternative(
                change=f"-c {headroom}",
                effect="the longest context that still keeps every block on the GPU",
                n_gpu_layers=offload.n_gpu_layers,
                freed_bytes=0,
            ),
        )

    options: list[Alternative] = []
    if cache_type_k == "f16" and cache_type_v == "f16":
        quantized = kv_module.estimate(model, context, _FALLBACK_CACHE_TYPE, _FALLBACK_CACHE_TYPE)
        if quantized is not None:
            options.append(
                _alternative(
                    model,
                    quantized,
                    ceiling,
                    change=f"-ctk {_FALLBACK_CACHE_TYPE} -ctv {_FALLBACK_CACHE_TYPE}",
                    effect="a quantized KV cache, at the same context",
                    freed_bytes=baseline.total_bytes - quantized.total_bytes,
                )
            )

    smaller = [c for c in _CONTEXT_LADDER if c < context]
    if smaller:
        shrunk = kv_module.estimate(model, smaller[-1], cache_type_k, cache_type_v)
        if shrunk is not None:
            options.append(
                _alternative(
                    model,
                    shrunk,
                    ceiling,
                    change=f"-c {smaller[-1]}",
                    effect="a shorter context, at the same cache precision",
                    freed_bytes=baseline.total_bytes - shrunk.total_bytes,
                )
            )

    return tuple(o for o in options if o.n_gpu_layers > offload.n_gpu_layers)


def _alternative(
    model: ModelInfo,
    estimate: KvEstimate,
    ceiling_bytes: int,
    change: str,
    effect: str,
    freed_bytes: int,
) -> Alternative:
    placed = fit(model, estimate, ceiling_bytes)
    return Alternative(
        change=change,
        effect=effect,
        n_gpu_layers=placed.n_gpu_layers,
        freed_bytes=freed_bytes,
    )


def seed_candidates(
    model: ModelInfo,
    offload: OffloadPlan,
    cache_type_k: str,
    cache_type_v: str,
) -> tuple[Candidate, ...]:
    """Starting configurations for the autotuner.

    The static estimate is close but not exact, so the seeds bracket it: the tuner
    should measure just past the predicted edge as well as safely below it.
    """
    fitted = offload.n_gpu_layers
    origins = {
        fitted: "static estimate",
        max(0, fitted - 2): "two blocks of headroom",
        min(model.block_count + 1, fitted + 2): "past the predicted edge",
    }
    candidates = [
        Candidate(
            n_gpu_layers=layers,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            flash_attn=True,
            origin=origin,
        )
        for layers, origin in sorted(origins.items())
    ]
    if not offload.fits_fully and cache_type_k == "f16":
        candidates.append(
            Candidate(
                n_gpu_layers=fitted,
                cache_type_k=_FALLBACK_CACHE_TYPE,
                cache_type_v=_FALLBACK_CACHE_TYPE,
                flash_attn=True,
                origin="quantized cache, more blocks on the GPU",
            )
        )
    return tuple(candidates)
