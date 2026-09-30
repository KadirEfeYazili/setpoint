"""Host RAM claimed by the server's prompt cache.

The VRAM budget has never counted this. `llama-server` parks the context of a displaced
conversation in host RAM so that returning to it costs almost nothing, and the ceiling
on that store defaults to 8 GiB. On a machine where a partial offload already keeps
weights in RAM, the two claims are on the same memory.
"""

from __future__ import annotations

from .types import KvEstimate, PromptCacheEstimate

# llama-server's own default for --cache-ram, in MiB.
DEFAULT_CACHE_RAM_MIB = 8192

CALIBRATION = (
    "measured within 1% of the KV formula on a dense model; a sliding-window model "
    "cost more, so no window discount is taken here"
)


def estimate(
    kv: KvEstimate | None,
    cache_ram_mib: int = DEFAULT_CACHE_RAM_MIB,
) -> PromptCacheEstimate | None:
    """Size the prompt cache from the KV estimate the same context already produced.

    A parked conversation holds its whole context, so one conversation costs what the
    KV cache costs. The sliding-window discount is deliberately not applied: on the one
    such model measured here the host cost exceeded the undiscounted formula.
    """
    if kv is None or cache_ram_mib == 0:
        return None
    per_conversation = kv.total_bytes
    ceiling = kv.total_bytes * 1_000_000 if cache_ram_mib < 0 else cache_ram_mib * 1024 * 1024
    notes = []
    if cache_ram_mib < 0:
        notes.append("--cache-ram is unlimited, so the only ceiling is the RAM itself.")
    if kv.upper_bound:
        notes.append(
            "The KV figure is an upper bound in VRAM because of the sliding window, but "
            "the host copy measured larger than it, so this is not reduced."
        )
    return PromptCacheEstimate(
        context=kv.context,
        ceiling_bytes=ceiling,
        bytes_per_conversation=per_conversation,
        notes=tuple(notes),
    )


def pressure_note(
    cache: PromptCacheEstimate | None,
    cpu_bytes: int,
    host_ram_bytes: int | None,
) -> str | None:
    """Warn only where the two claims on host RAM actually collide.

    The ceiling on its own is a permission, not a reservation, and reporting it as a
    cost would fire on every machine. What is worth saying is that a partial offload
    and the prompt cache are spending the same RAM.
    """
    if cache is None or not host_ram_bytes or cpu_bytes <= 0:
        return None
    room = host_ram_bytes - cpu_bytes
    if room <= 0 or cache.bytes_per_conversation <= 0:
        return None
    fits = room // cache.bytes_per_conversation
    if fits > 8:
        return None
    gib = cache.bytes_per_conversation / 1024**3
    return (
        f"This model keeps {cpu_bytes / 1024**3:.2f} GiB in RAM, and the prompt cache "
        f"parks {gib:.2f} GiB per displaced conversation in the same RAM. "
        f"{fits} conversation(s) fit in what is left; lower --cache-ram to bound it."
    )
