"""KV cache sizing.

The cache holds one key and one value vector per KV head per token, so its size grows
linearly with context and is unaffected by weight quantization. On a small card it is
usually the term that decides how many blocks reach the GPU.
"""

from __future__ import annotations

from ..model import LATENT_ATTENTION_ARCHITECTURES, ModelInfo
from ..model.reader import QUANT_TYPES
from .types import KvEstimate

# llama.cpp -ctk/-ctv values, mapped to the GGML type that backs them.
CACHE_TYPES: dict[str, str] = {
    "f32": "F32",
    "f16": "F16",
    "bf16": "BF16",
    "q8_0": "Q8_0",
    "q5_1": "Q5_1",
    "q5_0": "Q5_0",
    "q4_1": "Q4_1",
    "q4_0": "Q4_0",
    "iq4_nl": "IQ4_NL",
}

_BY_NAME = {name: (block, block_bytes) for name, block, block_bytes in QUANT_TYPES.values()}

# Quantized cache entries are only readable by the flash-attention kernel.
QUANTIZED_CACHE_TYPES = frozenset(CACHE_TYPES) - {"f32", "f16", "bf16"}


def bytes_per_element(cache_type: str) -> float:
    """Storage cost of one cache element, in bytes."""
    quant = CACHE_TYPES.get(cache_type.lower())
    if quant is None:
        raise ValueError(f"unknown cache type {cache_type!r}")
    block, block_bytes = _BY_NAME[quant]
    return block_bytes / block


def estimate(
    model: ModelInfo,
    context: int,
    cache_type_k: str = "f16",
    cache_type_v: str = "f16",
) -> KvEstimate | None:
    """Size the KV cache, or return `None` when the formula does not describe the model."""
    if model.architecture in LATENT_ATTENTION_ARCHITECTURES:
        return None

    attention = model.attention
    per_token_k = attention.key_length * bytes_per_element(cache_type_k)
    per_token_v = attention.value_length * bytes_per_element(cache_type_v)
    per_block = tuple(
        int(context * kv_heads * (per_token_k + per_token_v))
        for kv_heads in attention.head_count_kv
    )

    notes: list[str] = []
    window = attention.sliding_window
    upper_bound = bool(window and context > window)
    if upper_bound:
        notes.append(
            f"Sliding-window blocks hold at most {window} tokens, so the real cache is "
            "smaller than this figure."
        )
    if {cache_type_k, cache_type_v} & QUANTIZED_CACHE_TYPES:
        notes.append("A quantized KV cache requires flash attention to be enabled.")

    return KvEstimate(
        context=context,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        bytes_per_block=per_block,
        upper_bound=upper_bound,
        notes=tuple(notes),
    )
