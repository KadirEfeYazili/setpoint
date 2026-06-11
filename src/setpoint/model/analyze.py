"""Turn a GGUF header into a `ModelInfo`.

Metadata keys are architecture-prefixed and optional, so every lookup has a
documented fallback. A value we cannot derive stays `None` rather than being guessed.
"""

from __future__ import annotations

import collections
from collections.abc import Callable
from pathlib import Path

from .reader import ArraySummary, GgufHeader, read_header
from .types import Attention, Experts, ModelError, ModelInfo, QuantShare, Weights

# Architectures whose KV cache is a compressed latent, not a per-head key/value pair.
# The standard formula does not describe them, so setpoint declines to estimate.
LATENT_ATTENTION_ARCHITECTURES = frozenset({"deepseek2", "kimi-k2"})

_FILE_TYPES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4_MOE",
    39: "NVFP4",
    40: "Q1_0",
}


def analyze(path: str | Path) -> ModelInfo:
    """Describe a GGUF model file. Raises `ModelError` if the header cannot be read."""
    return describe(read_header(path))


def describe(header: GgufHeader) -> ModelInfo:
    meta = header.metadata
    architecture = _text(meta, "general.architecture")
    if not architecture:
        raise ModelError(f"{header.path} has no general.architecture key")

    def key(suffix: str) -> str:
        return f"{architecture}.{suffix}"

    block_count = _int(meta, key("block_count"))
    if not block_count:
        raise ModelError(f"{header.path} has no {key('block_count')} key")
    embedding_length = _int(meta, key("embedding_length")) or 0

    notes: list[str] = []
    attention = _attention(meta, key, block_count, embedding_length, notes)
    weights = _weights(header, block_count, notes)

    return ModelInfo(
        path=header.path,
        file_bytes=header.file_bytes,
        architecture=architecture,
        name=_text(meta, "general.name"),
        block_count=block_count,
        embedding_length=embedding_length,
        attention=attention,
        weights=weights,
        parameter_count=sum(t.element_count for t in header.tensors),
        quant_mix=_quant_mix(header),
        train_context=_int(meta, key("context_length")),
        vocab_size=_vocab_size(meta, key("vocab_size")),
        file_type=_file_type(meta),
        experts=_experts(meta, key),
        notes=tuple(notes),
    )


def _attention(
    meta: dict[str, object],
    key: Callable[[str], str],
    block_count: int,
    embedding_length: int,
    notes: list[str],
) -> Attention:
    head_count = _per_block(meta, key("attention.head_count"), block_count) or (1,) * block_count
    head_count_kv = _per_block(meta, key("attention.head_count_kv"), block_count) or head_count

    head_dim = embedding_length // head_count[0] if head_count[0] else 0
    key_length = _int(meta, key("attention.key_length")) or head_dim
    value_length = _int(meta, key("attention.value_length")) or head_dim

    if key("attention.key_length_mla") in meta:
        notes.append(
            "This architecture compresses the KV cache into a latent vector; "
            "the standard per-head formula does not apply."
        )

    return Attention(
        head_count=head_count,
        head_count_kv=head_count_kv,
        key_length=key_length,
        value_length=value_length,
        sliding_window=_int(meta, key("attention.sliding_window")),
        sliding_window_pattern=_int(meta, key("attention.sliding_window_pattern")),
    )


def _weights(header: GgufHeader, block_count: int, notes: list[str]) -> Weights:
    block_bytes = [0] * block_count
    expert_bytes = [0] * block_count
    input_bytes = 0
    output_bytes = 0
    stray = 0

    for tensor in header.tensors:
        index = _block_index(tensor.name)
        if index is None:
            if tensor.name.startswith("token_embd"):
                input_bytes += tensor.byte_count
            else:
                output_bytes += tensor.byte_count
            continue
        if index >= block_count:
            stray += tensor.byte_count
            continue
        block_bytes[index] += tensor.byte_count
        if "_exps" in tensor.name:
            expert_bytes[index] += tensor.byte_count

    if stray:
        notes.append(
            f"{stray} bytes of tensors sit in blocks beyond the declared block count "
            "and are counted as always resident."
        )
        output_bytes += stray

    return Weights(
        block_bytes=tuple(block_bytes),
        expert_bytes=tuple(expert_bytes),
        input_bytes=input_bytes,
        output_bytes=output_bytes,
    )


def _block_index(name: str) -> int | None:
    if not name.startswith("blk."):
        return None
    rest = name[4:].split(".", 1)[0]
    return int(rest) if rest.isdigit() else None


def _quant_mix(header: GgufHeader) -> tuple[QuantShare, ...]:
    counts: collections.Counter[str] = collections.Counter()
    sizes: collections.Counter[str] = collections.Counter()
    for tensor in header.tensors:
        counts[tensor.quant] += 1
        sizes[tensor.quant] += tensor.byte_count
    shares = [QuantShare(q, counts[q], sizes[q]) for q in sizes]
    shares.sort(key=lambda s: -s.byte_count)
    return tuple(shares)


def _experts(meta: dict[str, object], key: Callable[[str], str]) -> Experts | None:
    count = _int(meta, key("expert_count"))
    if not count:
        return None
    return Experts(
        count=count,
        used=_int(meta, key("expert_used_count")) or 0,
        shared=_int(meta, key("expert_shared_count")) or 0,
    )


def _file_type(meta: dict[str, object]) -> str | None:
    value = meta.get("general.file_type")
    return _FILE_TYPES.get(value) if isinstance(value, int) else None


def _vocab_size(meta: dict[str, object], fallback_key: str) -> int | None:
    tokens = meta.get("tokenizer.ggml.tokens")
    if isinstance(tokens, ArraySummary):
        return tokens.count
    if isinstance(tokens, list):
        return len(tokens)
    return _int(meta, fallback_key)


def _text(meta: dict[str, object], key: str) -> str | None:
    value = meta.get(key)
    return value if isinstance(value, str) else None


def _int(meta: dict[str, object], key: str) -> int | None:
    value = meta.get(key)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _per_block(meta: dict[str, object], key: str, block_count: int) -> tuple[int, ...] | None:
    """Read a value that is either uniform or listed per block."""
    value = meta.get(key)
    if isinstance(value, list) and value:
        listed = [int(v) for v in value[:block_count]]
        return tuple(listed + [listed[-1]] * (block_count - len(listed)))
    scalar = _int(meta, key)
    return (scalar,) * block_count if scalar else None
