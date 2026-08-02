"""Static model description.

Everything here comes from the GGUF header alone: no tensor data is read and no
inference is run. Attention shapes are stored per block because some architectures
vary the KV head count between layers, and the KV cache budget depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class ModelError(Exception):
    """setpoint cannot describe this model file."""


@dataclass(frozen=True)
class QuantShare:
    """How much of the file is stored in one quantization type."""

    quant: str
    tensor_count: int
    byte_count: int


@dataclass(frozen=True)
class Attention:
    head_count: tuple[int, ...]
    head_count_kv: tuple[int, ...]
    key_length: int
    value_length: int
    sliding_window: int | None = None
    sliding_window_pattern: int | None = None

    @property
    def uniform_head_count(self) -> int | None:
        return self.head_count[0] if len(set(self.head_count)) == 1 else None

    @property
    def uniform_head_count_kv(self) -> int | None:
        return self.head_count_kv[0] if len(set(self.head_count_kv)) == 1 else None

    @property
    def gqa_ratio(self) -> float | None:
        """Query heads per KV head. 1.0 means plain multi-head attention."""
        heads, kv_heads = self.uniform_head_count, self.uniform_head_count_kv
        if not heads or not kv_heads:
            return None
        return heads / kv_heads


@dataclass(frozen=True)
class Experts:
    count: int
    used: int
    shared: int = 0


@dataclass(frozen=True)
class Weights:
    """Tensor bytes grouped the way llama.cpp places them.

    llama.cpp offloads the last `n_gpu_layers` blocks; the output head moves to the GPU
    only once every block already fits.

    `tied_embedding` marks a model with no separate output projection, where the token
    embedding doubles as the output head. Measured on such a model, its bytes stay
    resident on the GPU at every offload split, so treating them as CPU-side understates
    the requirement by their full size.
    """

    block_bytes: tuple[int, ...]
    expert_bytes: tuple[int, ...]
    input_bytes: int
    output_bytes: int
    tied_embedding: bool = False

    @property
    def block_total_bytes(self) -> int:
        return sum(self.block_bytes)

    @property
    def expert_total_bytes(self) -> int:
        return sum(self.expert_bytes)

    @property
    def total_bytes(self) -> int:
        return self.block_total_bytes + self.input_bytes + self.output_bytes

    @property
    def resident_bytes(self) -> int:
        """Bytes that sit on the GPU whenever any block does.

        On a tied-embedding model that is the token embedding, because it is also the
        output projection and the matmul runs where the blocks run.
        """
        return self.input_bytes if self.tied_embedding else 0


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    file_bytes: int
    architecture: str
    name: str | None
    block_count: int
    embedding_length: int
    attention: Attention
    weights: Weights
    parameter_count: int
    quant_mix: tuple[QuantShare, ...]
    train_context: int | None = None
    vocab_size: int | None = None
    file_type: str | None = None
    experts: Experts | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_moe(self) -> bool:
        return self.experts is not None and self.experts.count > 1

    @property
    def dominant_quant(self) -> str | None:
        return self.quant_mix[0].quant if self.quant_mix else None

    @property
    def bits_per_weight(self) -> float | None:
        if not self.parameter_count:
            return None
        return 8 * self.weights.total_bytes / self.parameter_count

    @property
    def parameter_label(self) -> str:
        """Rounded parameter count, the way model cards write it."""
        count = self.parameter_count
        for limit, suffix in ((10**12, "T"), (10**9, "B"), (10**6, "M"), (10**3, "K")):
            if count >= limit:
                return f"{count / limit:.1f}{suffix}"
        return str(count)
