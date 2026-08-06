"""What the inference process costs before it holds a single weight.

NVML reports free memory for the machine as it is now, and the process that will run the
model does not exist yet. Its buffers are absent from that reading, and on a 4 GB card
they are worth two layers.

These numbers were measured, not assumed. The dominant term turned out to be the output
logits buffer, which is sized by the vocabulary and the microbatch and does not care how
large the model is: doubling the embedding width left the per-token cost unchanged, and a
model with 1.73 times the vocabulary moved the per-token cost by 1.69. The readings and
the method are in the phase notes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import ModelInfo
from .types import MIB

# One float per vocabulary entry, for every token in the microbatch.
LOGITS_BYTES_PER_VOCAB = 4

# The compute graph itself, per embedding unit per microbatch token. Measured between
# 12 and 16 across two models; the upper end is used, because under-reserving spills.
GRAPH_BYTES_PER_EMBEDDING = 14

# Whatever does not scale: driver context and allocator bookkeeping.
#
# Fitted at 8 MiB, then re-measured with a per-run baseline across nine points on two
# architectures. The residual above the per-token terms sat between 10 and 19 MiB with no
# dependence on the microbatch, so the constant was too small and the estimate fell below
# the real peak. 24 MiB clears the worst point with room for the sampling noise. The cost
# is checked: at the offload boundary this machine actually measured, 89 MiB went unspent
# against a 109 MiB block, so the larger constant does not lose a layer.
BASE_BYTES = 24 * MIB

# llama.cpp's default microbatch, and what the budget assumes unless told otherwise.
DEFAULT_UBATCH = 512

# Used when the model does not say how large its vocabulary is.
FALLBACK_BYTES = 320 * MIB

# The hardware and backend the constants above were fitted on.
CALIBRATION = "Vulkan, GTX 1650, qwen2 and gemma3 (152k/262k vocab)"


@dataclass(frozen=True)
class RuntimeAllowance:
    """Room held back for the process that has not started yet."""

    total_bytes: int
    ubatch_size: int
    logits_bytes: int = 0
    graph_bytes: int = 0
    base_bytes: int = 0
    calibrated: bool = True
    detail: str | None = None


def runtime_allowance(model: ModelInfo, ubatch_size: int = DEFAULT_UBATCH) -> RuntimeAllowance:
    """Estimate the process overhead for one model at one microbatch size."""
    if not model.vocab_size:
        return RuntimeAllowance(
            total_bytes=FALLBACK_BYTES,
            ubatch_size=ubatch_size,
            calibrated=False,
            detail="the model does not report a vocabulary size, so a flat default is used",
        )

    logits = ubatch_size * model.vocab_size * LOGITS_BYTES_PER_VOCAB
    graph = ubatch_size * model.embedding_length * GRAPH_BYTES_PER_EMBEDDING
    return RuntimeAllowance(
        total_bytes=logits + graph + BASE_BYTES,
        ubatch_size=ubatch_size,
        logits_bytes=logits,
        graph_bytes=graph,
        base_bytes=BASE_BYTES,
        detail=f"calibrated on {CALIBRATION}",
    )
