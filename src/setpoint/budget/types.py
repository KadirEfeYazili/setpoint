"""Budget data types.

The split between measured and estimated numbers is deliberate: `VramBudget.measured`
says whether the free-memory figure came from the driver or from an assumption, and
`runtime_allowance_bytes` is an allowance, not a computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import ModelInfo

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024


@dataclass(frozen=True)
class KvEstimate:
    """KV cache size for one context length and cache quantization."""

    context: int
    cache_type_k: str
    cache_type_v: str
    bytes_per_block: tuple[int, ...]
    upper_bound: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_bytes(self) -> int:
        return sum(self.bytes_per_block)

    @property
    def bytes_per_token(self) -> float:
        return self.total_bytes / self.context if self.context else 0.0


@dataclass(frozen=True)
class VramBudget:
    """What is actually available, after everything that is not ours is subtracted."""

    total_bytes: int
    free_bytes: int
    fragmentation_bytes: int
    reserve_bytes: int
    runtime_allowance_bytes: int
    measured: bool
    gpu_index: int | None = None
    gpu_name: str | None = None
    detail: str | None = None

    @property
    def in_use_bytes(self) -> int:
        return max(0, self.total_bytes - self.free_bytes)

    @property
    def ceiling_bytes(self) -> int:
        """The safe ceiling. Above this, allocations start spilling into system RAM."""
        claimed = self.fragmentation_bytes + self.reserve_bytes + self.runtime_allowance_bytes
        return max(0, self.free_bytes - claimed)


@dataclass(frozen=True)
class OffloadPlan:
    """How many blocks fit on the GPU, and what that leaves for the CPU."""

    n_gpu_layers: int
    block_count: int
    output_on_gpu: bool
    gpu_bytes: int
    cpu_bytes: int
    weights_on_gpu_bytes: int
    kv_on_gpu_bytes: int

    @property
    def blocks_on_gpu(self) -> int:
        return min(self.n_gpu_layers, self.block_count)

    @property
    def fits_fully(self) -> bool:
        return self.blocks_on_gpu == self.block_count

    @property
    def cpu_weight_fraction(self) -> float:
        total = self.cpu_bytes + self.weights_on_gpu_bytes + self.kv_on_gpu_bytes
        return self.cpu_bytes / total if total else 0.0


@dataclass(frozen=True)
class Alternative:
    """A change to the request, and what it buys."""

    change: str
    effect: str
    n_gpu_layers: int
    freed_bytes: int


@dataclass(frozen=True)
class Candidate:
    """A starting point for the autotuner, seeded by the static estimate."""

    n_gpu_layers: int
    cache_type_k: str
    cache_type_v: str
    flash_attn: bool
    origin: str


@dataclass(frozen=True)
class BudgetPlan:
    model: ModelInfo
    context: int
    vram: VramBudget
    offload: OffloadPlan
    kv: KvEstimate | None = None
    alternatives: tuple[Alternative, ...] = field(default_factory=tuple)
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)
    host_ram_bytes: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def required_bytes(self) -> int:
        """Everything the request needs, before any of it is pushed to the CPU."""
        kv_bytes = self.kv.total_bytes if self.kv else 0
        return self.model.weights.total_bytes + kv_bytes

    def to_dict(self) -> dict[str, object]:
        """The machine-readable form. This is the schema `--json` promises."""
        model, vram, offload = self.model, self.vram, self.offload
        payload: dict[str, object] = {
            "model": {
                "path": str(model.path),
                "architecture": model.architecture,
                "name": model.name,
                "file_type": model.file_type,
                "parameters": model.parameter_count,
                "block_count": model.block_count,
                "train_context": model.train_context,
                "vocab_size": model.vocab_size,
                "moe": model.is_moe,
                "weight_bytes": {
                    "blocks": model.weights.block_total_bytes,
                    "experts": model.weights.expert_total_bytes,
                    "input": model.weights.input_bytes,
                    "output": model.weights.output_bytes,
                    "total": model.weights.total_bytes,
                },
            },
            "vram": {
                "gpu_index": vram.gpu_index,
                "gpu_name": vram.gpu_name,
                "measured": vram.measured,
                "total_bytes": vram.total_bytes,
                "free_bytes": vram.free_bytes,
                "fragmentation_bytes": vram.fragmentation_bytes,
                "reserve_bytes": vram.reserve_bytes,
                "runtime_allowance_bytes": vram.runtime_allowance_bytes,
                "ceiling_bytes": vram.ceiling_bytes,
                "detail": vram.detail,
            },
            "request": {"context": self.context, "required_bytes": self.required_bytes},
            "plan": {
                "n_gpu_layers": offload.n_gpu_layers,
                "block_count": offload.block_count,
                "output_on_gpu": offload.output_on_gpu,
                "gpu_bytes": offload.gpu_bytes,
                "cpu_bytes": offload.cpu_bytes,
                "weights_on_gpu_bytes": offload.weights_on_gpu_bytes,
                "kv_on_gpu_bytes": offload.kv_on_gpu_bytes,
                "fits_fully": offload.fits_fully,
            },
            "alternatives": [
                {
                    "change": a.change,
                    "effect": a.effect,
                    "n_gpu_layers": a.n_gpu_layers,
                    "freed_bytes": a.freed_bytes,
                }
                for a in self.alternatives
            ],
            "candidates": [
                {
                    "n_gpu_layers": c.n_gpu_layers,
                    "cache_type_k": c.cache_type_k,
                    "cache_type_v": c.cache_type_v,
                    "flash_attn": c.flash_attn,
                    "origin": c.origin,
                }
                for c in self.candidates
            ],
            "host_ram_bytes": self.host_ram_bytes,
            "notes": list(self.notes),
        }
        payload["kv_cache"] = (
            None
            if self.kv is None
            else {
                "context": self.kv.context,
                "cache_type_k": self.kv.cache_type_k,
                "cache_type_v": self.kv.cache_type_v,
                "total_bytes": self.kv.total_bytes,
                "bytes_per_token": self.kv.bytes_per_token,
                "upper_bound": self.kv.upper_bound,
            }
        )
        return payload
