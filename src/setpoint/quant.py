"""Comparing the quantizations of one model that this machine actually holds.

Three numbers decide the question, and they are not equally easy to get. Speed and VRAM
are measured the same way everything else in setpoint is measured. Quality is measured
as KL divergence against the highest-precision copy present, which is the deviation
between two quantizations of the same weights rather than an absolute score: it answers
"how far did this drift" instead of "how good is this", and only the first question has
a defensible answer here.

Quality needs a reference copy and a corpus. Without either, speed and VRAM are still
reported and quality is reported as not measured. That is the whole of the discipline:
a missing measurement is a missing measurement, not a zero.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .model import ModelError, ModelInfo, analyze, local_models

# The reference logits file holds one entry per vocabulary token per corpus token.
# Measured on a 262144-token vocabulary: 510 MiB for 2048 corpus tokens, so 255 KiB per
# token, or about one byte per vocabulary entry. A wikitext-sized corpus would need tens
# of gigabytes, which is why the cost is projected before anything is written.
LOGITS_BYTES_PER_VOCAB_TOKEN = 1

DEFAULT_CHUNKS = 4
DEFAULT_CONTEXT = 512
PERPLEXITY_BINARY = "llama-perplexity"


@dataclass(frozen=True)
class Variant:
    """One quantization of a model, as it sits on this machine."""

    reference: str
    info: ModelInfo

    @property
    def file_type(self) -> str | None:
        return self.info.file_type

    @property
    def bits_per_weight(self) -> float | None:
        return self.info.bits_per_weight

    @property
    def path(self) -> Path:
        return self.info.path


@dataclass(frozen=True)
class Quality:
    """How far one quantization drifted from the reference, on one corpus."""

    variant: str
    reference_variant: str
    corpus: str
    chunks: int
    ppl_ratio: float | None = None
    ppl_ratio_error: float | None = None
    median_kld: float | None = None
    same_top_pct: float | None = None
    rms_delta_p_pct: float | None = None
    detail: str | None = None

    @property
    def measured(self) -> bool:
        return self.ppl_ratio is not None or self.same_top_pct is not None


@dataclass(frozen=True)
class Comparison:
    """Every variant found, with whatever could be measured about each."""

    variants: tuple[Variant, ...] = field(default_factory=tuple)
    reference: Variant | None = None
    qualities: tuple[Quality, ...] = field(default_factory=tuple)
    detail: str | None = None

    def quality_of(self, reference: str) -> Quality | None:
        return next((q for q in self.qualities if q.variant == reference), None)


def same_model(one: ModelInfo, other: ModelInfo) -> bool:
    """Whether two files are the same weights at different precision.

    Architecture, depth, width and vocabulary all have to agree. Parameter count is
    deliberately not compared exactly: it is read from metadata that some converters
    round.
    """
    return (
        one.architecture == other.architecture
        and one.block_count == other.block_count
        and one.embedding_length == other.embedding_length
        and one.vocab_size == other.vocab_size
    )


def find_variants(target: ModelInfo) -> tuple[Variant, ...]:
    """Every local file that is the same model as `target`, most precise first."""
    found: list[Variant] = []
    seen: set[Path] = set()
    for resolved in local_models():
        if resolved.path in seen:
            continue
        seen.add(resolved.path)
        try:
            info = analyze(resolved.path)
        except ModelError:
            continue
        if same_model(target, info):
            found.append(Variant(reference=resolved.reference, info=info))
    return tuple(sorted(found, key=lambda v: v.bits_per_weight or 0, reverse=True))


def logits_bytes(vocab_size: int | None, chunks: int, context: int) -> int | None:
    """How large the reference logits file will be, before it is written."""
    if not vocab_size:
        return None
    return vocab_size * chunks * context * LOGITS_BYTES_PER_VOCAB_TOKEN


def measure_quality(
    variant: Variant,
    reference: Variant,
    corpus: str | Path,
    logits_path: str | Path,
    chunks: int = DEFAULT_CHUNKS,
    context: int = DEFAULT_CONTEXT,
    n_gpu_layers: int | None = None,
    devices: tuple[str, ...] = (),
    binary: str | Path = PERPLEXITY_BINARY,
    timeout_s: float = 1800.0,
) -> Quality:
    """Run the divergence pass for one variant against an already written logits file."""
    argv = [
        str(binary),
        "--model",
        str(variant.path),
        "-f",
        str(corpus),
        "--kl-divergence",
        "--kl-divergence-base",
        str(logits_path),
        "--ctx-size",
        str(context),
    ]
    if n_gpu_layers is not None:
        argv += ["-ngl", str(n_gpu_layers)]
    if devices:
        argv += ["--device", "/".join(devices)]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Quality(
            variant=variant.reference,
            reference_variant=reference.reference,
            corpus=str(corpus),
            chunks=chunks,
            detail=str(exc),
        )
    if done.returncode != 0:
        return Quality(
            variant=variant.reference,
            reference_variant=reference.reference,
            corpus=str(corpus),
            chunks=chunks,
            detail=_tail(done.stderr or done.stdout),
        )
    return parse_quality(
        (done.stdout or "") + (done.stderr or ""),
        variant=variant.reference,
        reference_variant=reference.reference,
        corpus=str(corpus),
        chunks=chunks,
    )


def write_reference_logits(
    reference: Variant,
    corpus: str | Path,
    logits_path: str | Path,
    chunks: int = DEFAULT_CHUNKS,
    context: int = DEFAULT_CONTEXT,
    n_gpu_layers: int | None = None,
    devices: tuple[str, ...] = (),
    binary: str | Path = PERPLEXITY_BINARY,
    timeout_s: float = 1800.0,
) -> str | None:
    """Save the reference logits, or say why it could not be done."""
    argv = [
        str(binary),
        "--model",
        str(reference.path),
        "-f",
        str(corpus),
        "--kl-divergence-base",
        str(logits_path),
        "--chunks",
        str(chunks),
        "--ctx-size",
        str(context),
    ]
    if n_gpu_layers is not None:
        argv += ["-ngl", str(n_gpu_layers)]
    if devices:
        argv += ["--device", "/".join(devices)]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    return None if done.returncode == 0 else _tail(done.stderr or done.stdout)


_NUMBER = r"(-?\d+\.?\d*)"
_PATTERNS = {
    # Whitespace in the report is column padding, not structure, and the separator
    # between a figure and its error is a plus-minus the locale codec may have mangled.
    "ppl_ratio": re.compile(r"Mean\s+PPL\(Q\)/PPL\(base\)\s*:\s*" + _NUMBER + r"\s*\D*" + _NUMBER),
    "median_kld": re.compile(r"Median\s+KLD:\s*" + _NUMBER),
    "same_top_pct": re.compile(r"Same top p:\s*" + _NUMBER),
    # The label carries a delta and the subprocess output is decoded with the locale
    # codec, which on this machine cannot represent it at all. Matching around the
    # character rather than on it keeps the figure readable whatever the codec did.
    "rms_delta_p_pct": re.compile(r"RMS\s+\S*p\s*:\s*" + _NUMBER),
}


def parse_quality(
    text: str, variant: str, reference_variant: str, corpus: str, chunks: int
) -> Quality:
    """Pull the headline figures out of the divergence report."""
    values: dict[str, float | None] = {}
    ratio_error = None
    for name, pattern in _PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            values[name] = None
            continue
        values[name] = float(match.group(1))
        if name == "ppl_ratio" and match.lastindex and match.lastindex >= 2:
            ratio_error = float(match.group(2))
    return Quality(
        variant=variant,
        reference_variant=reference_variant,
        corpus=corpus,
        chunks=chunks,
        ppl_ratio=values.get("ppl_ratio"),
        ppl_ratio_error=ratio_error,
        median_kld=values.get("median_kld"),
        same_top_pct=values.get("same_top_pct"),
        rms_delta_p_pct=values.get("rms_delta_p_pct"),
        detail=None if any(v is not None for v in values.values()) else "no figures were reported",
    )


def _tail(text: str | None, lines: int = 2) -> str:
    """The last few lines of a failure, which is where the reason usually is."""
    kept = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return " ".join(kept[-lines:]) if kept else "the run failed without output"
