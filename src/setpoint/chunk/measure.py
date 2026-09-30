"""Running each chunking strategy over a corpus and recording what it cost.

The published comparisons of these strategies report throughput and retrieval quality.
Neither says what peak VRAM a strategy reaches, and on a small card that is the figure
that decides whether the generator still fits beside it. This measures it.

Sizes are reported in the tokens of the model that will read the chunks, because that
is the budget being spent. Getting them there without distorting the timings needs
care: see `calibrate`.
"""

from __future__ import annotations

import random
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..hardware import GpuWatcher

# Below this a corpus is too small for the timings to mean anything: the fixed cost of
# starting a strategy dominates and the comparison measures startup.
MIN_CORPUS_BYTES = 64 * 1024

# Enough chunks for a stable median without a round trip per chunk.
SIZE_SAMPLE = 150

# Enough text to pin the ratio; it varies by a few percent over a corpus, not by a lot.
CALIBRATION_BYTES = 32 * 1024

TEXT_SUFFIXES = (".txt", ".md", ".rst", ".text")


@dataclass(frozen=True)
class Document:
    """One piece of the corpus, already read."""

    path: Path
    text: str

    @property
    def bytes(self) -> int:
        return len(self.text.encode("utf-8", "replace"))


@dataclass(frozen=True)
class Corpus:
    """What is being chunked, and whether it is worth measuring on."""

    documents: tuple[Document, ...] = field(default_factory=tuple)
    detail: str | None = None

    @property
    def total_bytes(self) -> int:
        return sum(document.bytes for document in self.documents)

    @property
    def reliable(self) -> bool:
        return self.total_bytes >= MIN_CORPUS_BYTES

    @property
    def label(self) -> str:
        return f"{len(self.documents)} documents, {self.total_bytes / 1024**2:.1f} MB"

    def head(self, nbytes: int) -> str:
        """The first stretch of text, for calibration."""
        out: list[str] = []
        taken = 0
        for document in self.documents:
            piece = document.text[: max(0, nbytes - taken)]
            out.append(piece)
            taken += len(piece)
            if taken >= nbytes:
                break
        return "".join(out)


@dataclass(frozen=True)
class Result:
    """What one strategy cost on this corpus."""

    strategy: str
    seconds: float
    chunks: int
    tokens_median: float
    tokens_p90: float
    peak_vram_mib: int | None
    sampled: int = 0
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.detail is None

    def throughput(self, corpus_bytes: int) -> float:
        return (corpus_bytes / 1024**2) / self.seconds if self.seconds else 0.0

    def context_per_query(self, k: int) -> float:
        """What retrieving k of these chunks would put into the prompt."""
        return self.tokens_median * k


def read_corpus(root: Path, limit_bytes: int | None = None) -> Corpus:
    """Read a directory of text, or a single file, in a stable order."""
    root = Path(root)
    if not root.exists():
        return Corpus(detail=f"{root} does not exist")
    paths = (
        [root]
        if root.is_file()
        else sorted(p for p in root.rglob("*") if p.suffix.lower() in TEXT_SUFFIXES)
    )
    if not paths:
        return Corpus(detail=f"no text files under {root}")

    documents: list[Document] = []
    total = 0
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            continue
        documents.append(Document(path=path, text=text))
        total += len(text.encode("utf-8", "replace"))
        if limit_bytes and total >= limit_bytes:
            break
    if not documents:
        return Corpus(detail=f"nothing readable under {root}")
    return Corpus(documents=tuple(documents))


def calibrate(corpus: Corpus, count_tokens: Callable[[str], int]) -> float:
    """Characters per token for this corpus, in the generator's own vocabulary.

    A chunker that asks a remote tokenizer for every candidate split pays a round trip
    each time: measured here, that turned a 1.4 s strategy into 47 s and the table would
    have blamed the strategy. So the chunkers are driven by character counts and the
    budget is converted once, using the real tokenizer on a sample of this corpus.
    """
    head = corpus.head(CALIBRATION_BYTES)
    tokens = count_tokens(head) if head else 0
    return len(head) / tokens if tokens else 4.0


def run(
    strategy,
    chunker,
    corpus: Corpus,
    count_tokens: Callable[[str], int],
    sample_size: int = SIZE_SAMPLE,
    seed: int = 0,
) -> Result:
    """Chunk the whole corpus once, timing it and watching the card.

    The card is watched rather than assumed: a strategy that runs no model should show
    no rise, and that is a claim worth checking rather than stating. Token counting
    happens after the clock stops, on a sample, for the same reason as `calibrate`.
    """
    watcher = GpuWatcher()
    started = time.monotonic()
    chunks: list[str] = []
    try:
        with watcher:
            for document in corpus.documents:
                chunks.extend(_texts(chunker(document.text)))
    except Exception as exc:  # noqa: BLE001
        return Result(
            strategy=strategy.name,
            seconds=time.monotonic() - started,
            chunks=0,
            tokens_median=0.0,
            tokens_p90=0.0,
            peak_vram_mib=None,
            detail=str(exc),
        )
    seconds = time.monotonic() - started

    if len(chunks) <= sample_size:
        picked = chunks
    else:
        picked = random.Random(seed).sample(chunks, sample_size)
    sizes = sorted(count_tokens(text) for text in picked) or [0]
    return Result(
        strategy=strategy.name,
        seconds=seconds,
        chunks=len(chunks),
        tokens_median=statistics.median(sizes),
        tokens_p90=sizes[min(len(sizes) - 1, int(len(sizes) * 0.9))],
        peak_vram_mib=watcher.result.peak_vram_mib,
        sampled=len(picked),
    )


def _texts(chunks: Iterable[object]) -> list[str]:
    """Chunk objects vary between libraries; all of them carry their text."""
    out = []
    for chunk in chunks:
        text = getattr(chunk, "text", None)
        out.append(text if isinstance(text, str) else str(chunk))
    return out
