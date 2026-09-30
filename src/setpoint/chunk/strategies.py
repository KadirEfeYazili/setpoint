"""The chunking strategies setpoint measures, and what each one costs to run.

setpoint does not split text. The strategies come from an existing library; what is
here is the list of them, what each one needs, and whether it can run on this machine
at all. The library is an optional extra, so nothing is imported until asked for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

DEFAULT_CHUNK_TOKENS = 512


@dataclass(frozen=True)
class Strategy:
    """One way of splitting a corpus, and what running it demands."""

    name: str
    summary: str
    needs_embeddings: bool = False
    needs_long_context: bool = False
    build: Callable[..., object] | None = field(default=None, repr=False)

    @property
    def cost_note(self) -> str:
        if self.needs_long_context:
            return "a long-context embedding model stays resident"
        if self.needs_embeddings:
            return "one embedding pass over the whole corpus"
        return "no model runs"


def available() -> bool:
    """Whether the chunking extra is installed."""
    try:
        import chonkie  # noqa: F401
    except ImportError:
        return False
    return True


def strategies(chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> tuple[Strategy, ...]:
    """The strategies, in increasing order of what they cost to run."""
    return (
        Strategy(
            name="token",
            summary="fixed windows of tokens, ignoring where sentences end",
            build=lambda tokenizer, embedder: _token(tokenizer, chunk_tokens),
        ),
        Strategy(
            name="recursive",
            summary="splits on a hierarchy of separators: paragraph, sentence, word",
            build=lambda tokenizer, embedder: _recursive(tokenizer, chunk_tokens),
        ),
        Strategy(
            name="sentence",
            summary="whole sentences packed up to the size limit",
            build=lambda tokenizer, embedder: _sentence(tokenizer, chunk_tokens),
        ),
        Strategy(
            name="semantic",
            summary="breaks where consecutive sentences stop resembling each other",
            needs_embeddings=True,
            build=lambda tokenizer, embedder: _semantic(embedder, chunk_tokens),
        ),
    )


def by_name(name: str, chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> Strategy | None:
    return next((s for s in strategies(chunk_tokens) if s.name == name), None)


def _token(tokenizer, chunk_tokens: int):
    from chonkie import TokenChunker

    return TokenChunker(tokenizer=tokenizer, chunk_size=chunk_tokens)


def _recursive(tokenizer, chunk_tokens: int):
    from chonkie import RecursiveChunker

    return RecursiveChunker(tokenizer=tokenizer, chunk_size=chunk_tokens)


def _sentence(tokenizer, chunk_tokens: int):
    from chonkie import SentenceChunker

    return SentenceChunker(tokenizer=tokenizer, chunk_size=chunk_tokens)


def _semantic(embedder, chunk_tokens: int):
    from chonkie import SemanticChunker

    return SemanticChunker(embedding_model=embedder, chunk_size=chunk_tokens)
