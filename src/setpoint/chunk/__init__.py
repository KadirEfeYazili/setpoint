"""Chunking: measuring what each strategy costs on this machine.

setpoint does not split text and does not embed it. Both come from existing tools. The
contribution here is the column nobody publishes: peak VRAM per strategy, and what it
leaves for the model that has to answer.

The chunking library is an optional extra, so this package imports it lazily.
"""

from __future__ import annotations

from .measure import (
    CALIBRATION_BYTES,
    MIN_CORPUS_BYTES,
    SIZE_SAMPLE,
    Corpus,
    Document,
    Result,
    calibrate,
    read_corpus,
    run,
)
from .providers import ServerTokenizer, chonkie_tokenizer, server_embeddings
from .strategies import (
    DEFAULT_CHUNK_TOKENS,
    Strategy,
    available,
    by_name,
    strategies,
)

__all__ = [
    "CALIBRATION_BYTES",
    "DEFAULT_CHUNK_TOKENS",
    "MIN_CORPUS_BYTES",
    "SIZE_SAMPLE",
    "Corpus",
    "Document",
    "Result",
    "ServerTokenizer",
    "Strategy",
    "available",
    "by_name",
    "calibrate",
    "chonkie_tokenizer",
    "read_corpus",
    "run",
    "server_embeddings",
    "strategies",
]
