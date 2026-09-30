"""Embedding and tokenizing through setpoint's own measured server.

The chunking library expects an embedding model and a tokenizer. Both are supplied by
a `llama-server` started from a measured profile, which is what makes the cost of a
strategy a cost on this machine rather than on someone's API bill.

The classes here are the pluggable-provider rule in its first real form: the default is
local and keyless, and a remote embedder can take the same place.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence

TIMEOUT_S = 600.0
BATCH = 32


class ServerTokenizer:
    """Token counts from the model that will actually read the chunks.

    Chunk sizes stated in someone else's vocabulary are the wrong sizes: a chunk of
    512 GPT tokens is not 512 tokens of context in a Qwen model, and context is the
    budget being spent.
    """

    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        tokens = _post(f"{self.base}/tokenize", {"content": text}).get("tokens")
        return list(tokens) if isinstance(tokens, list) else []

    def decode(self, tokens: Sequence[int]) -> str:
        if not tokens:
            return ""
        return str(_post(f"{self.base}/detokenize", {"tokens": list(tokens)}).get("content", ""))

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    # Chonkie's protocol name for the same thing.
    def tokenize(self, text: str) -> list[int]:
        return self.encode(text)

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def decode_batch(self, batch: Sequence[Sequence[int]]) -> list[str]:
        return [self.decode(tokens) for tokens in batch]

    def count_tokens_batch(self, texts: Sequence[str]) -> list[int]:
        return [self.count_tokens(text) for text in texts]


def server_embeddings(base: str, dimension: int | None = None):
    """A chonkie embedding provider backed by a local server.

    Built here rather than imported so that nothing in this module requires the extra
    until a strategy that needs embeddings is actually asked for.
    """
    from chonkie import BaseEmbeddings

    class ServerEmbeddings(BaseEmbeddings):
        """The default embedding provider: local, keyless, already measured."""

        def __init__(self) -> None:
            super().__init__()
            self.base = base.rstrip("/")
            self._tokenizer = ServerTokenizer(self.base)
            self._dimension = dimension

        @property
        def dimension(self) -> int:
            if self._dimension is None:
                self._dimension = len(self.embed("dimension probe"))
            return self._dimension

        def embed(self, text: str):
            return self.embed_batch([text])[0]

        def embed_batch(self, texts: Sequence[str]):
            out = []
            for start in range(0, len(texts), BATCH):
                window = list(texts[start : start + BATCH])
                body = _post(f"{self.base}/v1/embeddings", {"input": window, "model": "setpoint"})
                out.extend(_vector(row) for row in body.get("data", []))
            return out

        def get_tokenizer(self):
            return self._tokenizer

        def count_tokens(self, text: str) -> int:
            return self._tokenizer.count_tokens(text)

    return ServerEmbeddings()


def _vector(row: dict):
    """One embedding, as a numpy array when numpy is present and a list otherwise."""
    values = row.get("embedding") or []
    try:
        import numpy

        return numpy.asarray(values, dtype=numpy.float32)
    except ImportError:
        return values


def _post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            body = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}
