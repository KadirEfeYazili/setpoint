"""Static model analysis: read a GGUF header, describe what is inside it."""

from __future__ import annotations

from .analyze import LATENT_ATTENTION_ARCHITECTURES, analyze, describe
from .reader import GgufError, GgufHeader, TensorEntry, read_header
from .resolve import ResolvedModel, resolve, resolve_ollama
from .types import Attention, Experts, ModelError, ModelInfo, QuantShare, Weights

__all__ = [
    "LATENT_ATTENTION_ARCHITECTURES",
    "Attention",
    "Experts",
    "GgufError",
    "GgufHeader",
    "ModelError",
    "ModelInfo",
    "QuantShare",
    "ResolvedModel",
    "TensorEntry",
    "Weights",
    "analyze",
    "describe",
    "read_header",
    "resolve",
    "resolve_ollama",
]
