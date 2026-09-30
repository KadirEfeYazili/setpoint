"""Static model analysis: read a GGUF header, describe what is inside it."""

from __future__ import annotations

from .analyze import LATENT_ATTENTION_ARCHITECTURES, analyze, describe
from .index import ResolvedModel, Store, local_models, resolve, stores
from .reader import GgufError, GgufHeader, TensorEntry, read_header
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
    "Store",
    "TensorEntry",
    "Weights",
    "analyze",
    "describe",
    "local_models",
    "read_header",
    "resolve",
    "stores",
]
