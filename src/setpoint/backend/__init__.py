"""Backend adapters: turn a configuration into a measurement.

Phase 1 drives llama.cpp only. The contract in `types.py` is what a second backend
would implement.
"""

from __future__ import annotations

from .llamacpp import BINARY_ENV_VAR, BINARY_NAME, LlamaCppBackend, find_binary, parse_output
from .types import (
    BackendBuild,
    BackendError,
    BenchRun,
    BenchSample,
    MeasurementKind,
    RunSpec,
)

__all__ = [
    "BINARY_ENV_VAR",
    "BINARY_NAME",
    "BackendBuild",
    "BackendError",
    "BenchRun",
    "BenchSample",
    "LlamaCppBackend",
    "RunSpec",
    "MeasurementKind",
    "find_binary",
    "parse_output",
]
