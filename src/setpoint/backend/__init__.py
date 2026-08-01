"""Backend adapters: turn a configuration into a measurement.

Phase 1 drives llama.cpp only. The contract in `types.py` is what a second backend
would implement.
"""

from __future__ import annotations

from .llamacpp import (
    BINARY_ENV_VAR,
    BINARY_NAME,
    SERVER_BINARY_NAME,
    SERVER_ENV_VAR,
    LlamaCppBackend,
    find_binary,
    find_server_binary,
    parse_devices,
    parse_output,
    select_device,
    server_argv,
)
from .types import (
    BackendBuild,
    BackendDevice,
    BackendError,
    BenchRun,
    BenchSample,
    MeasurementKind,
    RunSpec,
)

__all__ = [
    "BINARY_ENV_VAR",
    "BINARY_NAME",
    "SERVER_BINARY_NAME",
    "SERVER_ENV_VAR",
    "BackendBuild",
    "BackendDevice",
    "BackendError",
    "BenchRun",
    "BenchSample",
    "LlamaCppBackend",
    "MeasurementKind",
    "RunSpec",
    "find_binary",
    "find_server_binary",
    "parse_devices",
    "parse_output",
    "select_device",
    "server_argv",
]
