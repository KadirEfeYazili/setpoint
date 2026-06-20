"""Turn a model reference into a GGUF file path.

A reference is either a filesystem path or an Ollama model name. Ollama stores plain
GGUF blobs behind a manifest, so resolving a name is a directory lookup, not a backend
integration: setpoint reads the manifest and never talks to the Ollama process.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .types import ModelError

OLLAMA_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
DEFAULT_REGISTRY = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"
DEFAULT_TAG = "latest"


@dataclass(frozen=True)
class ResolvedModel:
    path: Path
    reference: str
    source: str


def resolve(reference: str) -> ResolvedModel:
    """Resolve a path or an Ollama name. Raises `ModelError` if nothing matches."""
    candidate = Path(reference).expanduser()
    if candidate.is_file():
        return ResolvedModel(candidate, reference, "path")

    path = resolve_ollama(reference)
    if path is not None:
        return ResolvedModel(path, reference, "ollama")

    raise ModelError(f"no model file found for {reference!r}")


def ollama_root() -> Path:
    """Where Ollama keeps its manifests and blobs."""
    override = os.environ.get("OLLAMA_MODELS")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".ollama" / "models"


def resolve_ollama(reference: str) -> Path | None:
    """Look up `[registry/][namespace/]name[:tag]` in the local Ollama store."""
    manifest = _manifest_path(reference)
    if manifest is None or not manifest.is_file():
        return None
    try:
        layers = json.loads(manifest.read_text(encoding="utf-8")).get("layers", [])
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"cannot read Ollama manifest {manifest}: {exc}") from exc

    for layer in layers:
        if layer.get("mediaType") != OLLAMA_MODEL_MEDIA_TYPE:
            continue
        digest = str(layer.get("digest", "")).replace(":", "-")
        blob = ollama_root() / "blobs" / digest
        if blob.is_file():
            return blob
        raise ModelError(f"Ollama manifest points at a missing blob: {blob}")
    return None


def _manifest_path(reference: str) -> Path | None:
    name, _, tag = reference.partition(":")
    parts = name.split("/")
    if not all(parts):
        return None
    if len(parts) == 1:
        parts = [DEFAULT_REGISTRY, DEFAULT_NAMESPACE, parts[0]]
    elif len(parts) == 2:
        parts = [DEFAULT_REGISTRY, *parts]
    elif len(parts) != 3:
        return None
    return ollama_root().joinpath("manifests", *parts, tag or DEFAULT_TAG)
