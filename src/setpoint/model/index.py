"""Finding the GGUF files this machine already has.

A reference is a path or a name. Names are resolved by looking through the places model
files are kept, which is a configuration question rather than an integration: setpoint
reads files off the disk and never starts, connects to, or speaks the protocol of
whatever put them there.

Stores are a list, and no entry in it is privileged. Point `SETPOINT_MODELS_DIR` at your
own directory and it is searched first.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .types import ModelError

# A store either holds the files directly, or holds blobs behind a manifest that maps a
# name to one of them. Both end at a plain GGUF file.
FLAT = "flat"
MANIFEST = "manifest"

SUFFIX = ".gguf"
MODELS_DIR_ENV = "SETPOINT_MODELS_DIR"

# The media type a manifest store marks the weights layer with. Read, not written.
MANIFEST_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
MANIFEST_DEFAULT_REGISTRY = "registry.ollama.ai"
MANIFEST_DEFAULT_NAMESPACE = "library"
MANIFEST_DEFAULT_TAG = "latest"


@dataclass(frozen=True)
class Store:
    """One directory models are kept in, and how it is laid out."""

    label: str
    root: Path
    layout: str


@dataclass(frozen=True)
class ResolvedModel:
    path: Path
    reference: str
    source: str


def stores() -> tuple[Store, ...]:
    """Every place to look, in the order they are looked in.

    A directory that does not exist is still listed: `doctor` and the panel say where
    they searched, and "nothing there" is more useful than a silently shorter list.
    """
    found: list[Store] = []
    for raw in (os.environ.get(MODELS_DIR_ENV) or "").split(os.pathsep):
        if raw.strip():
            found.append(Store("configured", Path(raw.strip()).expanduser(), FLAT))

    home = Path.home()
    manifest_root = os.environ.get("OLLAMA_MODELS")
    found.append(
        Store(
            "manifest store",
            Path(manifest_root).expanduser() if manifest_root else home / ".ollama" / "models",
            MANIFEST,
        )
    )
    found.append(Store("lm studio", home / ".lmstudio" / "models", FLAT))
    found.append(Store("hugging face cache", home / ".cache" / "huggingface" / "hub", FLAT))
    return tuple(found)


def resolve(reference: str) -> ResolvedModel:
    """Resolve a path or a name. Raises `ModelError` if nothing matches."""
    candidate = Path(reference).expanduser()
    if candidate.is_file():
        return ResolvedModel(candidate, reference, "path")

    for store in stores():
        path = _lookup(store, reference)
        if path is not None:
            return ResolvedModel(path, reference, store.label)

    raise ModelError(f"no model file found for {reference!r}")


def local_models() -> list[ResolvedModel]:
    """Every model this machine already has, so one can be found without being named.

    An imported profile carries a digest rather than a path - a path from someone else's
    machine means nothing here - so adopting it means looking through what is installed
    for the file it describes.
    """
    found: list[ResolvedModel] = []
    seen: set[Path] = set()
    for store in stores():
        for resolved in _walk(store):
            if resolved.path not in seen:
                seen.add(resolved.path)
                found.append(resolved)
    return found


def _lookup(store: Store, reference: str) -> Path | None:
    if store.layout == MANIFEST:
        return _manifest_lookup(store, reference)
    return _flat_lookup(store, reference)


def _flat_lookup(store: Store, reference: str) -> Path | None:
    """Match a name against the file names in a directory, with or without the suffix."""
    if not store.root.is_dir():
        return None
    wanted = reference.lower()
    for path in sorted(store.root.rglob(f"*{SUFFIX}")):
        if wanted in (path.name.lower(), path.stem.lower()):
            return path
    return None


def _walk(store: Store) -> list[ResolvedModel]:
    if store.layout == MANIFEST:
        return _walk_manifests(store)
    if not store.root.is_dir():
        return []
    return [
        ResolvedModel(path, path.stem, store.label)
        for path in sorted(store.root.rglob(f"*{SUFFIX}"))
        if path.is_file()
    ]


def _walk_manifests(store: Store) -> list[ResolvedModel]:
    root = store.root / "manifests"
    if not root.is_dir():
        return []
    found: list[ResolvedModel] = []
    for manifest in sorted(root.rglob("*")):
        if not manifest.is_file():
            continue
        reference = _reference_of(manifest, root)
        try:
            path = _manifest_lookup(store, reference) if reference else None
        except ModelError:
            continue
        if path is not None:
            found.append(ResolvedModel(path, reference or str(manifest), store.label))
    return found


def _reference_of(manifest: Path, root: Path) -> str | None:
    """`.../library/qwen3/8b` -> `registry/library/qwen3:8b`."""
    parts = manifest.relative_to(root).parts
    if len(parts) < 2:
        return None
    return "/".join(parts[:-1]) + ":" + parts[-1]


def _manifest_lookup(store: Store, reference: str) -> Path | None:
    """Look up `[registry/][namespace/]name[:tag]` in a manifest store."""
    manifest = _manifest_path(store, reference)
    if manifest is None or not manifest.is_file():
        return None
    try:
        layers = json.loads(manifest.read_text(encoding="utf-8")).get("layers", [])
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"cannot read the manifest at {manifest}: {exc}") from exc

    for layer in layers:
        if layer.get("mediaType") != MANIFEST_MODEL_MEDIA_TYPE:
            continue
        digest = str(layer.get("digest", "")).replace(":", "-")
        blob = store.root / "blobs" / digest
        if blob.is_file():
            return blob
        raise ModelError(f"the manifest points at a missing file: {blob}")
    return None


def _manifest_path(store: Store, reference: str) -> Path | None:
    name, _, tag = reference.partition(":")
    parts = name.split("/")
    if not all(parts):
        return None
    if len(parts) == 1:
        parts = [MANIFEST_DEFAULT_REGISTRY, MANIFEST_DEFAULT_NAMESPACE, parts[0]]
    elif len(parts) == 2:
        parts = [MANIFEST_DEFAULT_REGISTRY, *parts]
    elif len(parts) != 3:
        return None
    return store.root.joinpath("manifests", *parts, tag or MANIFEST_DEFAULT_TAG)
