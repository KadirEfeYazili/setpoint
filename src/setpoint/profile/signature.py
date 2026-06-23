"""Profile signatures.

The digest covers the GGUF header rather than the whole file. The header carries every
tensor name, shape, quantization and offset, so two different models cannot share one,
and `setpoint run` does not have to read gigabytes before it can look a profile up.
What it does not catch is corrupted tensor data; `content` exists for callers who need
that and can pay for it. See spec section 4.1.
"""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path

from ..backend import BackendBuild
from ..hardware import HardwareSnapshot
from ..model import ModelInfo, read_header
from .types import DIGEST_CONTENT, DIGEST_HEADER, ProfileError, Signature

# Read size for the whole-file digest. Large enough to keep the syscall count low.
_CHUNK_BYTES = 4 * 1024 * 1024


def model_digest(path: str | Path, kind: str = DIGEST_HEADER) -> tuple[str, int]:
    """Digest a model file. Returns `("sha256:...", size_in_bytes)`."""
    file_path = Path(path)
    size = file_path.stat().st_size
    digest = hashlib.sha256()

    if kind == DIGEST_HEADER:
        header = read_header(file_path)
        with open(file_path, "rb") as fh:
            digest.update(fh.read(header.data_offset))
    elif kind == DIGEST_CONTENT:
        with open(file_path, "rb") as fh:
            while chunk := fh.read(_CHUNK_BYTES):
                digest.update(chunk)
    else:
        raise ProfileError(f"unknown digest kind {kind!r}")

    digest.update(str(size).encode("ascii"))
    return f"sha256:{digest.hexdigest()}", size


def platform_tag() -> str:
    """`"windows/amd64"`. Lowercase, stable across the machines setpoint runs on."""
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
    return f"{platform.system().lower()}/{arch.get(machine, machine or 'unknown')}"


def build_signature(
    model: ModelInfo,
    snapshot: HardwareSnapshot,
    backend: BackendBuild,
    gpu_index: int | None = None,
    digest_kind: str = DIGEST_HEADER,
) -> Signature:
    """Describe the world one measurement was taken in."""
    gpu = _select_gpu(snapshot, gpu_index)
    if gpu is None:
        raise ProfileError("a profile needs a GPU, and none was found in the snapshot")
    if not snapshot.driver.driver_version:
        raise ProfileError("a profile needs a driver version, and none could be read")

    digest, size = model_digest(model.path, digest_kind)
    return Signature(
        model_digest=digest,
        model_digest_kind=digest_kind,
        model_size_bytes=size,
        gpu=gpu.name,
        vram_total_mb=gpu.vram_total_bytes // (1024 * 1024),
        driver=snapshot.driver.driver_version,
        backend=str(backend),
        platform=platform_tag(),
    )


def signature_id(signature: Signature) -> str:
    """Stable filename for a signature. A collision here is a signature collision."""
    parts = "\n".join(
        [
            signature.model_digest,
            signature.model_digest_kind,
            str(signature.model_size_bytes),
            signature.gpu,
            str(signature.vram_total_mb),
            signature.driver,
            signature.backend,
            signature.platform,
        ]
    )
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:16]


def _select_gpu(snapshot: HardwareSnapshot, gpu_index: int | None):
    if not snapshot.gpus:
        return None
    if gpu_index is None:
        return max(snapshot.gpus, key=lambda g: g.vram_total_bytes)
    return next((g for g in snapshot.gpus if g.index == gpu_index), None)
