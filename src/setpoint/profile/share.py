"""Moving a profile between machines.

A profile is a claim about one machine, so sharing one is only useful when the receiving
machine is the same machine in every way that mattered to the measurement. Everything
here exists to make that check explicit rather than hopeful.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..model import ModelInfo, ResolvedModel, analyze
from .signature import model_digest
from .types import ModelRef, Profile


def for_sharing(profile: Profile) -> Profile:
    """Strip what belongs to one machine and to one person.

    The file path is the obvious case: it is meaningless on another machine and it
    names a directory the sender may not want to publish. What identifies the model
    travels in the signature instead, as a digest.
    """
    kept = ModelRef(
        name=profile.model.name,
        architecture=profile.model.architecture,
        file_type=profile.model.file_type,
        path=None,
    )
    return replace(profile, model=kept)


def find_model(profile: Profile, candidates: list[ResolvedModel]) -> ResolvedModel | None:
    """The local file this profile was measured on, if this machine has it.

    Matched on the digest in the signature, not on the name: two files with the same
    name can be different quantizations, and a profile for one does not describe the
    other.
    """
    wanted = profile.signature.model_digest
    kind = profile.signature.model_digest_kind
    for candidate in candidates:
        try:
            digest, size = model_digest(candidate.path, kind)
        except Exception:
            continue
        if digest == wanted and size == profile.signature.model_size_bytes:
            return candidate
    return None


def adopt(profile: Profile, path: str | Path) -> Profile:
    """Attach a local file to an imported profile, filling in what sharing removed."""
    info: ModelInfo | None = None
    try:
        info = analyze(path)
    except Exception:
        info = None
    model = ModelRef(
        name=profile.model.name or (info.name if info else None),
        architecture=profile.model.architecture or (info.architecture if info else None),
        file_type=profile.model.file_type or (info.file_type if info else None),
        path=str(path),
    )
    return replace(profile, model=model)
