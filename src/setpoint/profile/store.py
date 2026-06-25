"""Reading and writing profiles.

Two rules from the spec are enforced here rather than left to callers. A profile whose
measurement is not reliable is refused on write, and `reliable` is recomputed on read
instead of being trusted: the files are meant to be edited by hand, so the flag in the
file is a claim, not a fact.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .signature import signature_id
from .types import (
    SCHEMA,
    Baseline,
    Config,
    Measurement,
    ModelRef,
    Objective,
    Profile,
    ProfileError,
    Signature,
    Summary,
    Target,
)

HOME_ENV_VAR = "SETPOINT_HOME"
PROFILE_SUFFIX = ".yaml"


def home() -> Path:
    """Where setpoint keeps its state, following the platform convention."""
    override = os.environ.get(HOME_ENV_VAR)
    if override:
        return Path(override).expanduser()
    appdata = os.environ.get("APPDATA")
    if appdata and os.name == "nt":
        return Path(appdata) / "setpoint"
    config = os.environ.get("XDG_CONFIG_HOME")
    return (Path(config) if config else Path.home() / ".config") / "setpoint"


def profiles_dir() -> Path:
    return home() / "profiles"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save(profile: Profile, directory: Path | None = None) -> Path:
    """Write a profile. Raises `ProfileError` if its evidence does not support it."""
    if not profile.writable:
        raise ProfileError(f"this measurement cannot back a profile: {profile.why_not_writable()}")
    target = directory or profiles_dir()
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{signature_id(profile.signature)}{PROFILE_SUFFIX}"
    path.write_text(dumps(profile), encoding="utf-8")
    return path


def load(path: str | Path) -> Profile:
    """Read one profile file."""
    file_path = Path(path)
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProfileError(f"cannot read {file_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProfileError(f"{file_path} does not hold a profile")
    return from_mapping(raw, source=file_path)


def load_all(directory: Path | None = None) -> list[Profile]:
    """Every readable profile in the directory. Unreadable files are skipped, not fatal."""
    target = directory or profiles_dir()
    if not target.is_dir():
        return []
    profiles: list[Profile] = []
    for path in sorted(target.glob(f"*{PROFILE_SUFFIX}")):
        try:
            profiles.append(load(path))
        except ProfileError:
            continue
    return profiles


def find(signature: Signature, directory: Path | None = None) -> Profile | None:
    """The profile for this exact world, or nothing. Partial matches do not exist."""
    target = directory or profiles_dir()
    path = target / f"{signature_id(signature)}{PROFILE_SUFFIX}"
    if not path.is_file():
        return None
    profile = load(path)
    return profile if profile.signature.matches(signature) else None


def dumps(profile: Profile) -> str:
    """Serialise to YAML, keeping the field order the spec presents."""
    return yaml.safe_dump(to_mapping(profile), sort_keys=False, allow_unicode=True)


def to_mapping(profile: Profile) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "signature": {
            "model_digest": profile.signature.model_digest,
            "model_digest_kind": profile.signature.model_digest_kind,
            "model_size_bytes": profile.signature.model_size_bytes,
            "gpu": profile.signature.gpu,
            "vram_total_mb": profile.signature.vram_total_mb,
            "driver": profile.signature.driver,
            "backend": profile.signature.backend,
            "platform": profile.signature.platform,
        },
        "model": _model_mapping(profile.model),
        "target": {
            "context": profile.target.context,
            "optimize": profile.target.optimize.value,
        },
        "config": _config_mapping(profile.config),
        "measurement": _measurement_mapping(profile.measurement),
        "baseline": {
            "label": profile.baseline.label,
            "config": _config_mapping(profile.baseline.config),
            "decode_tok_s": profile.baseline.decode_tok_s,
            "speedup": profile.baseline.speedup,
        },
        "created": profile.created,
    }
    if profile.notes:
        payload["notes"] = list(profile.notes)
    return payload


def from_mapping(raw: dict[str, Any], source: Path | None = None) -> Profile:
    where = f" in {source}" if source else ""
    schema = raw.get("schema")
    if schema != SCHEMA:
        raise ProfileError(f"unsupported schema {schema!r}{where}; setpoint reads {SCHEMA}")

    measurement = _read_measurement(_section(raw, "measurement", where), where)
    if not measurement.reliable:
        raise ProfileError(
            f"the measurement{where} does not support a profile: {measurement.why_unreliable()}"
        )

    signature_raw = _section(raw, "signature", where)
    target_raw = _section(raw, "target", where)
    baseline_raw = _section(raw, "baseline", where)

    return Profile(
        signature=Signature(
            model_digest=_require(signature_raw, "model_digest", str, where),
            model_digest_kind=_require(signature_raw, "model_digest_kind", str, where),
            model_size_bytes=_require(signature_raw, "model_size_bytes", int, where),
            gpu=_require(signature_raw, "gpu", str, where),
            vram_total_mb=_require(signature_raw, "vram_total_mb", int, where),
            driver=_require(signature_raw, "driver", str, where),
            backend=_require(signature_raw, "backend", str, where),
            platform=_require(signature_raw, "platform", str, where),
        ),
        target=Target(
            context=_require(target_raw, "context", int, where),
            optimize=_read_objective(target_raw.get("optimize"), where),
        ),
        config=_read_config(_section(raw, "config", where)),
        measurement=measurement,
        baseline=Baseline(
            label=_require(baseline_raw, "label", str, where),
            config=_read_config(baseline_raw.get("config") or {}),
            decode_tok_s=float(_require(baseline_raw, "decode_tok_s", (int, float), where)),
            speedup=float(_require(baseline_raw, "speedup", (int, float), where)),
        ),
        created=_require(raw, "created", str, where),
        model=_read_model(raw.get("model")),
        notes=tuple(str(n) for n in raw.get("notes") or ()),
    )


def _model_mapping(model: ModelRef) -> dict[str, Any]:
    return {
        "name": model.name,
        "architecture": model.architecture,
        "file_type": model.file_type,
        "path": model.path,
    }


def _read_model(raw: Any) -> ModelRef:
    if not isinstance(raw, dict):
        return ModelRef()
    return ModelRef(
        name=raw.get("name"),
        architecture=raw.get("architecture"),
        file_type=raw.get("file_type"),
        path=raw.get("path"),
    )


def _config_mapping(config: Config) -> dict[str, Any]:
    return {
        "n_gpu_layers": config.n_gpu_layers,
        "n_cpu_moe": config.n_cpu_moe,
        "cache_type_k": config.cache_type_k,
        "cache_type_v": config.cache_type_v,
        "flash_attn": config.flash_attn,
        "batch_size": config.batch_size,
        "ubatch_size": config.ubatch_size,
        "threads": config.threads,
        "tensor_overrides": list(config.tensor_overrides),
    }


def _measurement_mapping(measurement: Measurement) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "runs": measurement.runs,
        "reliable": measurement.reliable,
        "decode_tok_s": _summary_mapping(measurement.decode_tok_s),
    }
    if measurement.prefill_tok_s is not None:
        payload["prefill_tok_s"] = _summary_mapping(measurement.prefill_tok_s)
    if measurement.peak_vram_mb is not None:
        payload["peak_vram_mb"] = measurement.peak_vram_mb
    if measurement.avg_watt is not None:
        payload["avg_watt"] = measurement.avg_watt
    payload["measured_at"] = measurement.measured_at
    return payload


def _summary_mapping(summary: Summary) -> dict[str, float]:
    return {"median": summary.median, "iqr": summary.iqr, "spread": summary.spread}


def _read_measurement(raw: dict[str, Any], where: str) -> Measurement:
    return Measurement(
        runs=_require(raw, "runs", int, where),
        decode_tok_s=_read_summary(_section(raw, "decode_tok_s", where), where),
        measured_at=_require(raw, "measured_at", str, where),
        prefill_tok_s=(
            _read_summary(raw["prefill_tok_s"], where) if raw.get("prefill_tok_s") else None
        ),
        peak_vram_mb=raw.get("peak_vram_mb"),
        avg_watt=raw.get("avg_watt"),
    )


def _read_summary(raw: Any, where: str) -> Summary:
    if not isinstance(raw, dict):
        raise ProfileError(f"a measurement statistic{where} is not a mapping")
    return Summary(
        median=float(_require(raw, "median", (int, float), where)),
        iqr=float(_require(raw, "iqr", (int, float), where)),
        spread=float(_require(raw, "spread", (int, float), where)),
    )


def _read_config(raw: Any) -> Config:
    if not isinstance(raw, dict):
        return Config()
    return Config(
        n_gpu_layers=raw.get("n_gpu_layers"),
        n_cpu_moe=raw.get("n_cpu_moe"),
        cache_type_k=raw.get("cache_type_k") or "f16",
        cache_type_v=raw.get("cache_type_v") or "f16",
        flash_attn=raw.get("flash_attn"),
        batch_size=raw.get("batch_size"),
        ubatch_size=raw.get("ubatch_size"),
        threads=raw.get("threads"),
        tensor_overrides=tuple(str(o) for o in raw.get("tensor_overrides") or ()),
    )


def _read_objective(value: Any, where: str) -> Objective:
    try:
        return Objective(value)
    except ValueError:
        raise ProfileError(f"unknown optimize target {value!r}{where}") from None


def _section(raw: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ProfileError(f"missing or malformed {key!r} section{where}")
    return value


def _require(raw: dict[str, Any], key: str, kind: type | tuple[type, ...], where: str) -> Any:
    value = raw.get(key)
    if value is None or isinstance(value, bool) or not isinstance(value, kind):
        raise ProfileError(f"missing or malformed {key!r}{where}")
    return value
