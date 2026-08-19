"""What the panel shows, gathered without measuring anything.

Every figure here already exists behind a command: the budget comes from `budget`, the
profiles and their evidence from the profile store, the history from the regression
sentinel, the switch costs from what `route` measured earlier. Nothing in this module
starts a model or times anything, which is what keeps the panel a viewer.

Keeping the gathering apart from the widgets also makes it testable without a terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .. import budget, route, sentinel
from .. import profile as profiles
from ..hardware import HardwareSnapshot, probe
from ..model import ModelError, ModelInfo, analyze, local_models

DEFAULT_CONTEXT = 4096


@dataclass(frozen=True)
class Row:
    """One labelled figure, with the note that says where it came from."""

    label: str
    value: str
    note: str = ""


@dataclass(frozen=True)
class Card:
    """The card as it is right now."""

    name: str | None = None
    total_mib: int | None = None
    free_mib: int | None = None
    used_mib: int | None = None
    utilization_pct: int | None = None
    temperature_c: int | None = None
    throttling: tuple[str, ...] = field(default_factory=tuple)
    detail: str | None = None

    @property
    def rows(self) -> tuple[Row, ...]:
        out = [
            Row("card", self.name or "unknown"),
            Row("vram total", _mib(self.total_mib)),
            Row("vram free", _mib(self.free_mib), "read now"),
            Row("vram in use", _mib(self.used_mib)),
        ]
        if self.utilization_pct is not None:
            out.append(Row("utilisation", f"{self.utilization_pct}%"))
        if self.temperature_c is not None:
            out.append(Row("temperature", f"{self.temperature_c} C"))
        if self.throttling:
            out.append(Row("throttling", ", ".join(self.throttling)))
        return tuple(out)


@dataclass(frozen=True)
class BudgetView:
    """The budget broken into its terms, which is the thing nothing else shows."""

    model: str
    context: int
    rows: tuple[Row, ...] = field(default_factory=tuple)
    plan_rows: tuple[Row, ...] = field(default_factory=tuple)
    alternatives: tuple[Row, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    detail: str | None = None


@dataclass(frozen=True)
class ProfileView:
    """One stored profile, with what was measured around it."""

    signature_id: str
    model: str
    context: int
    decode_tok_s: float | None
    spread: float | None
    speedup: float | None
    n_gpu_layers: int | None
    ubatch_size: int | None
    speculator: str | None
    peak_vram_mib: int | None
    switch_seconds: float | None
    checks: int
    created: str | None

    @property
    def rows(self) -> tuple[Row, ...]:
        return (
            Row("model", self.model),
            Row("context", str(self.context)),
            Row("-ngl", str(self.n_gpu_layers) if self.n_gpu_layers is not None else "-"),
            Row("-ub", str(self.ubatch_size) if self.ubatch_size is not None else "-"),
            Row("speculator", self.speculator or "none measured"),
            Row("decode", _tok_s(self.decode_tok_s), _spread(self.spread)),
            Row("against default", f"{self.speedup:.2f}x" if self.speedup else "-"),
            Row("peak vram", _mib(self.peak_vram_mib)),
            Row("switch cost", _seconds(self.switch_seconds), "measured by route"),
            Row("history", f"{self.checks} check(s)"),
            Row("measured", self.created or "-"),
        )


@dataclass(frozen=True)
class HistoryEntry:
    """One regression check, as the sentinel wrote it."""

    when: str
    decode_tok_s: float | None
    note: str = ""


@dataclass(frozen=True)
class Snapshot:
    """Everything the panel needs for one refresh."""

    card: Card
    models: tuple[str, ...] = field(default_factory=tuple)
    profiles: tuple[ProfileView, ...] = field(default_factory=tuple)
    budget: BudgetView | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def _mib(value: int | None) -> str:
    return f"{value} MiB" if value is not None else "-"


def _tok_s(value: float | None) -> str:
    return f"{value:.2f} t/s" if value is not None else "-"


def _seconds(value: float | None) -> str:
    return f"{value:.2f}s" if value is not None else "not measured"


def _spread(value: float | None) -> str:
    return f"spread {value:.1%}" if value is not None else ""


def read_card(snapshot: HardwareSnapshot | None = None, gpu_index: int | None = None) -> Card:
    """The card's live figures. Cheap enough to poll: NVML answers in milliseconds."""
    snapshot = snapshot or probe(include_wddm=False)
    if not snapshot.gpus:
        return Card(detail="no GPU was found")
    gpu = snapshot.gpus[0] if gpu_index is None else _gpu_at(snapshot, gpu_index)
    if gpu is None:
        return Card(detail=f"no GPU with index {gpu_index}")
    sample = snapshot.sample_for(gpu.index)
    return Card(
        name=gpu.name,
        total_mib=gpu.vram_total_mib,
        free_mib=sample.vram_free_mib if sample else None,
        used_mib=sample.vram_used_mib if sample else None,
        utilization_pct=sample.utilization_pct if sample else None,
        temperature_c=sample.temperature_c if sample else None,
        throttling=tuple(sample.throttle_reasons) if sample else (),
    )


def _gpu_at(snapshot: HardwareSnapshot, index: int):
    return next((g for g in snapshot.gpus if g.index == index), None)


def known_models() -> tuple[str, ...]:
    """Model references this machine already has, shortened for display."""
    return tuple(sorted({resolved.reference.rsplit("/", 1)[-1] for resolved in local_models()}))


def read_budget(
    reference: str,
    context: int = DEFAULT_CONTEXT,
    snapshot: HardwareSnapshot | None = None,
    info: ModelInfo | None = None,
) -> BudgetView:
    """The budget for one model, in the terms it is actually made of."""
    snapshot = snapshot or probe(include_wddm=False)
    try:
        model = info or analyze(_path_of(reference))
    except ModelError as exc:
        return BudgetView(model=reference, context=context, detail=str(exc))

    allowance = budget.runtime_allowance(model, budget.DEFAULT_UBATCH)
    vram = budget.from_snapshot(snapshot, runtime_allowance_bytes=allowance.total_bytes)
    if vram is None:
        return BudgetView(model=reference, context=context, detail="no usable GPU")

    plan = budget.plan(model, context, vram, host_ram_bytes=snapshot.host.available_ram_bytes)
    rows = (
        Row("free", _gib(vram.free_bytes), "measured now" if vram.measured else "assumed"),
        Row("fragmentation", "-" + _gib(vram.fragmentation_bytes), "measured at 8%"),
        Row("runtime allowance", "-" + _gib(vram.runtime_allowance_bytes), budget.CALIBRATION),
        Row("safe ceiling", _gib(vram.ceiling_bytes)),
    )
    offload = plan.offload
    plan_rows = [
        Row("-ngl", str(offload.n_gpu_layers), f"{offload.blocks_on_gpu} of {model.block_count}"),
        Row("weights", _gib(offload.weights_on_gpu_bytes)),
        Row("kv cache", _gib(offload.kv_on_gpu_bytes), "upper bound" if _upper(plan) else ""),
        Row("on gpu", _gib(offload.gpu_bytes)),
    ]
    if offload.next_block_bytes:
        plan_rows.append(
            Row("next block needs", _gib(offload.next_block_bytes), "drift of this moves the plan")
        )
    if not offload.fits_fully:
        plan_rows.append(
            Row("on cpu", _gib(offload.cpu_bytes), f"{offload.cpu_weight_fraction:.0%}")
        )

    alternatives = tuple(
        Row(option.change, f"-ngl {option.n_gpu_layers}", option.effect)
        for option in plan.alternatives
    )
    return BudgetView(
        model=reference,
        context=context,
        rows=rows,
        plan_rows=tuple(plan_rows),
        alternatives=alternatives,
        notes=tuple(plan.notes),
    )


def _upper(plan: object) -> bool:
    estimate = getattr(plan, "kv", None)
    return bool(getattr(estimate, "upper_bound", False))


def _gib(value: int | None) -> str:
    return f"{(value or 0) / 1024**3:.2f} GiB"


def _path_of(reference: str) -> Path:
    from ..model import resolve

    return resolve(reference).path


def read_profiles(
    directory: Path | None = None,
    history_root: Path | None = None,
    loads_dir: Path | None = None,
) -> tuple[ProfileView, ...]:
    """Stored profiles, with the switch cost and history count beside each.

    The three directories are separate arguments because they are three separate
    stores: profiles are edited by hand, history is append-only, and switch costs are
    keyed by the model rather than the signature.
    """
    views = []
    for profile in profiles.load_all(directory):
        history = sentinel.load(
            sentinel.history_path(
                profile.signature.model_digest, profile.target.context, history_root
            )
        )
        cost = route.read_load(profile.signature.model_digest, loads_dir)
        views.append(
            ProfileView(
                signature_id=profiles.signature_id(profile.signature),
                model=profile.model.label,
                context=profile.target.context,
                decode_tok_s=profile.measurement.decode_tok_s.median,
                spread=profile.measurement.decode_tok_s.spread,
                speedup=profile.baseline.speedup,
                n_gpu_layers=profile.config.n_gpu_layers,
                ubatch_size=profile.config.ubatch_size,
                speculator=profile.config.spec_type,
                peak_vram_mib=profile.measurement.peak_vram_mb,
                switch_seconds=cost.median if cost else None,
                checks=len(history),
                created=profile.created,
            )
        )
    return tuple(sorted(views, key=lambda v: (v.model, v.context)))


def read_history(
    signature_id: str,
    directory: Path | None = None,
    history_root: Path | None = None,
) -> tuple[HistoryEntry, ...]:
    """The regression checks for one profile, oldest first."""
    for profile in profiles.load_all(directory):
        if profiles.signature_id(profile.signature) != signature_id:
            continue
        path = sentinel.history_path(
            profile.signature.model_digest, profile.target.context, history_root
        )
        return tuple(
            HistoryEntry(
                when=record.at or "-",
                decode_tok_s=record.statistic.median,
                note=record.note or "",
            )
            for record in sentinel.load(path)
        )
    return ()


def gather(
    reference: str | None = None,
    context: int = DEFAULT_CONTEXT,
    gpu_index: int | None = None,
) -> Snapshot:
    """One refresh worth of everything, with nothing measured to produce it."""
    snapshot = probe(include_wddm=False)
    models = known_models()
    chosen = reference or (models[0] if models else None)
    return Snapshot(
        card=read_card(snapshot, gpu_index),
        models=models,
        profiles=read_profiles(),
        budget=read_budget(chosen, context, snapshot) if chosen else None,
        notes=tuple(snapshot.notes),
    )
