"""Command line entry point.

Exit codes are part of the contract: 0 healthy, 1 a problem was found, 2 setpoint
could not complete the check. Data goes to stdout, diagnostics to stderr, so every
command composes with jq and shell pipelines.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import (
    __version__,
    budget,
    doctor,
    export,
    monitor,
    quant,
    route,
    sentinel,
    speculate,
    tune,
)
from . import profile as profiles
from .backend import (
    BINARY_ENV_VAR,
    BINARY_NAME,
    SERVER_BINARY_NAME,
    SERVER_ENV_VAR,
    BackendError,
    LlamaCppBackend,
    MeasurementKind,
    find_server_binary,
    select_device,
    server_argv,
)
from .hardware import probe as probe_hardware
from .model import ModelError, analyze, local_models, resolve
from .render import Style, color_enabled, gib, human_bytes, short_path, term_width, wrap

EXIT_OK = 0
EXIT_PROBLEM = 1
EXIT_ERROR = 2

# How far a re-measurement may drift before a profile stops describing the machine.
# Wider than the run-to-run spread observed on real hardware, narrow enough that a
# driver or backend change shows up.
REGRESSION_TOLERANCE = 0.05

_MARKS = {
    doctor.Outcome.PASS: ("ok", "green"),
    doctor.Outcome.FAIL: ("!!", "red"),
    doctor.Outcome.SKIP: ("--", "grey"),
}


def _as_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _as_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, tuple | list):
        return [_as_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    return value


def _render_report(report: doctor.Report, style: Style) -> None:
    width = term_width()

    for finding in report.findings:
        mark, colour = _MARKS[finding.outcome]
        painted = getattr(style, colour)(mark)
        severity = ""
        if finding.outcome is doctor.Outcome.FAIL:
            severity = style.dim(f"  [{finding.severity.value}]")

        print(f"{painted}  {style.bold(finding.title)}{severity}")
        for line in wrap(finding.what, width - 4, indent=""):
            print(f"    {line}")
        if finding.why:
            for line in wrap(finding.why, width - 4, indent=""):
                print(style.dim(f"    {line}"))
        if finding.fix:
            fix_lines = wrap(finding.fix, width - 9, indent="")
            print(f"    {style.blue('fix:')} {fix_lines[0]}")
            for line in fix_lines[1:]:
                print(f"         {line}")
        print()

    failed = len(report.failures)
    skipped = len(report.skipped)
    total = len(report.findings)

    if failed:
        summary = style.red(f"{failed} problem(s) found")
    else:
        summary = style.green("no problems found")
    tail = f"{total} checks run"
    if skipped:
        tail += f", {skipped} could not be evaluated"
    print(f"{summary}  {style.dim('(' + tail + ')')}")


def cmd_doctor(args: argparse.Namespace) -> int:
    try:
        report = doctor.run()
    except Exception as exc:
        print(f"setpoint: doctor failed to run: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _render_report(report, Style(color_enabled()))
    return report.exit_code


def cmd_hardware(args: argparse.Namespace) -> int:
    try:
        snapshot = probe_hardware(include_wddm=not args.no_wddm)
    except Exception as exc:
        print(f"setpoint: hardware probe failed: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.json:
        print(json.dumps(_as_jsonable(snapshot), indent=2, default=str))
        return EXIT_OK

    style = Style(color_enabled())
    host = snapshot.host
    print(style.bold("host"))
    print(f"  {host.os} {host.arch}, python {host.python_version}")
    if host.total_ram_bytes:
        print(f"  ram {host.total_ram_bytes / (1024**3):.1f} GiB")

    print(style.bold("\ndriver"))
    if snapshot.driver.driver_version:
        cuda = snapshot.driver.cuda_driver_version or "unknown"
        print(f"  version {snapshot.driver.driver_version}, supports CUDA {cuda}")
    else:
        print(f"  {style.grey(snapshot.driver.detail or 'unavailable')}")

    if snapshot.gpus:
        print(style.bold("\ngpu"))
    for gpu in snapshot.gpus:
        sample = snapshot.sample_for(gpu.index)
        print(f"  [{gpu.index}] {gpu.name}")
        print(f"      vram {gpu.vram_total_mib} MiB total", end="")
        if sample and sample.vram_free_mib is not None:
            print(f", {sample.vram_free_mib} MiB free", end="")
        print()
        if gpu.compute_capability:
            print(
                f"      compute capability {gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
            )
        if sample:
            bits = []
            if sample.temperature_c is not None:
                bits.append(f"{sample.temperature_c} C")
            if sample.power_w is not None:
                bits.append(f"{sample.power_w:.1f} W")
            if sample.pcie_width is not None:
                bits.append(f"pcie x{sample.pcie_width} gen{sample.pcie_gen}")
            if bits:
                print(f"      {', '.join(bits)}")
            if sample.throttle_reasons:
                print(f"      throttle: {', '.join(sample.throttle_reasons)}")

    if snapshot.adapters:
        print(style.bold("\nwddm adapters"))
        for adapter in snapshot.adapters:
            ded = (adapter.dedicated_bytes or 0) / (1024**2)
            shared = (adapter.shared_bytes or 0) / (1024**2)
            print(f"  {adapter.instance}")
            print(f"      dedicated {ded:>10.1f} MiB    shared {shared:>10.1f} MiB")

    for note in snapshot.notes:
        print(style.grey(f"\nnote: {note}"))
    return EXIT_OK


def _row(style: Style, label: str, value: str, comment: str = "") -> None:
    line = f"  {label:<22}{value:>10}"
    print(f"{line}   {style.dim(comment)}" if comment else line)


def _render_budget(
    plan: budget.BudgetPlan, reference: str, style: Style, allowance_note: str | None = None
) -> None:
    model, vram, offload = plan.model, plan.vram, plan.offload
    width = term_width()

    kind = "MoE" if model.is_moe else "dense"
    parts = [f"{model.architecture} {model.parameter_label} {kind}"]
    if model.file_type:
        parts.append(model.file_type)
    parts.append(f"{model.block_count} blocks")
    print(style.bold("model"))
    print(f"  {reference}   {style.dim(', '.join(parts))}")
    print(f"  {style.grey(short_path(model.path))}   {style.grey(gib(model.file_bytes))}")
    shape = []
    if model.train_context:
        shape.append(f"trained context {model.train_context}")
    heads, kv_heads = model.attention.uniform_head_count, model.attention.uniform_head_count_kv
    if heads and kv_heads:
        shape.append(f"{heads} heads over {kv_heads} KV heads")
    if model.experts:
        shape.append(f"{model.experts.used} of {model.experts.count} experts active")
    if shape:
        print(f"  {style.dim(', '.join(shape))}")

    title = f"vram  {vram.gpu_name}" if vram.gpu_name else "vram"
    print(style.bold(f"\n{title}"))
    _row(style, "total", gib(vram.total_bytes))
    _row(style, "free", gib(vram.free_bytes), "measured now" if vram.measured else "assumed")
    _row(style, "fragmentation", "-" + gib(vram.fragmentation_bytes))
    if vram.reserve_bytes:
        _row(style, "your reserve", "-" + gib(vram.reserve_bytes))
    _row(
        style,
        "runtime allowance",
        "-" + gib(vram.runtime_allowance_bytes),
        allowance_note or "",
    )
    _row(style, "safe ceiling", gib(vram.ceiling_bytes))
    if vram.detail:
        print(f"  {style.grey(vram.detail)}")

    print(style.bold(f"\nneed  at {plan.context} tokens"))
    _row(style, "weights", gib(model.weights.total_bytes))
    if plan.kv:
        label = f"KV cache {plan.kv.cache_type_k}/{plan.kv.cache_type_v}"
        note = f"{human_bytes(plan.kv.bytes_per_token)} per token"
        if plan.kv.upper_bound:
            note += ", upper bound"
        _row(style, label, gib(plan.kv.total_bytes), note)
    else:
        _row(style, "KV cache", "unknown", "not modelled for this architecture")
    _row(style, "total", gib(plan.required_bytes))

    print(style.bold("\nplan"))
    placed = f"{offload.blocks_on_gpu} of {offload.block_count}"
    _row(style, f"-ngl {offload.n_gpu_layers}", placed, "blocks on the GPU")
    _row(
        style,
        "on gpu",
        gib(offload.gpu_bytes),
        f"weights {gib(offload.weights_on_gpu_bytes)} + cache {gib(offload.kv_on_gpu_bytes)}",
    )
    if offload.next_block_bytes:
        _row(
            style,
            "next block needs",
            gib(offload.next_block_bytes),
            "desktop usage drifting by this much moves the plan",
        )

    if not offload.fits_fully:
        left_behind = f"{offload.cpu_weight_fraction:.0%} of the model"
    elif offload.cpu_bytes:
        left_behind = "the token embedding, which llama.cpp keeps in RAM"
    else:
        left_behind = "nothing; this model ties its embedding to the output"
    _row(style, "on cpu", gib(offload.cpu_bytes), left_behind)

    if plan.alternatives:
        print(style.bold("\ninstead"))
        for option in plan.alternatives:
            freed = f"frees {gib(option.freed_bytes)}" if option.freed_bytes else ""
            _row(style, option.change, f"-ngl {option.n_gpu_layers}", freed)
            for line in wrap(option.effect, width - 6):
                print(style.dim(f"      {line}"))

    for note in plan.notes:
        print()
        for line in wrap(f"note: {note}", width - 2):
            print(style.grey(f"  {line}"))


def cmd_budget(args: argparse.Namespace) -> int:
    try:
        resolved = resolve(args.model)
        model = analyze(resolved.path)
    except ModelError as exc:
        print(f"setpoint: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except OSError as exc:
        print(f"setpoint: cannot read {args.model}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.overhead is None:
        estimated = budget.runtime_allowance(model, args.ubatch)
        allowance, allowance_note = estimated.total_bytes, estimated.detail
    else:
        allowance = args.overhead * budget.MIB
        allowance_note = "your value"
    ram: int | None = None
    if args.vram is not None:
        vram = budget.assumed(
            args.vram * budget.MIB,
            reserve_bytes=args.reserve * budget.MIB,
            fragmentation_pct=args.fragmentation,
            runtime_allowance_bytes=allowance,
        )
    else:
        snapshot = probe_hardware(include_wddm=False)
        ram = snapshot.host.available_ram_bytes
        vram = budget.from_snapshot(
            snapshot,
            gpu_index=args.gpu,
            reserve_bytes=args.reserve * budget.MIB,
            fragmentation_pct=args.fragmentation,
            runtime_allowance_bytes=allowance,
        )
        if vram is None:
            print(
                "setpoint: no NVIDIA GPU was found. Pass --vram MB to budget against a "
                "card setpoint cannot see.",
                file=sys.stderr,
            )
            return EXIT_ERROR

    plan = budget.plan(model, args.context, vram, args.kv_type, args.kv_type, host_ram_bytes=ram)
    plan_note = allowance_note

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        _render_budget(plan, resolved.reference, Style(color_enabled()), plan_note)
    return EXIT_OK if plan.offload.fits_fully else EXIT_PROBLEM


def _render_profile(profile: profiles.Profile, style: Style) -> None:
    signature, config, measurement = profile.signature, profile.config, profile.measurement
    print(style.bold("model"))
    print(f"  {profile.model.label}")
    if profile.model.path:
        print(f"  {style.grey(short_path(profile.model.path))}")
    print(f"  {style.dim(signature.model_digest[:23] + '...')} ({signature.model_digest_kind})")

    print()
    print(style.bold("valid for"))
    _row(style, "gpu", "", signature.gpu)
    _row(style, "vram", f"{signature.vram_total_mb} MB")
    _row(style, "driver", signature.driver)
    _row(style, "backend", "", signature.backend)
    _row(style, "platform", "", signature.platform)

    print()
    print(style.bold(f"target  {profile.target.context} tokens, {profile.target.optimize.value}"))
    for name, value in _config_rows(config):
        _row(style, name, value)

    print()
    print(style.bold("measured"))
    _row(
        style,
        "decode",
        f"{measurement.decode_tok_s.median:.2f} t/s",
        f"{measurement.runs} runs, spread {measurement.decode_tok_s.spread:.1%}",
    )
    if measurement.prefill_tok_s:
        _row(style, "prefill", f"{measurement.prefill_tok_s.median:.1f} t/s")
    if measurement.peak_vram_mb:
        _row(style, "peak vram", f"{measurement.peak_vram_mb} MB")
    if measurement.avg_watt:
        _row(style, "power", f"{measurement.avg_watt:.0f} W", "indicative, see NVML sampling")

    baseline = profile.baseline
    verdict = f"{baseline.speedup:.2f}x"
    print()
    print(style.bold("against baseline"))
    _row(style, baseline.label, f"{baseline.decode_tok_s:.2f} t/s")
    painted = style.green(verdict) if baseline.improved else style.yellow(verdict)
    _row(style, "speedup", "", painted + ("" if baseline.improved else "  slower than baseline"))

    for note in profile.notes:
        print()
        print(style.grey(f"  note: {note}"))


def _config_rows(config: profiles.Config) -> list[tuple[str, str]]:
    rows = [("-ngl", str(config.n_gpu_layers)) if config.n_gpu_layers is not None else None]
    if config.n_cpu_moe is not None:
        rows.append(("-ncmoe", str(config.n_cpu_moe)))
    if config.spec_type:
        rows.append(("speculator", config.spec_type))
    rows.append(("kv cache", f"{config.cache_type_k}/{config.cache_type_v}"))
    if config.flash_attn is not None:
        rows.append(("flash attention", "on" if config.flash_attn else "off"))
    for label, value in (
        ("-b", config.batch_size),
        ("-ub", config.ubatch_size),
        ("-t", config.threads),
    ):
        if value is not None:
            rows.append((label, str(value)))
    for override in config.tensor_overrides:
        rows.append(("-ot", override))
    return [r for r in rows if r]


def _profile_export(args: argparse.Namespace, stored: list, style: Style) -> int:
    """Write a profile in the form another machine can read."""
    wanted = [p for p in stored if profiles.signature_id(p.signature).startswith(args.id or "")]
    if not args.id or len(wanted) != 1:
        print(
            f"setpoint: `profile export` needs one profile id; {len(wanted)} matched.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    shareable = profiles.for_sharing(wanted[0])
    text = profiles.dumps(shareable)
    if args.out:
        try:
            Path(args.out).write_text(text, encoding="utf-8")
        except OSError as exc:
            print(f"setpoint: cannot write {args.out}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(f"wrote {profiles.signature_id(shareable.signature)} to {short_path(args.out)}")
        print(style.dim("  the file path was removed; the digest identifies the model"))
    else:
        print(text, end="")
    return EXIT_OK


def _profile_import(args: argparse.Namespace, style: Style) -> int:
    """Read a profile from elsewhere, and refuse it unless it describes this machine."""
    try:
        incoming = profiles.load(args.id)
    except profiles.ProfileError as exc:
        print(f"setpoint: {exc}", file=sys.stderr)
        return EXIT_ERROR

    snapshot = probe_hardware(include_wddm=False)
    facts = profiles.machine_facts(snapshot, args.gpu)
    if facts is None:
        print("setpoint: this machine's GPU could not be identified.", file=sys.stderr)
        return EXIT_ERROR

    differences = profiles.machine_mismatch(incoming.signature, facts)
    if differences:
        print("setpoint: this profile was not measured on this machine.", file=sys.stderr)
        for line in differences:
            print(f"  {line}", file=sys.stderr)
        return EXIT_PROBLEM

    found = profiles.find_model(incoming, list(local_models()))
    if found is None:
        print(
            "setpoint: this machine does not have the model the profile was measured on.",
            file=sys.stderr,
        )
        print(f"  {incoming.signature.model_digest}", file=sys.stderr)
        return EXIT_PROBLEM

    adopted = profiles.adopt(incoming, found.path)
    path = profiles.save(adopted)
    print(f"imported {profiles.signature_id(adopted.signature)}  {adopted.model.label}")
    print(f"  matched {found.reference}")
    print(f"  stored at {short_path(path)}")
    print()
    print(
        style.yellow(
            "someone else's measurement is not evidence on this machine. Verify it with "
            f"`setpoint bench {found.reference}`."
        )
    )
    return EXIT_OK


def cmd_profile(args: argparse.Namespace) -> int:
    directory = profiles.profiles_dir()
    style = Style(color_enabled())

    if args.action == "path":
        print(directory)
        return EXIT_OK

    if args.action == "import":
        return _profile_import(args, style)

    try:
        stored = profiles.load_all(directory)
    except OSError as exc:
        print(f"setpoint: cannot read {directory}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.action == "export":
        return _profile_export(args, stored, style)

    if args.action == "show":
        wanted = [p for p in stored if profiles.signature_id(p.signature).startswith(args.id)]
        if not wanted:
            print(f"setpoint: no profile starts with {args.id!r}", file=sys.stderr)
            return EXIT_ERROR
        if len(wanted) > 1:
            print(f"setpoint: {args.id!r} matches {len(wanted)} profiles", file=sys.stderr)
            return EXIT_ERROR
        if args.json:
            print(json.dumps(profiles.to_mapping(wanted[0]), indent=2))
        else:
            _render_profile(wanted[0], style)
        return EXIT_OK

    if args.json:
        print(json.dumps([profiles.to_mapping(p) for p in stored], indent=2))
        return EXIT_OK

    if not stored:
        print("no profiles yet. `setpoint tune <model>` measures one and writes it here.")
        print(style.grey(f"  {short_path(directory)}"))
        return EXIT_OK

    print(f"  {'id':<18}{'model':<30}{'ctx':>7}{'-ngl':>6}{'decode':>12}{'vs base':>10}")
    for entry in stored:
        layers = entry.config.n_gpu_layers
        print(
            f"  {profiles.signature_id(entry.signature):<18}"
            f"{entry.model.label[:29]:<30}"
            f"{entry.target.context:>7}"
            f"{layers if layers is not None else '-':>6}"
            f"{entry.measurement.decode_tok_s.median:>10.2f} t/s"
            f"{entry.baseline.speedup:>9.2f}x"
        )
    return EXIT_OK


def _tune_effort(context: int, repetitions: int, label: str) -> tune.Effort:
    return tune.Effort(repetitions=repetitions, n_depth=context, label=label)


def _search_space(model: object, threads: int) -> tune.SearchSpace:
    return tune.SearchSpace(
        max_gpu_layers=model.block_count + 1,
        max_threads=max(1, threads),
        moe=model.is_moe,
    )


def _step_line(step: tune.Step, style: Style) -> str:
    score = f"{step.score:8.2f}" if step.score is not None else "       -"
    colour = {
        tune.Verdict.IMPROVED: style.green,
        tune.Verdict.DROPPED: style.grey,
        tune.Verdict.FAILED: style.red,
    }.get(step.verdict, style.dim)
    layers = step.config.n_gpu_layers
    where = f"-ngl {layers}" if layers is not None else "auto"
    note = f"  {step.note}" if step.note else ""
    return (
        f"  {step.stage.value:<9}{where:<10}{score} "
        f"{colour(step.verdict.value):<20}{style.dim(note)}"
    )


def _profile_from(
    result: tune.SearchResult,
    model: object,
    snapshot: object,
    target: profiles.Target,
    gpu_index: int | None,
) -> profiles.Profile | None:
    """Turn a finished search into a profile, or nothing if it cannot back one."""
    best, baseline = result.best, result.baseline
    if best is None or best.run is None or baseline is None:
        return None

    decode = best.run.sample_of(MeasurementKind.DECODE)
    prefill = best.run.sample_of(MeasurementKind.PREFILL)
    if decode is None:
        return None

    measurement = profiles.Measurement.from_statistics(
        decode=decode.throughput,
        measured_at=profiles.now(),
        prefill=prefill.throughput if prefill else None,
        peak_vram_mb=best.peak_vram_mib,
        avg_watt=best.average_power_w,
    )
    signature = profiles.build_signature(model, snapshot, best.run.build, gpu_index)
    return profiles.Profile(
        signature=signature,
        model=profiles.ModelRef(model.name, model.architecture, model.file_type, str(model.path)),
        target=target,
        config=best.config,
        measurement=measurement,
        baseline=_baseline_of(baseline, result),
        created=profiles.now(),
        notes=tuple(n for n in (result.reason, best.detail) if n),
    )


def _same_card(device_name: str, gpu_name: str) -> bool:
    """Whether a backend device and an NVML GPU are the same piece of hardware."""
    a, b = device_name.lower(), gpu_name.lower()
    return a in b or b in a


def _baseline_of(baseline: tune.Trial, result: tune.SearchResult) -> profiles.Baseline:
    """The comparison, including the case where the backend default never started."""
    label = "llama.cpp default (-ngl 99)"
    if baseline.score is None:
        return profiles.Baseline(
            label=label,
            config=baseline.config,
            failed=True,
            detail=baseline.detail or "the baseline configuration did not run",
        )
    return profiles.Baseline(
        label=label,
        config=baseline.config,
        decode_tok_s=baseline.score,
        speedup=result.speedup or 0.0,
    )


def cmd_tune(args: argparse.Namespace) -> int:
    style = Style(color_enabled())
    try:
        resolved = resolve(args.model)
        model = analyze(resolved.path)
    except ModelError as exc:
        print(f"setpoint: {exc}", file=sys.stderr)
        return EXIT_ERROR

    backend = LlamaCppBackend()
    if not backend.available:
        print(
            f"setpoint: {BINARY_NAME} was not found. Put it on PATH or set {BINARY_ENV_VAR}.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    snapshot = probe_hardware(include_wddm=False)
    allowance = budget.runtime_allowance(model, args.ubatch)
    vram = budget.from_snapshot(
        snapshot,
        gpu_index=args.gpu,
        reserve_bytes=args.reserve * budget.MIB,
        runtime_allowance_bytes=allowance.total_bytes,
    )
    if vram is None:
        print("setpoint: no NVIDIA GPU was found to tune against.", file=sys.stderr)
        return EXIT_ERROR

    plan = budget.plan(
        model,
        args.context,
        vram,
        args.kv_type,
        args.kv_type,
        host_ram_bytes=snapshot.host.available_ram_bytes,
    )

    # Device ids are renumbered across reboots, so the one to use is resolved now and
    # checked against the card the profile will be filed under.
    try:
        listing = backend.devices()
    except BackendError as exc:
        print(f"setpoint: {exc}", file=sys.stderr)
        return EXIT_ERROR
    wanted = args.device or vram.gpu_name
    chosen = select_device(listing, wanted)
    if chosen is None:
        offered = ", ".join(f"{d.id} ({d.name})" for d in listing) or "nothing"
        print(f"setpoint: no accelerator matches {wanted!r}. Offered: {offered}", file=sys.stderr)
        return EXIT_ERROR
    devices = (chosen.id,)
    wrong_card = bool(vram.gpu_name) and not _same_card(chosen.name, vram.gpu_name)

    print(style.bold("model"))
    print(f"  {resolved.reference}   {style.dim(model.architecture + ' ' + model.parameter_label)}")
    print(style.bold("\nplan"))
    seeded = plan.offload
    print(
        f"  seeded from the budget: -ngl {seeded.n_gpu_layers}, "
        f"{seeded.blocks_on_gpu} of {model.block_count} blocks"
    )
    print(f"  target {args.context} tokens, optimising for {args.optimize}")
    print(f"  measuring on {chosen.id} -- {chosen.name}")
    free_ram = snapshot.host.available_ram_bytes
    if free_ram and plan.offload.cpu_bytes > free_ram:
        print(
            f"setpoint: the plan leaves {gib(plan.offload.cpu_bytes)} on the CPU but only "
            f"{gib(free_ram)} of RAM is free. Measuring this would page the machine to a "
            "halt. Close something, or lower the context.",
            file=sys.stderr,
        )
        return EXIT_ERROR
    if wrong_card:
        print(
            style.yellow(f"  that is not {vram.gpu_name}, so this run cannot back a profile for it")
        )

    # The expert axis is off unless asked for. Measured on a MoE model where the experts
    # are 94% of every block: at equal VRAM, moving them to the CPU ran 35-44% slower
    # than keeping fewer whole blocks on the GPU, because each affected block then costs
    # a round trip instead of one contiguous split. Searching it by default would spend
    # measurements on a move that lost every time it was tried.
    if args.ncmoe is not None and not model.is_moe:
        print(style.yellow("  --ncmoe only applies to MoE models; ignoring it"))
        args.ncmoe = None

    seeds = [
        profiles.Config(
            n_gpu_layers=candidate.n_gpu_layers,
            cache_type_k=candidate.cache_type_k,
            cache_type_v=candidate.cache_type_v,
            flash_attn=candidate.flash_attn,
            ubatch_size=args.ubatch,
            n_cpu_moe=args.ncmoe,
        )
        for candidate in plan.candidates
    ]
    baseline = profiles.Config(
        n_gpu_layers=99,
        cache_type_k=args.kv_type,
        cache_type_v=args.kv_type,
        flash_attn=True,
        ubatch_size=args.ubatch,
    )

    measure = tune.BackendMeasure(
        backend=backend,
        model_path=model.path,
        objective=profiles.Objective(args.optimize),
        devices=devices,
        gpu_index=vram.gpu_index,
        vram_ceiling_bytes=vram.ceiling_bytes,
        expected_free_bytes=vram.free_bytes,
    )

    print(style.bold("\nsearch"))
    print(style.dim("  interrupt with Ctrl-C to keep the best result so far"))
    result = tune.search(
        seeds=seeds,
        space=_search_space(model, snapshot.host.cpu_count or 8),
        measure=measure,
        screen_effort=_tune_effort(min(args.context, args.screen_context), 2, "screen"),
        full_effort=_tune_effort(args.context, args.repetitions, "full"),
        baseline=baseline,
        measurement_budget=args.budget,
        on_step=lambda step: print(_step_line(step, style)),
    )

    target = profiles.Target(context=args.context, optimize=profiles.Objective(args.optimize))
    profile = _profile_from(result, model, snapshot, target, vram.gpu_index)

    if args.json:
        print(
            json.dumps(
                {
                    "reason": result.reason,
                    "interrupted": result.interrupted,
                    "measurements": result.measurements,
                    "speedup": result.speedup,
                    "profile": profiles.to_mapping(profile) if profile else None,
                },
                indent=2,
            )
        )
    else:
        _render_tune_result(result, profile, style)

    if wrong_card:
        print(
            style.yellow(
                f"\nnot written: measured on {chosen.name}, but the profile would "
                f"claim {vram.gpu_name}"
            )
        )
        return EXIT_PROBLEM
    if profile is None:
        return EXIT_PROBLEM
    if not profile.writable:
        print(style.yellow(f"\nnot written: {profile.why_not_writable()}"))
        return EXIT_PROBLEM
    if args.dry_run:
        print(style.dim("\ndry run: nothing was written"))
        return EXIT_OK
    path = profiles.save(profile)
    print(f"\nwritten to {short_path(path)}")
    _seed_history(profile, result.best, style)
    return EXIT_OK


def _seed_history(profile: profiles.Profile, best: tune.Trial | None, style: Style) -> None:
    """Keep the winning run's raw repetitions beside the profile.

    A profile stores a summary, which is right for a file people read and edit, but a
    summary cannot be compared statistically. Without this the sentinel could only ask
    whether today is within five percent of a number from last week, and a fixed
    percentage cannot tell a real regression from a busy afternoon.
    """
    sample = best.run.sample_of(MeasurementKind.DECODE) if best and best.run else None
    if sample is None:
        return
    history = sentinel.history_path(profile.signature.model_digest, profile.target.context)
    record = sentinel.record_of(
        profile,
        tuple(sample.throughput.samples),
        profile.measurement.measured_at,
        peak_vram_mb=profile.measurement.peak_vram_mb,
        note="written with the profile",
    )
    try:
        sentinel.append(history, record)
    except OSError as exc:
        print(style.grey(f"  history not written: {exc}"))


def _render_tune_result(
    result: tune.SearchResult, profile: profiles.Profile | None, style: Style
) -> None:
    print(style.bold("\nresult"))
    print(f"  {result.measurements} measurements, {result.reason}")
    if result.best is None:
        print(style.red("  nothing measurable was found"))
        return

    for label, value in _config_rows(result.best.config):
        _row(style, label, value)
    if result.best.score is not None:
        _row(style, "score", f"{result.best.score:.2f}", f"spread {result.best.spread or 0:.1%}")
    if result.best.peak_vram_mib:
        _row(style, "peak vram", f"{result.best.peak_vram_mib} MiB")

    if result.baseline is not None and result.baseline.score is not None:
        speedup = result.speedup or 0.0
        painted = style.green(f"{speedup:.2f}x") if speedup > 1 else style.yellow(f"{speedup:.2f}x")
        print()
        _row(style, "baseline (-ngl 99)", f"{result.baseline.score:.2f}")
        verdict = "" if speedup > 1 else "  no better than the default"
        _row(style, "speedup", "", painted + verdict)
    elif result.baseline is not None:
        print()
        print(style.green("  the llama.cpp default did not start on this card at all"))
        for line in wrap(f"it reported: {result.baseline.detail}", term_width() - 4):
            print(style.dim(f"    {line}"))
        print(style.dim("  a configuration that runs is the result here, not a ratio"))

    if result.best.detail:
        print()
        for line in wrap(f"note: {result.best.detail}", term_width() - 2):
            print(style.grey(f"  {line}"))


def _stored_profile(
    args: argparse.Namespace, style: Style
) -> tuple[profiles.Profile | None, object, object, int]:
    """Find the profile for this machine and this model, or explain why there is none."""
    try:
        resolved = resolve(args.model)
        model = analyze(resolved.path)
    except ModelError as exc:
        print(f"setpoint: {exc}", file=sys.stderr)
        return None, None, None, EXIT_ERROR

    snapshot = probe_hardware(include_wddm=False)
    backend = LlamaCppBackend()
    if not backend.available:
        print(
            f"setpoint: {BINARY_NAME} was not found, so the signature cannot name a "
            f"backend build. Put it on PATH or set {BINARY_ENV_VAR}.",
            file=sys.stderr,
        )
        return None, model, snapshot, EXIT_ERROR

    # The build is part of the signature, and only a run reports it, so the stored
    # profiles are searched by everything else and the build is checked against them.
    candidates = [
        p
        for p in profiles.load_all()
        if p.model.architecture == model.architecture
        and p.signature.gpu == (snapshot.gpus[0].name if snapshot.gpus else None)
    ]
    digest, size = profiles.model_digest(model.path)
    matched = [
        p
        for p in candidates
        if p.signature.model_digest == digest and p.signature.model_size_bytes == size
    ]
    if args.context is not None:
        matched = [p for p in matched if p.target.context == args.context]

    if not matched:
        where = short_path(profiles.profiles_dir())
        print(
            f"setpoint: no profile for this model on this machine. Run `setpoint tune "
            f"{args.model}` first; profiles live in {where}.",
            file=sys.stderr,
        )
        return None, model, snapshot, EXIT_PROBLEM

    matched.sort(key=lambda p: p.created, reverse=True)
    if len(matched) > 1:
        print(
            style.dim(
                f"  {len(matched)} profiles match; using the newest, for "
                f"{matched[0].target.context} tokens"
            )
        )
    return matched[0], model, snapshot, EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    style = Style(color_enabled())
    profile, model, snapshot, code = _stored_profile(args, style)
    if profile is None:
        return code

    server = find_server_binary()
    if server is None:
        print(
            f"setpoint: {SERVER_BINARY_NAME} was not found. Put it on PATH or set "
            f"{SERVER_ENV_VAR}.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    devices: tuple[str, ...] = ()
    gpu_name = profile.signature.gpu
    try:
        listing = LlamaCppBackend().devices()
    except BackendError:
        listing = ()
    if listing:
        chosen = select_device(listing, args.device or gpu_name)
        if chosen is None:
            offered = ", ".join(f"{d.id} ({d.name})" for d in listing)
            print(
                f"setpoint: no accelerator matches {args.device or gpu_name!r}. Offered: {offered}",
                file=sys.stderr,
            )
            return EXIT_ERROR
        devices = (chosen.id,)
        if not _same_card(chosen.name, gpu_name):
            print(
                style.yellow(
                    f"setpoint: this profile was measured on {gpu_name} but the backend "
                    f"offers {chosen.name}; the settings may not suit it."
                ),
                file=sys.stderr,
            )

    argv = server_argv(
        server,
        profile.model.path or model.path,
        profile.config,
        profile.target.context,
        devices=devices,
        extra=tuple(args.forward),
    )

    print(style.bold("profile"))
    print(f"  {profiles.signature_id(profile.signature)}   {profile.model.label}")
    _row(style, "measured", f"{profile.measurement.decode_tok_s.median:.2f} t/s", profile.created)
    print(style.bold("\ncommand"))
    for line in wrap(" ".join(argv), term_width() - 4):
        print(f"  {line}")

    if args.print_only:
        return EXIT_OK

    print()
    try:
        return subprocess.run(argv, check=False).returncode
    except OSError as exc:
        print(f"setpoint: {SERVER_BINARY_NAME} could not be started: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        return EXIT_OK


def cmd_bench(args: argparse.Namespace) -> int:
    """Re-measure a stored profile and say whether it still holds."""
    style = Style(color_enabled())
    profile, model, snapshot, code = _stored_profile(args, style)
    if profile is None:
        return code

    backend = LlamaCppBackend()
    try:
        listing = backend.devices()
    except BackendError as exc:
        print(f"setpoint: {exc}", file=sys.stderr)
        return EXIT_ERROR
    chosen = select_device(listing, args.device or profile.signature.gpu)
    if chosen is None or not _same_card(chosen.name, profile.signature.gpu):
        print(
            f"setpoint: this profile describes {profile.signature.gpu}, which the backend "
            "does not offer right now.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    gpu_index = snapshot.gpus[0].index if snapshot.gpus else None
    measure = tune.BackendMeasure(
        backend=backend,
        model_path=profile.model.path or model.path,
        objective=profile.target.optimize,
        devices=(chosen.id,),
        gpu_index=gpu_index,
    )
    effort = tune.Effort(
        repetitions=args.repetitions, n_depth=profile.target.context, label="verify"
    )

    print(style.bold("profile"))
    print(f"  {profiles.signature_id(profile.signature)}   {profile.model.label}")
    print(f"  {style.dim('measuring on ' + chosen.id + ' -- ' + chosen.name)}")

    print(style.bold("\nverifying"))
    print(style.dim("  the first run is a warm-up and is thrown away"))
    measure(profile.config, effort)
    now = measure(profile.config, effort)

    if not now.usable:
        print(style.red(f"\nthe stored configuration no longer runs: {now.detail}"))
        return EXIT_PROBLEM

    claimed = profile.measurement.decode_tok_s.median
    drift = now.score / claimed - 1
    print()
    _row(style, "profile claims", f"{claimed:.2f} t/s", profile.created)
    _row(style, "measured now", f"{now.score:.2f} t/s", f"spread {now.spread or 0:.1%}")
    within = abs(drift) <= REGRESSION_TOLERANCE
    painted = style.green if within else style.red
    _row(style, "difference", "", painted(f"{drift:+.1%}"))

    comparison = _record_and_compare(profile, now, style)

    if args.json:
        print(
            json.dumps(
                {
                    "signature_id": profiles.signature_id(profile.signature),
                    "claimed_tok_s": claimed,
                    "measured_tok_s": now.score,
                    "difference": drift,
                    "within_tolerance": within,
                    "peak_vram_mb": now.peak_vram_mib,
                    "verdict": comparison.verdict.value,
                    "verdict_detail": comparison.detail,
                    "p_value": comparison.p_value,
                    "causes": list(comparison.causes),
                },
                indent=2,
            )
        )

    if now.detail:
        print()
        for line in wrap(f"note: {now.detail}", term_width() - 2):
            print(style.grey(f"  {line}"))

    # Two questions get asked here and only one of them has good evidence behind it.
    # Drift against the profile compares today's median with a number from another day;
    # the sentinel compares repetitions with repetitions. Where the sentinel could run it
    # is the better answer, and letting the weaker one raise the alarm anyway is how a
    # monitoring tool teaches people to ignore it.
    tested = comparison.verdict in (sentinel.Verdict.SAME, sentinel.Verdict.SLOWER)
    if comparison.actionable:
        # Re-tuning is the wrong advice when the card was already shared: the search
        # would measure the neighbour's load and write it into the profile as if it
        # were this machine's ceiling.
        if now.busy_before_pct is not None:
            print(
                style.red(
                    f"\nthis run was slower: {comparison.detail}. The card was already "
                    f"{now.busy_before_pct}% busy before it started, so find out what else "
                    "is using it before re-tuning."
                )
            )
        else:
            print(
                style.red(
                    f"\nthis machine has slowed since the last check: {comparison.detail}. "
                    "Re-run `setpoint tune`."
                )
            )
        return EXIT_PROBLEM
    if not within and not tested:
        print(
            style.red(
                f"\nthe profile no longer describes this machine: {drift:+.1%} against a "
                f"{REGRESSION_TOLERANCE:.0%} tolerance. Re-run `setpoint tune`."
            )
        )
        return EXIT_PROBLEM
    if not within:
        print(
            style.yellow(
                f"\nthe profile reads {drift:+.1%} today, but nothing has changed since the "
                "last check. The profile is stale rather than the machine slower."
            )
        )
        return EXIT_OK
    print(style.green("\nthe profile still holds"))
    return EXIT_OK


def cmd_quant(args: argparse.Namespace) -> int:
    """Compare the local quantizations of one model on speed, size and drift."""
    style = Style(color_enabled())
    try:
        resolved = resolve(args.model)
        target = analyze(resolved.path)
    except ModelError as exc:
        print(f"setpoint: {exc}", file=sys.stderr)
        return EXIT_ERROR

    variants = quant.find_variants(target)
    if len(variants) < 2:
        print("only one local quantization of this model, so there is nothing to compare.")
        print(style.dim("  pull another one, then run this again"))
        return EXIT_OK

    snapshot = probe_hardware(include_wddm=False)
    backend = LlamaCppBackend()
    try:
        listing = backend.devices()
    except BackendError as exc:
        print(f"setpoint: {exc}", file=sys.stderr)
        return EXIT_ERROR
    card = snapshot.gpus[0].name if snapshot.gpus else None
    chosen = select_device(listing, args.device or card)
    devices = (chosen.id,) if chosen else ()

    print(style.bold("model"))
    print(f"  {args.model}   {style.dim(_quant_shape(target))}")
    print(style.dim(f"  {len(variants)} local quantizations, most precise first"))

    print(style.bold("\nspeed and size"), style.dim(f"at {args.context} tokens of context"))
    print(f"  {'variant':<26}{'bpw':>6}{'file':>9}{'decode':>12}{'spread':>8}{'peak vram':>12}")
    speeds = {}
    warmed = False
    for variant in variants:
        trial = _measure_variant(variant, backend, devices, args, warm_up=not warmed)
        warmed = True
        speeds[variant.reference] = trial
        _render_variant_speed(variant, trial, style)

    quality_note = _run_quality(variants, args, devices, style)

    print(style.bold("\ndecision"))
    _render_quant_decision(variants, speeds, quality_note, style)

    if args.json:
        print(json.dumps(_quant_mapping(variants, speeds, quality_note), indent=2))
    return EXIT_OK


def _quant_shape(target: object) -> str:
    return (
        f"{target.architecture} {target.parameter_label}, "
        f"{target.block_count} blocks, vocab {target.vocab_size}"
    )


def _quant_label(reference: str) -> str:
    """Ollama references carry a registry prefix that says nothing here."""
    return reference.rsplit("/", 1)[-1]


def _measure_variant(
    variant: quant.Variant,
    backend: LlamaCppBackend,
    devices: tuple[str, ...],
    args: argparse.Namespace,
    warm_up: bool,
) -> tune.Trial:
    measure = tune.BackendMeasure(
        backend=backend,
        model_path=variant.path,
        devices=devices,
    )
    config = profiles.Config(
        n_gpu_layers=variant.info.block_count + 1,
        ubatch_size=args.ubatch,
        flash_attn=True,
    )
    effort = tune.Effort(repetitions=args.repetitions, n_depth=args.context, label="quant")
    if warm_up:
        # The card ramps its clocks over the first minute, and the first variant
        # measured would otherwise carry that cost alone.
        measure(config, tune.Effort(repetitions=1, n_depth=args.context, label="warmup"))
    return measure(config, effort)


def _render_variant_speed(variant: quant.Variant, trial: tune.Trial, style: Style) -> None:
    label = _quant_label(variant.reference)
    bpw = f"{variant.bits_per_weight:.2f}" if variant.bits_per_weight else "-"
    size = f"{variant.info.file_bytes / 1024**3:.2f}G"
    if trial.score is None:
        print(f"  {label:<26}{bpw:>6}{size:>9}   {style.grey(trial.detail or 'did not run')}")
        return
    spread = f"{trial.spread:.1%}" if trial.spread is not None else "-"
    peak = f"{trial.peak_vram_mib} MiB" if trial.peak_vram_mib is not None else "-"
    print(f"  {label:<26}{bpw:>6}{size:>9}{trial.score:>9.2f} t/s{spread:>8}{peak:>12}")
    if not trial.reliable:
        print(style.grey(f"  {'':<26}unreliable: {trial.detail or 'the spread is too wide'}"))


def _run_quality(
    variants: tuple[quant.Variant, ...],
    args: argparse.Namespace,
    devices: tuple[str, ...],
    style: Style,
) -> dict[str, quant.Quality]:
    """Measure drift against the most precise local copy, if there is a corpus to use."""
    reference = variants[0]
    if not args.corpus:
        print(style.bold("\nquality"), style.dim("not measured"))
        print(
            style.dim(
                "  divergence needs a corpus. pass one with -f and it will be measured "
                f"against {_quant_label(reference.reference)}, the most precise copy here"
            )
        )
        return {}

    projected = quant.logits_bytes(reference.info.vocab_size, args.chunks, args.context)
    logits = Path(args.logits) if args.logits else Path(args.corpus).with_suffix(".logits")
    if projected:
        print(
            style.bold("\nquality"),
            style.dim(
                f"against {_quant_label(reference.reference)} over {args.chunks} chunks "
                f"of {Path(args.corpus).name}"
            ),
        )
        print(
            style.dim(
                f"  the reference logits file will be about {projected // 1024**2} MiB "
                f"({reference.info.vocab_size} vocabulary entries per corpus token)"
            )
        )
    failure = quant.write_reference_logits(
        reference=reference,
        corpus=args.corpus,
        logits_path=logits,
        chunks=args.chunks,
        context=args.context,
        n_gpu_layers=reference.info.block_count + 1,
        devices=devices,
        binary=args.perplexity,
    )
    if failure:
        print(style.red(f"  the reference pass failed: {failure}"))
        return {}

    print(f"  {'variant':<26}{'ppl ratio':>11}{'median KLD':>13}{'same top':>11}{'rms dp':>9}")
    out: dict[str, quant.Quality] = {}
    for variant in variants[1:]:
        quality = quant.measure_quality(
            variant=variant,
            reference=reference,
            corpus=args.corpus,
            logits_path=logits,
            chunks=args.chunks,
            context=args.context,
            n_gpu_layers=variant.info.block_count + 1,
            devices=devices,
            binary=args.perplexity,
        )
        out[variant.reference] = quality
        _render_quality(variant, quality, style)
    if not args.keep_logits:
        logits.unlink(missing_ok=True)
        print(style.dim("  the logits file was removed; keep it with --keep-logits"))
    return out


def _render_quality(variant: quant.Variant, quality: quant.Quality, style: Style) -> None:
    label = _quant_label(variant.reference)
    if not quality.measured:
        print(f"  {label:<26}{style.grey(quality.detail or 'nothing was reported')}")
        return
    ratio = f"{quality.ppl_ratio:.3f}" if quality.ppl_ratio is not None else "-"
    kld = f"{quality.median_kld:.4f}" if quality.median_kld is not None else "-"
    top = f"{quality.same_top_pct:.1f}%" if quality.same_top_pct is not None else "-"
    rms = f"{quality.rms_delta_p_pct:.1f}%" if quality.rms_delta_p_pct is not None else "-"
    print(f"  {label:<26}{ratio:>11}{kld:>13}{top:>11}{rms:>9}")


def _render_quant_decision(
    variants: tuple[quant.Variant, ...],
    speeds: dict[str, tune.Trial],
    qualities: dict[str, quant.Quality],
    style: Style,
) -> None:
    """State the trade rather than pick a side: quality tolerance is the user's call."""
    reference = variants[0]
    base_trial = speeds.get(reference.reference)
    for variant in variants[1:]:
        trial = speeds.get(variant.reference)
        if trial is None or trial.score is None or base_trial is None or base_trial.score is None:
            continue
        faster = trial.score / base_trial.score - 1
        smaller = reference.info.file_bytes - variant.info.file_bytes
        line = (
            f"  {_quant_label(variant.reference)} is {faster:+.0%} on decode and "
            f"{smaller / 1024**2:.0f} MiB smaller than {_quant_label(reference.reference)}"
        )
        print(line)
        quality = qualities.get(variant.reference)
        if quality is None or not quality.measured:
            print(style.dim("  what that costs in quality was not measured"))
            continue
        if quality.same_top_pct is not None:
            print(
                style.dim(
                    f"  it picks a different most-likely token {100 - quality.same_top_pct:.1f}% "
                    "of the time"
                )
            )
        if quality.ppl_ratio is not None:
            print(style.dim(f"  and its perplexity is {quality.ppl_ratio:.3f} times as high"))
    print(style.dim("  whether that trade is worth taking is not a measurement"))


def _quant_mapping(
    variants: tuple[quant.Variant, ...],
    speeds: dict[str, tune.Trial],
    qualities: dict[str, quant.Quality],
) -> dict[str, Any]:
    return {
        "reference": _quant_label(variants[0].reference) if variants else None,
        "variants": [
            {
                "variant": _quant_label(v.reference),
                "file_type": v.file_type,
                "bits_per_weight": v.bits_per_weight,
                "file_bytes": v.info.file_bytes,
                "decode_tok_s": (
                    speeds.get(v.reference).score if speeds.get(v.reference) else None
                ),
                "spread": (speeds.get(v.reference).spread if speeds.get(v.reference) else None),
                "peak_vram_mb": (
                    speeds.get(v.reference).peak_vram_mib if speeds.get(v.reference) else None
                ),
                "quality": (
                    {
                        "ppl_ratio": qualities[v.reference].ppl_ratio,
                        "median_kld": qualities[v.reference].median_kld,
                        "same_top_pct": qualities[v.reference].same_top_pct,
                        "rms_delta_p_pct": qualities[v.reference].rms_delta_p_pct,
                        "corpus": qualities[v.reference].corpus,
                        "chunks": qualities[v.reference].chunks,
                    }
                    if v.reference in qualities and qualities[v.reference].measured
                    else None
                ),
            }
            for v in variants
        ],
    }


def cmd_route(args: argparse.Namespace) -> int:
    """Say which measured model should answer, once the switch is paid for."""
    style = Style(color_enabled())
    snapshot = probe_hardware(include_wddm=False)
    card = snapshot.gpus[0].name if snapshot.gpus else None
    stored = [p for p in profiles.load_all() if p.signature.gpu == card]
    if not stored:
        print("no profiles measured on this card yet. `setpoint tune <model>` writes one.")
        return EXIT_OK

    candidates = []
    for profile in stored:
        cost = route.read_load(profile.signature.model_digest)
        if cost is None and args.measure:
            cost = _measure_switch(profile, style)
        candidates.append(
            route.Candidate(
                name=_route_label(profile),
                decode_tok_s=profile.measurement.decode_tok_s.median or 0.0,
                load_seconds=cost.median if cost else None,
                resident=_is_resident(profile, args.resident),
                context=profile.target.context,
            )
        )

    plan = route.plan(args.tokens, tuple(candidates))
    print(style.bold("request"))
    print(f"  {args.tokens} output tokens")
    print(
        style.dim("  decode only; prompt processing depends on a prompt this command was not given")
    )

    print(style.bold("\ncandidates"))
    print(f"  {'model':<{_ROUTE_LABEL_WIDTH}}{'decode':>10}{'switch':>10}{'total':>10}")
    for choice in sorted(plan.choices, key=lambda c: c.total_seconds):
        mark = "  <- loaded" if choice.candidate.resident else ""
        line = (
            f"  {choice.candidate.name:<{_ROUTE_LABEL_WIDTH}}"
            f"{choice.candidate.decode_tok_s:>7.2f} t/s"
            f"{choice.switch_seconds:>9.2f}s{choice.total_seconds:>9.2f}s"
        )
        print(f"{line}{style.dim(mark)}")
    for candidate in plan.unusable:
        why = "no switch cost measured; run with --measure"
        print(f"  {candidate.name:<{_ROUTE_LABEL_WIDTH}}{style.grey(why)}")

    best = plan.best
    if best is None:
        print(style.yellow("\nnothing can be costed yet"))
        return EXIT_OK

    print(style.bold("\ndecision"))
    print(f"  {best.candidate.name}   {best.total_seconds:.2f}s")
    saved = plan.saved_seconds
    if saved is not None and saved > 0:
        print(style.dim(f"  saves {saved:.2f}s against staying on the loaded model"))
    elif plan.resident is not None and plan.resident is best:
        print(style.dim("  already loaded, so nothing is paid for a switch"))
    if plan.switch_changed_the_answer and plan.fastest is not None:
        print(
            style.dim(
                f"  throughput alone would have picked {plan.fastest.candidate.name}, "
                f"which costs {plan.fastest.total_seconds:.2f}s once the switch is counted"
            )
        )

    if args.json:
        print(json.dumps(_route_mapping(plan), indent=2))
    return EXIT_OK


# Long enough for a model name and its context, short enough to keep the columns
# readable in a default terminal.
_ROUTE_LABEL_WIDTH = 26


def _route_label(profile: profiles.Profile) -> str:
    label = f"{profile.model.label} c{profile.target.context}"
    return label if len(label) <= _ROUTE_LABEL_WIDTH else label[: _ROUTE_LABEL_WIDTH - 1] + "~"


def _is_resident(profile: profiles.Profile, resident: str | None) -> bool:
    if not resident:
        return False
    needle = resident.lower()
    return needle in _route_label(profile).lower() or needle in (profile.model.label or "").lower()


def _measure_switch(profile: profiles.Profile, style: Style) -> route.LoadCost | None:
    """Time the load for one profile and keep it, so the next run does not pay again."""
    path = profile.model.path
    if not path:
        return None
    print(style.dim(f"  measuring the switch cost for {profile.model.label}"))
    cost = route.measure_load(
        model_path=path,
        config=profile.config,
        context=profile.target.context,
        model_digest=profile.signature.model_digest,
    )
    if cost is None:
        return None
    route.save_load(cost)
    return cost


def _route_mapping(plan: route.Plan) -> dict[str, Any]:
    return {
        "tokens": plan.tokens,
        "candidates": [
            {
                "model": choice.candidate.name,
                "decode_tok_s": choice.candidate.decode_tok_s,
                "switch_seconds": choice.switch_seconds,
                "total_seconds": choice.total_seconds,
                "resident": choice.candidate.resident,
            }
            for choice in plan.choices
        ],
        "uncosted": [c.name for c in plan.unusable],
        "decision": plan.best.candidate.name if plan.best else None,
        "saved_seconds": plan.saved_seconds,
        "switch_changed_the_answer": plan.switch_changed_the_answer,
    }


def cmd_spec(args: argparse.Namespace) -> int:
    """Measure whether speculative decoding pays here, and for which kind of work."""
    style = Style(color_enabled())
    profile, model, snapshot, code = _stored_profile(args, style)
    if profile is None:
        return code

    backend = LlamaCppBackend()
    try:
        listing = backend.devices()
    except BackendError as exc:
        print(f"setpoint: {exc}", file=sys.stderr)
        return EXIT_ERROR
    chosen = select_device(listing, args.device or profile.signature.gpu)
    if chosen is None or not _same_card(chosen.name, profile.signature.gpu):
        print(
            f"setpoint: this profile describes {profile.signature.gpu}, which the backend "
            "does not offer right now.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    speculators = tuple(args.speculators) if args.speculators else speculate.DRAFT_FREE
    print(style.bold("model"))
    print(f"  {profiles.signature_id(profile.signature)}   {profile.model.label}")
    print(f"  {style.dim('measuring on ' + chosen.id + ' -- ' + chosen.name)}")
    print(
        style.dim(
            f"  {len(speculators)} speculator(s), none of which needs a draft model, "
            "so none costs VRAM"
        )
    )

    try:
        report = speculate.measure(
            model_path=profile.model.path or model.path,
            config=profile.config,
            context=profile.target.context,
            devices=(chosen.id,),
            speculators=speculators,
            runs=args.repetitions,
            max_tokens=args.tokens,
        )
    except BackendError as exc:
        print(f"setpoint: {exc}", file=sys.stderr)
        return EXIT_ERROR

    for result in report.results:
        _render_workload(result, style)

    if args.json:
        print(json.dumps(_spec_mapping(report), indent=2))

    print(style.bold("\nverdict"))
    winners = report.winners
    if not any(winners.values()):
        print("  nothing here is worth turning on; the baseline stands")
        return EXIT_OK
    if report.agreed:
        print(f"  {report.agreed} won every workload measured")
    else:
        # The disagreement is the finding. Picking one would hide a factor of twenty.
        print("  it depends on the work, so there is no single setting for this machine:")
        for name, pick in winners.items():
            print(f"    {name:<8}{pick or 'nothing worth turning on'}")

    best_pair = max(
        (
            (result.gain(result.best) or 0, result.best)
            for result in report.results
            if result.best is not None
        ),
        default=(0.0, None),
    )
    best = best_pair[1]
    if best is not None:
        print(style.dim(f"\n  turn it on with: --spec-type {best.speculator}"))
        if args.write:
            updated = dataclasses.replace(
                profile, config=dataclasses.replace(profile.config, spec_type=best.speculator)
            )
            path = profiles.save(updated)
            print(f"  written to {short_path(path)}")
            print(style.dim("  `setpoint export` will now carry it into the runner config"))
        else:
            print(style.dim("  or store it in the profile with --write"))
    return EXIT_OK


def _render_workload(result: speculate.WorkloadResult, style: Style) -> None:
    print(style.bold(f"\n{result.workload.name}"), style.dim(result.workload.about))
    print(
        f"  {'speculator':<16}{'drafted':>9}{'accepted':>10}{'accept':>8}"
        f"{'tok/step':>10}{'t/s':>9}{'spread':>8}{'vs base':>10}"
    )
    for trial in result.trials:
        _render_trial(result, trial, style)
    base = result.baseline.throughput
    spread = f"{base.spread:.1%}" if base.spread is not None else "-"
    median = f"{base.median:.2f}" if base.median is not None else "-"
    print(f"  {'none':<16}{'-':>9}{'-':>10}{'-':>8}{1.00:>10.2f}{median:>9}{spread:>8}")


def _render_trial(result: speculate.WorkloadResult, trial: speculate.Trial, style: Style) -> None:
    if trial.throughput.median is None:
        print(f"  {trial.speculator:<16}   {style.grey(trial.detail or 'no measurement')}")
        return
    gain = result.gain(trial)
    accept = f"{trial.acceptance:.0%}" if trial.acceptance is not None else "-"
    per_step = f"{trial.tokens_per_step:.2f}" if trial.tokens_per_step is not None else "-"
    spread = f"{trial.throughput.spread:.1%}" if trial.throughput.spread is not None else "-"
    shown = f"{gain:+.1%}" if gain is not None else "-"
    painted = style.green if (gain or 0) > speculate.MIN_WORTH else style.dim
    line = (
        f"  {trial.speculator:<16}{trial.drafted:>9}{trial.accepted:>10}{accept:>8}"
        f"{per_step:>10}{trial.throughput.median:>9.2f}{spread:>8}"
    )
    print(f"{line}{painted(shown):>10}")
    if not trial.reliable and trial.throughput.runs:
        print(style.grey(f"  {'':<16}unreliable: the spread is above the 5% a profile may carry"))


def _spec_mapping(report: speculate.Report) -> dict[str, Any]:
    return {
        "model": report.model,
        "workloads": [
            {
                "name": result.workload.name,
                "about": result.workload.about,
                "baseline_tok_s": result.baseline.throughput.median,
                "best": result.best.speculator if result.best else None,
                "trials": [
                    {
                        "speculator": trial.speculator,
                        "drafted": trial.drafted,
                        "accepted": trial.accepted,
                        "acceptance": trial.acceptance,
                        "tokens_per_step": trial.tokens_per_step,
                        "tok_s": trial.throughput.median,
                        "spread": trial.throughput.spread,
                        "reliable": trial.reliable,
                        "gain": result.gain(trial),
                    }
                    for trial in result.trials
                ],
            }
            for result in report.results
        ],
        "winners": report.winners,
        "agreed": report.agreed,
    }


def _record_and_compare(
    profile: profiles.Profile, now: tune.Trial, style: Style
) -> sentinel.Comparison:
    """Add this reading to the history and say how it compares with the last one.

    The history is keyed by the model rather than by the whole signature: a driver
    update changes the signature, and a history that split on that could never show
    what the driver update cost.
    """
    sample = now.run.sample_of(MeasurementKind.DECODE) if now.run else None
    samples = tuple(sample.throughput.samples) if sample else ()
    path = sentinel.history_path(profile.signature.model_digest, profile.target.context)
    previous = sentinel.load(path)
    causes = sentinel.attribute(previous[-1].signature, profile.signature) if previous else ()

    record = sentinel.record_of(profile, samples, profiles.now(), peak_vram_mb=now.peak_vram_mib)
    comparison = (
        sentinel.compare(previous[-1], record, causes)
        if previous
        else sentinel.Comparison(
            sentinel.Verdict.UNKNOWN, "first check for this model; nothing to compare yet"
        )
    )
    try:
        sentinel.append(path, record)
    except OSError as exc:
        print(style.grey(f"  history not written: {exc}"))

    print()
    _row(style, "against last check", "", comparison.detail)
    _row(style, "history", f"{len(previous) + 1} checks", short_path(path))
    for cause in comparison.causes:
        print(style.yellow(f"  changed since: {cause}"))
    return comparison


def cmd_export(args: argparse.Namespace) -> int:
    """Write the runner configuration the measured profiles imply."""
    style = Style(color_enabled())
    snapshot = probe_hardware(include_wddm=False)
    facts = profiles.machine_facts(snapshot, args.gpu)
    if facts is None:
        print(
            "setpoint: this machine's GPU could not be identified, so no profile can be "
            "matched to it.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    # Only profiles measured here may be written: a configuration built from someone
    # else's measurement would start a model with settings nothing verified.
    stored = profiles.load_all()
    mine = [p for p in stored if profiles.describes_machine(p.signature, facts)]

    server = find_server_binary() or SERVER_BINARY_NAME
    devices: tuple[str, ...] = ()
    if args.device:
        devices = (args.device,)
    else:
        try:
            listing = LlamaCppBackend().devices()
        except BackendError:
            listing = ()
        chosen = select_device(listing, facts.gpu) if listing else None
        if chosen is not None and len(listing) > 1:
            devices = (chosen.id,)

    entries = export.build_entries(
        mine,
        server_binary=server,
        vram_total_mb=facts.vram_total_mb,
        ttl_override=args.ttl,
        devices=devices,
    )
    header = [
        "Generated by setpoint. Every entry below was measured on this machine.",
        f"{facts.gpu}, {facts.vram_total_mb} MiB, driver {facts.driver}, {facts.platform}",
        "Regenerate after `setpoint tune`; verify with `setpoint bench`.",
    ]
    if devices:
        header.append(f"Pinned to {'/'.join(devices)} because more than one is offered.")
    text = export.render(entries, header)

    if args.out:
        try:
            Path(args.out).write_text(text, encoding="utf-8")
        except OSError as exc:
            print(f"setpoint: cannot write {args.out}: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(f"wrote {len(entries)} entry(s) to {short_path(args.out)}")
    else:
        print(text, end="")

    if not entries:
        skipped = len(stored) - len(mine)
        message = "no profiles match this machine yet. Run `setpoint tune <model>`."
        if skipped:
            message += f" {skipped} stored profile(s) describe other hardware."
        print(style.yellow(f"setpoint: {message}"), file=sys.stderr)
        return EXIT_PROBLEM
    return EXIT_OK


_VERDICT_STYLE = {
    monitor.Verdict.SPILLING: ("red", "SPILLING"),
    monitor.Verdict.HEALTHY: ("green", "ok"),
    monitor.Verdict.IDLE: ("grey", "idle"),
    monitor.Verdict.UNKNOWN: ("grey", "--"),
}


def _bar(fraction: float | None, width: int = 28) -> str:
    if fraction is None:
        return "?" * width
    filled = max(0, min(width, round(fraction * width)))
    return "#" * filled + "." * (width - filled)


def _render_reading(
    reading: monitor.Reading, spill: monitor.SpillCheck, notes: list[str], style: Style
) -> list[str]:
    lines: list[str] = []
    used, total = reading.vram_used_bytes, reading.vram_total_bytes
    share = reading.dedicated_share

    lines.append(style.bold("dedicated"))
    value = f"{(used or 0) / budget.MIB:.0f} / {(total or 0) / budget.MIB:.0f} MiB"
    lines.append(f"  [{_bar(share)}]  {value}")

    lines.append("")
    lines.append(style.bold("shared"))
    if reading.shared_bytes is None:
        lines.append(f"  {style.grey('not readable on this machine')}")
    else:
        age = reading.shared_age_s or 0.0
        tail = f"{age:.0f}s old" if age >= 1 else "now"
        if not reading.shared_confident:
            tail += ", adapter matched by guess"
        lines.append(f"  {reading.shared_bytes / budget.MIB:>8.0f} MiB   {style.dim(tail)}")

    bits = []
    if reading.utilization_pct is not None:
        bits.append(f"{reading.utilization_pct}% busy")
    if reading.temperature_c is not None:
        bits.append(f"{reading.temperature_c} C")
    if reading.power_w is not None:
        bits.append(f"{reading.power_w:.0f} W")
    if bits:
        lines.append("")
        lines.append(f"  {style.dim('  '.join(bits))}")
    if reading.throttle_reasons:
        lines.append(f"  {style.yellow('throttle: ' + ', '.join(reading.throttle_reasons))}")

    colour, label = _VERDICT_STYLE[spill.verdict]
    lines.append("")
    lines.append(f"{getattr(style, colour)(label)}  {spill.detail}")
    for note in notes:
        lines.append(style.grey(f"  note: {note}"))
    return lines


def cmd_status(args: argparse.Namespace) -> int:
    """One reading, for a script or a quick look."""
    style = Style(color_enabled())
    with monitor.Monitor(args.gpu) as watch:
        # A spill is a trend, not a value, so even one look samples for a moment. The
        # shared counter is read on its own thread and takes about two seconds to answer.
        deadline = time.monotonic() + args.wait
        reading = watch.tick()
        while time.monotonic() < deadline:
            time.sleep(0.5)
            reading = watch.tick()
        spill = watch.spill
        notes = list(watch.notes)
        if spill.verdict is monitor.Verdict.UNKNOWN:
            notes.append("a spill shows as a trend; `setpoint top` watches for one over time")

    if args.json:
        print(
            json.dumps(
                {
                    "vram_used_bytes": reading.vram_used_bytes,
                    "vram_total_bytes": reading.vram_total_bytes,
                    "dedicated_share": reading.dedicated_share,
                    "shared_bytes": reading.shared_bytes,
                    "shared_age_s": reading.shared_age_s,
                    "shared_confident": reading.shared_confident,
                    "utilization_pct": reading.utilization_pct,
                    "temperature_c": reading.temperature_c,
                    "power_w": reading.power_w,
                    "throttle_reasons": list(reading.throttle_reasons),
                    "verdict": spill.verdict.value,
                    "detail": spill.detail,
                    "notes": notes,
                },
                indent=2,
            )
        )
    else:
        for line in _render_reading(reading, spill, notes, style):
            print(line)
    return EXIT_PROBLEM if spill.verdict is monitor.Verdict.SPILLING else EXIT_OK


def cmd_top(args: argparse.Namespace) -> int:
    """Redraw until interrupted."""
    style = Style(color_enabled())
    spilled = False
    try:
        with monitor.Monitor(args.gpu) as watch:
            while True:
                reading = watch.tick()
                spill = watch.spill
                spilled = spilled or spill.verdict is monitor.Verdict.SPILLING
                body = _render_reading(reading, spill, list(watch.notes), style)
                header = style.dim(f"setpoint top -- every {args.interval:.1f}s, Ctrl-C to stop")
                print("\033[H\033[J" + header + "\n", end="")
                print("\n".join(body), flush=True)
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        return EXIT_PROBLEM if spilled else EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="setpoint",
        description="Measurement-driven configuration for local LLM inference.",
    )
    parser.add_argument("--version", action="version", version=f"setpoint {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="scan for hardware and configuration traps")
    p_doctor.add_argument("--json", action="store_true", help="emit machine-readable output")
    p_doctor.set_defaults(func=cmd_doctor)

    p_budget = sub.add_parser(
        "budget",
        help="estimate what a model needs and what fits on the GPU",
        description=(
            "Estimate weights, KV cache and the resulting offload split without running "
            "anything. Exits 0 when the request fits entirely on the GPU, 1 when part of "
            "it has to stay on the CPU."
        ),
    )
    p_budget.add_argument("model", help="path to a .gguf file, or an Ollama model name")
    p_budget.add_argument(
        "-c", "--context", type=int, default=4096, help="target context length in tokens"
    )
    p_budget.add_argument(
        "--kv-type",
        default="f16",
        choices=sorted(budget.CACHE_TYPES),
        help="KV cache quantization",
    )
    p_budget.add_argument("--gpu", type=int, help="GPU index, when the machine has more than one")
    p_budget.add_argument(
        "--reserve", type=int, default=0, metavar="MB", help="VRAM to keep free for yourself"
    )
    p_budget.add_argument(
        "--ubatch",
        type=int,
        default=budget.DEFAULT_UBATCH,
        metavar="N",
        help="microbatch the plan assumes; it sets the size of the compute buffers",
    )
    p_budget.add_argument(
        "--overhead",
        type=int,
        metavar="MB",
        help="override the calculated allowance for the buffers of the process that "
        "does not exist yet",
    )
    p_budget.add_argument(
        "--fragmentation",
        type=float,
        default=budget.DEFAULT_FRAGMENTATION_PCT,
        metavar="PCT",
        help="share of free VRAM held back for allocator fragmentation",
    )
    p_budget.add_argument(
        "--vram",
        type=int,
        metavar="MB",
        help="budget against a card of this size instead of the installed one",
    )
    p_budget.add_argument("--json", action="store_true", help="emit machine-readable output")
    p_budget.set_defaults(func=cmd_budget)

    p_tune = sub.add_parser(
        "tune",
        help="measure configurations and write the best one as a profile",
        description=(
            "Seeds a search from the budget, screens the candidates cheaply, then walks "
            "one parameter at a time. Interrupting with Ctrl-C keeps the best result "
            "measured so far. A result whose spread is too wide is not written."
        ),
    )
    p_tune.add_argument("model", help="path to a .gguf file, or an Ollama model name")
    p_tune.add_argument(
        "-c", "--context", type=int, default=4096, help="target context length in tokens"
    )
    p_tune.add_argument(
        "--optimize",
        default="speed",
        choices=[o.value for o in profiles.Objective],
        help="what to maximise",
    )
    p_tune.add_argument(
        "--device",
        metavar="NAME",
        help="accelerator to measure on, by name or by the id the backend currently "
        "gives it; defaults to the GPU being budgeted for",
    )
    p_tune.add_argument("--gpu", type=int, help="GPU index, when the machine has more than one")
    p_tune.add_argument(
        "--kv-type", default="f16", choices=sorted(budget.CACHE_TYPES), help="KV cache quantization"
    )
    p_tune.add_argument(
        "--ubatch", type=int, default=budget.DEFAULT_UBATCH, metavar="N", help="microbatch size"
    )
    p_tune.add_argument(
        "--reserve", type=int, default=0, metavar="MB", help="VRAM to keep free for yourself"
    )
    p_tune.add_argument(
        "--ncmoe",
        type=int,
        metavar="N",
        help="MoE models only: start the search with the experts of N blocks on the CPU. "
        "Off unless asked for, because it measured slower than simply keeping fewer "
        "blocks on the GPU",
    )
    p_tune.add_argument(
        "--repetitions", type=int, default=5, metavar="N", help="runs per final measurement"
    )
    p_tune.add_argument(
        "--screen-context",
        type=int,
        default=1024,
        metavar="N",
        help="context used while screening candidates, which keeps early rounds cheap",
    )
    p_tune.add_argument(
        "--budget",
        type=int,
        default=tune.DEFAULT_MEASUREMENT_BUDGET,
        metavar="N",
        help="stop after this many measurements and report the best so far",
    )
    p_tune.add_argument("--dry-run", action="store_true", help="measure but write nothing")
    p_tune.add_argument("--json", action="store_true", help="emit machine-readable output")
    p_tune.set_defaults(func=cmd_tune)

    p_run = sub.add_parser(
        "run",
        help="start llama-server with the stored profile",
        description=(
            "Looks up the profile measured for this model on this machine and starts "
            "llama-server with it. Anything after -- is passed straight through."
        ),
    )
    p_run.add_argument("model", help="path to a .gguf file, or an Ollama model name")
    p_run.add_argument(
        "-c", "--context", type=int, help="pick the profile measured for this context"
    )
    p_run.add_argument("--device", metavar="NAME", help="accelerator to run on")
    p_run.add_argument(
        "--print-only", action="store_true", help="show the command without running it"
    )
    p_run.set_defaults(func=cmd_run)

    p_bench = sub.add_parser(
        "bench",
        help="re-measure a stored profile and say whether it still holds",
        description=(
            "Measures the stored configuration again and compares it against what the "
            "profile claims. Exits 1 when the machine no longer matches."
        ),
    )
    p_bench.add_argument("model", help="path to a .gguf file, or an Ollama model name")
    p_bench.add_argument(
        "-c", "--context", type=int, help="pick the profile measured for this context"
    )
    p_bench.add_argument("--device", metavar="NAME", help="accelerator to measure on")
    p_bench.add_argument(
        "--repetitions", type=int, default=5, metavar="N", help="runs per measurement"
    )
    p_bench.add_argument("--json", action="store_true", help="emit machine-readable output")
    p_bench.set_defaults(func=cmd_bench)

    p_quant = sub.add_parser(
        "quant",
        help="compare the local quantizations of one model",
        description=(
            "Measures speed and VRAM for every quantization of this model on this "
            "machine. Quality is measured as divergence from the most precise local "
            "copy, and only when a corpus is given: without one it is reported as not "
            "measured rather than guessed."
        ),
    )
    p_quant.add_argument("model", help="path to a .gguf file, or an Ollama model name")
    p_quant.add_argument(
        "-c", "--context", type=int, default=512, metavar="N", help="context for both passes"
    )
    p_quant.add_argument(
        "--ubatch", type=int, default=128, metavar="N", help="microbatch for the speed pass"
    )
    p_quant.add_argument(
        "--repetitions", type=int, default=3, metavar="N", help="runs per speed measurement"
    )
    p_quant.add_argument("--device", metavar="NAME", help="accelerator to measure on")
    p_quant.add_argument(
        "-f", "--corpus", metavar="PATH", help="text file to measure divergence over"
    )
    p_quant.add_argument(
        "--chunks",
        type=int,
        default=quant.DEFAULT_CHUNKS,
        metavar="N",
        help="corpus chunks; the reference logits file grows with this",
    )
    p_quant.add_argument("--logits", metavar="PATH", help="where to put the reference logits")
    p_quant.add_argument(
        "--keep-logits", action="store_true", help="do not delete the reference logits"
    )
    p_quant.add_argument(
        "--perplexity",
        default=quant.PERPLEXITY_BINARY,
        metavar="PATH",
        help="llama-perplexity binary, if it is not on PATH",
    )
    p_quant.add_argument("--json", action="store_true", help="emit machine-readable output")
    p_quant.set_defaults(func=cmd_quant)

    p_route = sub.add_parser(
        "route",
        help="say which measured model should answer, once the switch is paid for",
        description=(
            "Costs every profile measured on this card for a request of a given length, "
            "counting what it takes to load a model that is not already up. Prints the "
            "decision and its arithmetic; it does not serve requests."
        ),
    )
    p_route.add_argument(
        "--tokens", type=int, default=200, metavar="N", help="output tokens to cost for"
    )
    p_route.add_argument(
        "--resident", metavar="MODEL", help="which model is already loaded, if any"
    )
    p_route.add_argument(
        "--measure",
        action="store_true",
        help="measure and store the switch cost for any model that has none",
    )
    p_route.add_argument("--json", action="store_true", help="emit machine-readable output")
    p_route.set_defaults(func=cmd_route)

    p_spec = sub.add_parser(
        "spec",
        help="measure whether speculative decoding pays on this machine",
        description=(
            "Speculation only wins when the target model accepts enough of each draft. "
            "How much it accepts depends on the kind of work, so every speculator is "
            "measured on more than one workload and the answer may differ between them."
        ),
    )
    p_spec.add_argument("model", help="path to a .gguf file, or an Ollama model name")
    p_spec.add_argument(
        "-c", "--context", type=int, help="pick the profile measured for this context"
    )
    p_spec.add_argument("--device", metavar="NAME", help="accelerator to measure on")
    p_spec.add_argument(
        "--speculators",
        nargs="+",
        metavar="NAME",
        help="which to try; the default is every draft-free one",
    )
    p_spec.add_argument(
        "--repetitions", type=int, default=3, metavar="N", help="requests per measurement"
    )
    p_spec.add_argument(
        "--tokens",
        type=int,
        default=speculate.DEFAULT_MAX_TOKENS,
        metavar="N",
        help="tokens generated per request",
    )
    p_spec.add_argument("--write", action="store_true", help="store the winner in the profile")
    p_spec.add_argument("--json", action="store_true", help="emit machine-readable output")
    p_spec.set_defaults(func=cmd_spec)

    p_export = sub.add_parser(
        "export",
        help="write the runner configuration the measured profiles imply",
        description=(
            "setpoint does not serve requests; a runner does. This writes that runner's "
            "configuration from the profiles measured on this machine, with the "
            "measurement behind each entry kept as a comment. Profiles measured "
            "elsewhere are left out."
        ),
    )
    p_export.add_argument(
        "--target",
        default="llama-swap",
        choices=("llama-swap",),
        help="which runner the configuration is for",
    )
    p_export.add_argument("--out", metavar="PATH", help="write here instead of stdout")
    p_export.add_argument(
        "--ttl",
        type=int,
        metavar="SECONDS",
        help="idle unload timeout for every entry; the default is chosen from how much "
        "of the card each model holds",
    )
    p_export.add_argument("--gpu", type=int, help="GPU index, when the machine has more than one")
    p_export.add_argument("--device", metavar="NAME", help="accelerator to pin entries to")
    p_export.set_defaults(func=cmd_export)

    p_top = sub.add_parser(
        "top",
        help="watch the GPU live, and catch a spill into system RAM",
        description=(
            "Redraws until interrupted. Dedicated memory comes from the driver and is "
            "current; the shared figure comes from the operating system's adapter "
            "counters, costs about two seconds to read, and its age is shown. Exits 1 "
            "if a spill was seen."
        ),
    )
    p_top.add_argument(
        "--interval", type=float, default=1.0, metavar="S", help="seconds between redraws"
    )
    p_top.add_argument("--gpu", type=int, help="GPU index, when the machine has more than one")
    p_top.set_defaults(func=cmd_top)

    p_status = sub.add_parser(
        "status",
        help="one reading of what the GPU is doing right now",
        description="Exits 1 if the readings show a spill into system RAM.",
    )
    p_status.add_argument(
        "--wait",
        type=float,
        default=3.0,
        metavar="S",
        help="how long to wait for the shared-memory counter, which is slow to read",
    )
    p_status.add_argument("--gpu", type=int, help="GPU index, when the machine has more than one")
    p_status.add_argument("--json", action="store_true", help="emit machine-readable output")
    p_status.set_defaults(func=cmd_status)

    p_profile = sub.add_parser(
        "profile",
        help="inspect stored calibration profiles",
        description=(
            "Profiles are written by `setpoint tune` and keyed by a hardware and model "
            "signature. A profile whose measurement is not reliable is never stored."
        ),
    )
    p_profile.add_argument(
        "action",
        choices=("list", "show", "path", "export", "import"),
        nargs="?",
        default="list",
    )
    p_profile.add_argument("id", nargs="?", help="profile id or prefix; a file path for `import`")
    p_profile.add_argument("--out", metavar="PATH", help="write here instead of stdout, for export")
    p_profile.add_argument("--gpu", type=int, help="GPU index, when the machine has more than one")
    p_profile.add_argument("--json", action="store_true", help="emit machine-readable output")
    p_profile.set_defaults(func=cmd_profile)

    p_hardware = sub.add_parser("hardware", help="show the raw hardware snapshot")
    p_hardware.add_argument("--json", action="store_true", help="emit machine-readable output")
    p_hardware.add_argument(
        "--no-wddm",
        action="store_true",
        help="skip the Windows adapter memory counters, which take about a second",
    )
    p_hardware.set_defaults(func=cmd_hardware)

    return parser


def split_forwarded(argv: list[str]) -> tuple[list[str], tuple[str, ...]]:
    """Split on the first bare `--`. Everything after it belongs to the backend.

    argparse's REMAINDER would swallow setpoint's own flags along with the rest, so the
    boundary is drawn before parsing instead of during it.
    """
    if "--" not in argv:
        return argv, ()
    cut = argv.index("--")
    return argv[:cut], tuple(argv[cut + 1 :])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    own, forwarded = split_forwarded(list(argv if argv is not None else sys.argv[1:]))
    args = parser.parse_args(own)
    if forwarded:
        if getattr(args, "func", None) is not cmd_run:
            print("setpoint: only `run` forwards arguments after --", file=sys.stderr)
            return EXIT_ERROR
        args.forward = forwarded
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nsetpoint: interrupted", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
