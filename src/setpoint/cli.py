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

from . import __version__, budget, doctor, export, monitor, tune
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
from .model import ModelError, analyze, resolve
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

    left_behind = (
        "the token embedding, which llama.cpp keeps in RAM"
        if offload.fits_fully
        else f"{offload.cpu_weight_fraction:.0%} of the model"
    )
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


def cmd_profile(args: argparse.Namespace) -> int:
    directory = profiles.profiles_dir()
    style = Style(color_enabled())

    if args.action == "path":
        print(directory)
        return EXIT_OK

    try:
        stored = profiles.load_all(directory)
    except OSError as exc:
        print(f"setpoint: cannot read {directory}: {exc}", file=sys.stderr)
        return EXIT_ERROR

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

    seeds = [
        profiles.Config(
            n_gpu_layers=candidate.n_gpu_layers,
            cache_type_k=candidate.cache_type_k,
            cache_type_v=candidate.cache_type_v,
            flash_attn=candidate.flash_attn,
            ubatch_size=args.ubatch,
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
    return EXIT_OK


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
                },
                indent=2,
            )
        )

    if now.detail:
        print()
        for line in wrap(f"note: {now.detail}", term_width() - 2):
            print(style.grey(f"  {line}"))

    if not within:
        print(
            style.red(
                f"\nthe profile no longer describes this machine: {drift:+.1%} against a "
                f"{REGRESSION_TOLERANCE:.0%} tolerance. Re-run `setpoint tune`."
            )
        )
        return EXIT_PROBLEM
    print(style.green("\nthe profile still holds"))
    return EXIT_OK


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
    p_profile.add_argument("action", choices=("list", "show", "path"), nargs="?", default="list")
    p_profile.add_argument("id", nargs="?", help="profile id, or any unambiguous prefix")
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
