"""Command line entry point.

Exit codes are part of the contract: 0 healthy, 1 a problem was found, 2 setpoint
could not complete the check. Data goes to stdout, diagnostics to stderr, so every
command composes with jq and shell pipelines.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import Any

from . import __version__, budget, doctor
from . import profile as profiles
from .hardware import probe as probe_hardware
from .model import ModelError, analyze, resolve
from .render import Style, color_enabled, gib, human_bytes, short_path, term_width, wrap

EXIT_OK = 0
EXIT_PROBLEM = 1
EXIT_ERROR = 2

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


def _render_budget(plan: budget.BudgetPlan, reference: str, style: Style) -> None:
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
    _row(style, "free", gib(vram.free_bytes), "measured" if vram.measured else "assumed")
    _row(style, "fragmentation", "-" + gib(vram.fragmentation_bytes))
    if vram.reserve_bytes:
        _row(style, "your reserve", "-" + gib(vram.reserve_bytes))
    _row(
        style,
        "runtime allowance",
        "-" + gib(vram.runtime_allowance_bytes),
        "estimate; `setpoint tune` measures it",
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

    allowance = (
        budget.DEFAULT_RUNTIME_ALLOWANCE_BYTES
        if args.overhead is None
        else args.overhead * budget.MIB
    )
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
        ram = snapshot.host.total_ram_bytes
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

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        _render_budget(plan, resolved.reference, Style(color_enabled()))
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
    _row(style, "gpu", signature.gpu.split()[-1], signature.gpu)
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
    for stored_profile in stored:
        speedup = f"{stored_profile.baseline.speedup:.2f}x"
        print(
            f"  {profiles.signature_id(stored_profile.signature):<18}"
            f"{stored_profile.model.label[:29]:<30}"
            f"{stored_profile.target.context:>7}"
            f"{stored_profile.config.n_gpu_layers if stored_profile.config.n_gpu_layers else '-':>6}"
            f"{stored_profile.measurement.decode_tok_s.median:>10.2f} t/s"
            f"{speedup:>10}"
        )
    return EXIT_OK


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
        "--overhead",
        type=int,
        metavar="MB",
        help="allowance for the driver context and compute buffers of the process that "
        "does not exist yet (default 192)",
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

    p_profile = sub.add_parser(
        "profile",
        help="inspect stored calibration profiles",
        description=(
            "Profiles are written by `setpoint tune` and keyed by a hardware and model "
            "signature. A profile whose measurement is not reliable is never stored."
        ),
    )
    p_profile.add_argument(
        "action", choices=("list", "show", "path"), nargs="?", default="list"
    )
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nsetpoint: interrupted", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
