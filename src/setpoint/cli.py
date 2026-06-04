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

from . import __version__, doctor
from .hardware import probe as probe_hardware
from .render import Style, color_enabled, term_width, wrap

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
