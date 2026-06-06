"""Doctor: scan for the hardware and configuration traps that silently cost throughput."""

from __future__ import annotations

from ..hardware import HardwareSnapshot
from ..hardware import probe as probe_hardware
from .checks import ALL_CHECKS
from .types import Finding, Outcome, Report, Severity

__all__ = ["Finding", "Outcome", "Report", "Severity", "run"]


def run(snapshot: HardwareSnapshot | None = None) -> Report:
    """Run every check against a hardware snapshot.

    A check that raises becomes a skipped finding rather than taking the run down, so
    one broken probe does not cost the user the remaining answers.
    """
    snap = snapshot if snapshot is not None else probe_hardware()

    findings: list[Finding] = []
    for check in ALL_CHECKS:
        try:
            findings.extend(check(snap))
        except Exception as exc:  # a check bug must not break the report
            findings.append(
                Finding(
                    check_id=getattr(check, "__name__", "unknown"),
                    title="Check failed to run",
                    outcome=Outcome.SKIP,
                    severity=Severity.INFO,
                    what=f"This check raised an error and was skipped: {exc!r}",
                )
            )

    findings.sort(key=lambda f: f.sort_key)
    return Report(findings=tuple(findings))
