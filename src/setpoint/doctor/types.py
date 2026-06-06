"""Doctor result types.

A finding answers three questions in order: what was observed, why it costs
performance, and what to do about it.

A check that cannot reach a conclusion returns SKIP rather than PASS. Not being able
to look and having looked are different claims, and only one of them clears the
machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class Outcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


_SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
_OUTCOME_ORDER = {Outcome.FAIL: 0, Outcome.SKIP: 1, Outcome.PASS: 2}


@dataclass(frozen=True)
class Finding:
    check_id: str
    title: str
    outcome: Outcome
    severity: Severity
    what: str
    why: str | None = None
    fix: str | None = None
    evidence: dict[str, object] = field(default_factory=dict)

    @property
    def sort_key(self) -> tuple[int, int, str]:
        return (
            _OUTCOME_ORDER[self.outcome],
            _SEVERITY_ORDER[self.severity],
            self.check_id,
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "check": self.check_id,
            "title": self.title,
            "outcome": self.outcome.value,
            "severity": self.severity.value,
            "what": self.what,
        }
        if self.why:
            payload["why"] = self.why
        if self.fix:
            payload["fix"] = self.fix
        if self.evidence:
            payload["evidence"] = self.evidence
        return payload


@dataclass(frozen=True)
class Report:
    findings: tuple[Finding, ...]

    @property
    def failures(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.outcome is Outcome.FAIL)

    @property
    def skipped(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.outcome is Outcome.SKIP)

    @property
    def exit_code(self) -> int:
        """0 healthy, 1 performance problem found, 2 could not run.

        A skipped check does not raise the exit code: not being able to inspect the
        PCIe link is not the same as finding a broken one.
        """
        if any(
            f.severity is Severity.CRITICAL and f.outcome is Outcome.FAIL for f in self.findings
        ):
            return 1
        if any(f.severity is Severity.WARNING and f.outcome is Outcome.FAIL for f in self.findings):
            return 1
        return 0

    def to_dict(self) -> dict[str, object]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "summary": {
                "total": len(self.findings),
                "failed": len(self.failures),
                "skipped": len(self.skipped),
                "exit_code": self.exit_code,
            },
        }
