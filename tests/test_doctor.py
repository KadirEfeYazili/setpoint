"""Doctor reporting tests.

The exit code and ordering rules are part of the CLI contract, so they are pinned here.
"""

from __future__ import annotations

from setpoint.doctor.types import Finding, Outcome, Report, Severity


def finding(check_id: str, outcome: Outcome, severity: Severity) -> Finding:
    return Finding(
        check_id=check_id,
        title=check_id,
        outcome=outcome,
        severity=severity,
        what="observed",
    )


class TestExitCode:
    def test_all_passing_is_healthy(self):
        report = Report((finding("a", Outcome.PASS, Severity.INFO),))
        assert report.exit_code == 0

    def test_critical_failure_reports_a_problem(self):
        report = Report((finding("a", Outcome.FAIL, Severity.CRITICAL),))
        assert report.exit_code == 1

    def test_warning_failure_reports_a_problem(self):
        report = Report((finding("a", Outcome.FAIL, Severity.WARNING),))
        assert report.exit_code == 1

    def test_skipped_check_does_not_report_a_problem(self):
        # Not being able to inspect something is not the same as finding it broken.
        report = Report(
            (
                finding("a", Outcome.SKIP, Severity.WARNING),
                finding("b", Outcome.PASS, Severity.INFO),
            )
        )
        assert report.exit_code == 0

    def test_informational_failure_does_not_report_a_problem(self):
        report = Report((finding("a", Outcome.FAIL, Severity.INFO),))
        assert report.exit_code == 0


class TestOrdering:
    def test_failures_come_before_skips_and_passes(self):
        findings = [
            finding("pass", Outcome.PASS, Severity.INFO),
            finding("skip", Outcome.SKIP, Severity.INFO),
            finding("fail", Outcome.FAIL, Severity.WARNING),
        ]
        findings.sort(key=lambda f: f.sort_key)
        assert [f.check_id for f in findings] == ["fail", "skip", "pass"]

    def test_critical_comes_before_warning(self):
        findings = [
            finding("warn", Outcome.FAIL, Severity.WARNING),
            finding("crit", Outcome.FAIL, Severity.CRITICAL),
        ]
        findings.sort(key=lambda f: f.sort_key)
        assert [f.check_id for f in findings] == ["crit", "warn"]


class TestSerialisation:
    def test_optional_fields_are_omitted_when_absent(self):
        payload = finding("a", Outcome.PASS, Severity.INFO).to_dict()
        assert "why" not in payload
        assert "fix" not in payload
        assert "evidence" not in payload

    def test_summary_counts_match_the_findings(self):
        report = Report(
            (
                finding("a", Outcome.FAIL, Severity.CRITICAL),
                finding("b", Outcome.SKIP, Severity.INFO),
                finding("c", Outcome.PASS, Severity.INFO),
            )
        )
        summary = report.to_dict()["summary"]
        assert summary == {"total": 3, "failed": 1, "skipped": 1, "exit_code": 1}
