"""Regression sentinel tests.

The point of the rank-sum test over a fixed percentage is that two changes of the same
size are not the same news: one is a regression, the other is a busy afternoon. Those
two cases are the first tests here, because if they were not distinguishable there
would be no reason to have written the test at all.
"""

from __future__ import annotations

import json

import pytest

from setpoint.measure import rank_sum_p
from setpoint.profile import Signature
from setpoint.sentinel import (
    ALPHA,
    MIN_MATERIAL_CHANGE,
    Comparison,
    Record,
    Verdict,
    append,
    attribute,
    compare,
    history_path,
    load,
)

TIGHT_BEFORE = (10.0, 10.1, 9.9, 10.2, 9.8)
TIGHT_AFTER = (9.5, 9.6, 9.4, 9.7, 9.3)
NOISY_BEFORE = (10.0, 12.0, 8.0, 11.0, 9.0)
NOISY_AFTER = (9.0, 11.5, 8.5, 10.5, 9.5)

MACHINE = {
    "gpu": "NVIDIA GeForce GTX 1650",
    "vram_total_mb": 4096,
    "driver": "512.89",
    "backend": "llama.cpp b10850",
    "platform": "windows/amd64",
}


def record(samples: tuple[float, ...], at: str = "2026-09-09T00:00:00Z", **overrides) -> Record:
    signature = dict(MACHINE)
    signature.update(overrides.pop("signature", {}))
    return Record(at=at, samples=samples, signature=signature, context=1024, **overrides)


def signature(**overrides) -> Signature:
    fields = {
        "model_digest": "sha256:" + "a" * 64,
        "model_digest_kind": "header",
        "model_size_bytes": 1,
        **MACHINE,
    }
    fields.update(overrides)
    return Signature(**fields)


class TestWhyNotAThreshold:
    """Both of these are a five percent drop. Only one of them is news."""

    def test_a_consistent_drop_is_caught(self):
        check = compare(record(TIGHT_BEFORE), record(TIGHT_AFTER))
        assert check.verdict is Verdict.SLOWER
        assert check.actionable
        assert check.p_value < ALPHA

    def test_the_same_drop_inside_noise_is_not(self):
        check = compare(record(NOISY_BEFORE), record(NOISY_AFTER))
        assert check.verdict is Verdict.SAME
        assert not check.actionable
        assert check.p_value > ALPHA

    def test_both_report_the_same_change(self):
        caught = compare(record(TIGHT_BEFORE), record(TIGHT_AFTER))
        missed = compare(record(NOISY_BEFORE), record(NOISY_AFTER))
        assert caught.change == pytest.approx(missed.change, abs=0.01)


class TestVerdicts:
    def test_getting_faster_is_reported_but_not_actionable(self):
        check = compare(record(TIGHT_AFTER), record(TIGHT_BEFORE))
        assert check.verdict is Verdict.FASTER
        assert not check.actionable

    def test_a_real_but_tiny_change_is_not_worth_acting_on(self):
        # Significant and trivial is a thing. Measurements this tight separate cleanly
        # on a one percent shift, so the test fires; acting on it would be noise.
        before = (10.00, 10.01, 10.02, 10.03, 10.04)
        after = (9.90, 9.91, 9.92, 9.93, 9.94)
        check = compare(record(before), record(after))
        assert check.p_value < ALPHA
        assert abs(check.change) < MIN_MATERIAL_CHANGE
        assert check.verdict is Verdict.SAME
        assert "too small to act on" in check.detail

    def test_identical_readings_cannot_be_tested(self):
        same = (10.0, 10.0, 10.0)
        assert compare(record(same), record(same)).verdict is Verdict.UNKNOWN

    def test_nothing_to_compare_against_says_so(self):
        assert compare(record(()), record(TIGHT_AFTER)).verdict is Verdict.UNKNOWN


class TestRankSum:
    def test_it_is_symmetric(self):
        assert rank_sum_p(TIGHT_BEFORE, TIGHT_AFTER) == pytest.approx(
            rank_sum_p(TIGHT_AFTER, TIGHT_BEFORE)
        )

    def test_identical_sets_cannot_be_told_apart(self):
        assert rank_sum_p((1.0, 1.0), (1.0, 1.0)) is None

    def test_an_empty_side_has_no_answer(self):
        assert rank_sum_p((), (1.0, 2.0)) is None

    def test_complete_separation_is_the_smallest_p_the_counts_allow(self):
        # Five against five: 2/252 splits are at least this extreme.
        p = rank_sum_p((1.0, 2.0, 3.0, 4.0, 5.0), (6.0, 7.0, 8.0, 9.0, 10.0))
        assert p == pytest.approx(2 / 252)

    def test_a_p_value_is_always_a_probability(self):
        for a, b in ((TIGHT_BEFORE, TIGHT_AFTER), (NOISY_BEFORE, NOISY_AFTER)):
            assert 0.0 <= rank_sum_p(a, b) <= 1.0

    def test_larger_samples_still_answer(self):
        # Past the exact limit the normal approximation takes over; it must still work.
        big_a = tuple(float(i) for i in range(30))
        big_b = tuple(float(i) + 20 for i in range(30))
        assert rank_sum_p(big_a, big_b) < ALPHA


class TestAttribution:
    def test_a_driver_update_is_named(self):
        causes = attribute(MACHINE, signature(driver="580.10"))
        assert causes == ("driver 512.89 -> 580.10",)

    def test_a_backend_update_is_named(self):
        causes = attribute(MACHINE, signature(backend="llama.cpp b11000"))
        assert causes == ("backend llama.cpp b10850 -> llama.cpp b11000",)

    def test_an_unchanged_machine_yields_nothing(self):
        assert attribute(MACHINE, signature()) == ()

    def test_several_changes_are_all_named(self):
        causes = attribute(MACHINE, signature(driver="580.10", gpu="NVIDIA RTX 4090"))
        assert len(causes) == 2

    def test_a_field_the_old_record_never_had_is_not_a_change(self):
        assert attribute({"driver": "512.89"}, signature()) == ()


class TestHistory:
    def test_it_is_keyed_by_the_model_not_the_signature(self, tmp_path):
        # A driver update changes the signature; splitting the history on that would
        # hide exactly the comparison the sentinel exists to make.
        first = history_path("sha256:" + "a" * 64, 1024, tmp_path)
        second = history_path("sha256:" + "a" * 64, 1024, tmp_path)
        assert first == second
        assert history_path("sha256:" + "b" * 64, 1024, tmp_path) != first

    def test_the_context_is_part_of_the_key(self, tmp_path):
        digest = "sha256:" + "a" * 64
        assert history_path(digest, 1024, tmp_path) != history_path(digest, 8192, tmp_path)

    def test_records_round_trip(self, tmp_path):
        path = tmp_path / "history.jsonl"
        append(path, record(TIGHT_BEFORE, peak_vram_mb=2560))
        append(path, record(TIGHT_AFTER, at="2026-09-10T00:00:00Z"))
        loaded = load(path)
        assert [r.samples for r in loaded] == [TIGHT_BEFORE, TIGHT_AFTER]
        assert loaded[0].peak_vram_mb == 2560
        assert loaded[0].signature["driver"] == "512.89"

    def test_a_corrupt_line_is_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "history.jsonl"
        append(path, record(TIGHT_BEFORE))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        append(path, record(TIGHT_AFTER))
        assert len(load(path)) == 2

    def test_an_absent_history_is_empty_not_an_error(self, tmp_path):
        assert load(tmp_path / "nothing.jsonl") == []

    def test_the_stored_shape_is_readable_by_anything(self, tmp_path):
        path = tmp_path / "history.jsonl"
        append(path, record(TIGHT_BEFORE))
        line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert line["samples"] == list(TIGHT_BEFORE)
        assert line["signature"]["gpu"] == MACHINE["gpu"]


class TestCausesTravel:
    def test_the_comparison_carries_what_changed(self):
        causes = attribute(MACHINE, signature(driver="580.10"))
        check = compare(record(TIGHT_BEFORE), record(TIGHT_AFTER), causes)
        assert check.causes == causes
        assert check.verdict is Verdict.SLOWER

    def test_an_untestable_comparison_still_reports_the_cause(self):
        causes = ("driver 512.89 -> 580.10",)
        assert compare(record(()), record(()), causes).causes == causes

    def test_a_comparison_defaults_to_no_causes(self):
        assert Comparison(Verdict.SAME, "detail").causes == ()
