"""Spill detection tests.

The claim this makes is the strongest one setpoint makes about a running machine, so
the rules are pinned here rather than left to the display. Both halves of the signature
have to hold: shared memory alone climbs whenever a browser opens a tab, and dedicated
memory alone sits near capacity on any card doing its job.
"""

from __future__ import annotations

import pytest

from setpoint.monitor import (
    LOAD_UTILISATION_PCT,
    MIB,
    SPILL_SHARED_GROWTH_BYTES,
    Reading,
    Verdict,
    check_spill,
)

TOTAL = 4096 * MIB


def reading(
    used_mb: int = 2000,
    shared_mb: int | None = 100,
    busy: int = 80,
    at: float = 0.0,
) -> Reading:
    return Reading(
        at=at,
        vram_used_bytes=used_mb * MIB,
        vram_total_bytes=TOTAL,
        shared_bytes=None if shared_mb is None else shared_mb * MIB,
        shared_at=at,
        utilization_pct=busy,
    )


def series(shared_mb: list[int], used_mb: int = 2000, busy: int = 80) -> list[Reading]:
    return [
        reading(used_mb=used_mb, shared_mb=value, busy=busy, at=float(i))
        for i, value in enumerate(shared_mb)
    ]


class TestSpillSignature:
    def test_a_full_card_with_climbing_shared_memory_is_a_spill(self):
        check = check_spill(series([100, 200, 400, 700], used_mb=4000))
        assert check.verdict is Verdict.SPILLING
        assert "system RAM" in check.detail

    def test_climbing_shared_memory_alone_is_not(self):
        # Room left on the card means the growth belongs to something else.
        check = check_spill(series([100, 200, 400, 700], used_mb=1000))
        assert check.verdict is Verdict.HEALTHY

    def test_a_full_card_alone_is_not(self):
        # A card doing its job sits near capacity; that is the point of the card.
        check = check_spill(series([100, 100, 101, 100], used_mb=4000))
        assert check.verdict is Verdict.HEALTHY

    def test_growth_below_the_threshold_is_not_enough(self):
        under = SPILL_SHARED_GROWTH_BYTES // MIB - 8
        check = check_spill(series([100, 100 + under], used_mb=4000))
        assert check.verdict is Verdict.HEALTHY

    def test_growth_at_the_threshold_counts(self):
        over = SPILL_SHARED_GROWTH_BYTES // MIB
        check = check_spill(series([100, 100 + over], used_mb=4000))
        assert check.verdict is Verdict.SPILLING


class TestIdleMachine:
    def test_an_idle_gpu_is_reported_as_idle_not_healthy(self):
        # Saying "healthy" about a machine doing nothing would be a claim we cannot make.
        check = check_spill(series([100, 400, 900], used_mb=4000, busy=0))
        assert check.verdict is Verdict.IDLE
        assert "under load" in check.detail

    def test_one_loaded_reading_in_the_window_is_enough_to_judge(self):
        readings = series([100, 400, 900], used_mb=4000, busy=0)
        readings[-1] = reading(used_mb=4000, shared_mb=900, busy=LOAD_UTILISATION_PCT, at=2.0)
        assert check_spill(readings).verdict is Verdict.SPILLING


class TestWhatCannotBeSaid:
    def test_no_shared_reading_means_no_verdict(self):
        assert check_spill(series([100, 200])[:1]).verdict is Verdict.UNKNOWN

    def test_an_unreadable_counter_means_no_verdict(self):
        readings = [reading(shared_mb=None, at=float(i)) for i in range(4)]
        assert check_spill(readings).verdict is Verdict.UNKNOWN

    def test_an_unreadable_card_means_no_verdict(self):
        readings = [
            Reading(at=float(i), shared_bytes=100 * MIB, shared_at=float(i), utilization_pct=90)
            for i in range(4)
        ]
        assert check_spill(readings).verdict is Verdict.UNKNOWN
        assert "dedicated memory could not be read" in check_spill(readings).detail

    def test_an_empty_history_says_so_rather_than_guessing(self):
        assert check_spill([]).verdict is Verdict.UNKNOWN


class TestReadingFields:
    def test_the_shared_figure_carries_its_age(self):
        stale = Reading(at=10.0, shared_bytes=1, shared_at=7.5)
        assert stale.shared_age_s == pytest.approx(2.5)

    def test_no_shared_figure_has_no_age(self):
        assert Reading(at=10.0).shared_age_s is None

    def test_the_share_is_none_when_either_side_is_missing(self):
        assert Reading(at=0.0, vram_used_bytes=1).dedicated_share is None
        assert Reading(at=0.0, vram_total_bytes=1).dedicated_share is None

    @pytest.mark.parametrize(
        ("busy", "loaded"),
        [(0, False), (LOAD_UTILISATION_PCT - 1, False), (LOAD_UTILISATION_PCT, True)],
    )
    def test_the_load_threshold_is_where_it_says_it_is(self, busy, loaded):
        assert Reading(at=0.0, utilization_pct=busy).under_load is loaded


class TestTopLoop:
    """The live loop, with the machine stood in for."""

    def _stub_monitor(self, monkeypatch, verdicts, frames: list[int]):
        from setpoint import cli

        class Stub:
            def __init__(self, *_a, **_k):
                self.notes = []
                self._n = 0

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def tick(self):
                self._n += 1
                frames.append(self._n)
                return reading()

            @property
            def spill(self):
                from setpoint.monitor import SpillCheck

                return SpillCheck(verdicts[min(self._n - 1, len(verdicts) - 1)], "detail")

        monkeypatch.setattr(cli.monitor, "Monitor", Stub)

        def stop_after_two(_seconds):
            if len(frames) >= 2:
                raise KeyboardInterrupt

        monkeypatch.setattr(cli.time, "sleep", stop_after_two)
        return cli

    def test_it_redraws_until_interrupted(self, monkeypatch, capsys):
        frames: list[int] = []
        cli = self._stub_monitor(monkeypatch, [Verdict.HEALTHY], frames)
        args = type("A", (), {"interval": 0.01, "gpu": None})()
        assert cli.cmd_top(args) == 0
        assert len(frames) == 2
        assert "dedicated" in capsys.readouterr().out

    def test_a_spill_seen_at_any_point_sets_the_exit_code(self, monkeypatch, capsys):
        # The spill may pass; having seen it is still the answer the caller needs.
        frames: list[int] = []
        cli = self._stub_monitor(monkeypatch, [Verdict.SPILLING, Verdict.HEALTHY], frames)
        args = type("A", (), {"interval": 0.01, "gpu": None})()
        assert cli.cmd_top(args) == 1
        capsys.readouterr()
