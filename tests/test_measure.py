"""Measurement statistics tests.

The reliability rule decides which results are allowed to reach a profile, so it is
pinned rather than left to the caller.
"""

from __future__ import annotations

from setpoint.measure import MAX_RELIABLE_SPREAD, Statistic


class TestSummary:
    def test_median_ignores_a_single_outlier(self):
        assert Statistic((10.0, 10.0, 10.0, 10.0, 2.0)).median == 10.0

    def test_iqr_measures_the_middle_half(self):
        assert Statistic((1.0, 2.0, 3.0, 4.0, 5.0)).iqr == 2.0

    def test_spread_is_relative_to_the_median(self):
        assert Statistic((9.0, 10.0, 11.0)).spread == 0.2

    def test_no_samples_summarise_to_nothing(self):
        empty = Statistic(())
        assert empty.median is None
        assert empty.spread is None

    def test_one_sample_has_no_spread_to_report(self):
        single = Statistic((42.0,))
        assert single.iqr == 0.0
        assert single.spread == 0.0


class TestReliability:
    def test_a_tight_repeated_measurement_is_reliable(self):
        assert Statistic((100.0, 101.0, 100.5, 100.2, 100.8)).reliable

    def test_a_wide_spread_is_not(self):
        assert not Statistic((100.0, 140.0, 100.0, 145.0, 100.0)).reliable

    def test_too_few_runs_are_not_reliable_however_tight(self):
        # Two identical numbers prove nothing about the third run.
        assert not Statistic((100.0, 100.0)).reliable

    def test_the_threshold_is_where_it_says_it_is(self):
        below = Statistic((1.0 - MAX_RELIABLE_SPREAD / 2, 1.0, 1.0 + MAX_RELIABLE_SPREAD / 2))
        assert below.spread <= MAX_RELIABLE_SPREAD
        assert below.reliable

    def test_a_zero_median_yields_no_verdict(self):
        assert Statistic((0.0, 0.0, 0.0)).spread is None
        assert not Statistic((0.0, 0.0, 0.0)).reliable
