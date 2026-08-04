"""Measurement statistics.

The project reports the median and keeps the interquartile range, rather than the mean
and standard deviation, because a single slow run from a throttle event or a background
process skews a mean and leaves no trace. A spread wider than the threshold marks the
result unreliable, and unreliable results are never written to a profile.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

# A run that is warming up, throttling or competing for the GPU shows up here first.
MAX_RELIABLE_SPREAD = 0.05

# Fewer runs than this cannot show a spread worth trusting either way.
MIN_RELIABLE_RUNS = 3


# Above this many samples per side, enumerating every split costs more than the normal
# approximation is worth. Below it the approximation is poor and the exact count is cheap.
EXACT_TEST_LIMIT = 10


def rank_sum_p(before: Sequence[float], after: Sequence[float]) -> float | None:
    """Two-sided p-value for "these two sets came from the same distribution".

    A Wilcoxon rank-sum test, computed exactly for the sample counts a benchmark
    actually produces. Five repetitions per side is far too few for the normal
    approximation the textbooks reach for, and enumerating all 252 splits costs
    nothing, so the small case is counted rather than approximated.

    `None` means the question cannot be answered: one side is empty, or every value is
    identical and there is nothing to rank.
    """
    n1, n2 = len(before), len(after)
    if n1 == 0 or n2 == 0:
        return None
    combined = list(before) + list(after)
    if len(set(combined)) == 1:
        return None

    observed = _rank_sum(before, combined)
    expected = n1 * (n1 + n2 + 1) / 2

    if n1 <= EXACT_TEST_LIMIT and n2 <= EXACT_TEST_LIMIT:
        return _exact_p(combined, n1, observed, expected)
    return _normal_p(combined, n1, n2, observed, expected)


def _ranks(values: Sequence[float]) -> dict[float, float]:
    """Average ranks, so ties do not favour whichever side was listed first."""
    ordered = sorted(values)
    ranks: dict[float, float] = {}
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1] == ordered[index]:
            end += 1
        ranks[ordered[index]] = (index + end) / 2 + 1
        index = end + 1
    return ranks


def _rank_sum(subset: Sequence[float], combined: Sequence[float]) -> float:
    ranks = _ranks(combined)
    return sum(ranks[value] for value in subset)


def _exact_p(combined: list[float], n1: int, observed: float, expected: float) -> float:
    """Count every way the values could have been split between the two sides."""
    ranks = _ranks(combined)
    scores = [ranks[value] for value in combined]
    deviation = abs(observed - expected)
    total = 0
    extreme = 0
    for pick in combinations(range(len(scores)), n1):
        total += 1
        if abs(sum(scores[i] for i in pick) - expected) >= deviation - 1e-9:
            extreme += 1
    return extreme / total


def _normal_p(
    combined: list[float], n1: int, n2: int, observed: float, expected: float
) -> float | None:
    ranks = _ranks(combined)
    counts: dict[float, int] = {}
    for value in combined:
        counts[ranks[value]] = counts.get(ranks[value], 0) + 1
    n = n1 + n2
    ties = sum(c**3 - c for c in counts.values())
    variance = n1 * n2 * (n + 1) / 12 - n1 * n2 * ties / (12 * n * (n - 1))
    if variance <= 0:
        return None
    z = (abs(observed - expected) - 0.5) / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2))


@dataclass(frozen=True)
class Statistic:
    """Repeated measurements of one quantity, summarised the way the project reports it."""

    samples: tuple[float, ...]

    @property
    def runs(self) -> int:
        return len(self.samples)

    @property
    def median(self) -> float | None:
        return statistics.median(self.samples) if self.samples else None

    @property
    def iqr(self) -> float | None:
        """Interquartile range. Zero for a single sample, which says nothing about spread."""
        if not self.samples:
            return None
        if len(self.samples) < 2:
            return 0.0
        low, _, high = statistics.quantiles(self.samples, n=4, method="inclusive")
        return high - low

    @property
    def spread(self) -> float | None:
        """IQR as a fraction of the median, so thresholds hold across magnitudes."""
        median, iqr = self.median, self.iqr
        if median is None or iqr is None or median == 0:
            return None
        return iqr / median

    @property
    def reliable(self) -> bool:
        """Whether this result may be written to a profile."""
        spread = self.spread
        return (
            self.runs >= MIN_RELIABLE_RUNS
            and spread is not None
            and (spread <= MAX_RELIABLE_SPREAD)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "runs": self.runs,
            "median": self.median,
            "iqr": self.iqr,
            "spread": self.spread,
            "reliable": self.reliable,
        }
