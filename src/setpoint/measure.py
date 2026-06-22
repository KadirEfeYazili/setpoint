"""Measurement statistics.

The project reports the median and keeps the interquartile range, rather than the mean
and standard deviation, because a single slow run from a throttle event or a background
process skews a mean and leaves no trace. A spread wider than the threshold marks the
result unreliable, and unreliable results are never written to a profile.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

# A run that is warming up, throttling or competing for the GPU shows up here first.
MAX_RELIABLE_SPREAD = 0.05

# Fewer runs than this cannot show a spread worth trusting either way.
MIN_RELIABLE_RUNS = 3


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
        return self.runs >= MIN_RELIABLE_RUNS and spread is not None and (
            spread <= MAX_RELIABLE_SPREAD
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "runs": self.runs,
            "median": self.median,
            "iqr": self.iqr,
            "spread": self.spread,
            "reliable": self.reliable,
        }
