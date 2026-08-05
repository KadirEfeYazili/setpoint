"""Noticing when a machine stops being the machine a profile was measured on.

A profile is only worth what its measurement was worth on the day it was taken. Drivers
update, backends update, cards get swapped, and nothing announces that the numbers have
moved. This keeps the history and compares the newest reading against the last one.

The comparison is a rank-sum test rather than a fixed percentage, because a fixed
percentage cannot tell two identical-looking changes apart. Measured on this project's
own data: a 5% drop with tight repetitions comes out at p = 0.008, and a 5% drop with
overlapping repetitions at p = 0.889. One is a regression, the other is a quiet
afternoon, and a threshold sees them as the same number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .measure import Statistic, rank_sum_p
from .profile import Profile, Signature, profiles_dir

# A difference this unlikely under "nothing changed" is worth reporting.
ALPHA = 0.05

# A real difference smaller than this is not worth re-tuning for. Run-to-run spread was
# measured at 0.56% on real hardware, so this sits comfortably above the noise floor
# while staying small enough to catch a driver update that costs a few percent.
MIN_MATERIAL_CHANGE = 0.02

HISTORY_DIRNAME = "history"


class Verdict(str, Enum):
    UNKNOWN = "unknown"
    SAME = "same"
    FASTER = "faster"
    SLOWER = "slower"


@dataclass(frozen=True)
class Record:
    """One check, kept so the next one has something to compare against."""

    at: str
    samples: tuple[float, ...]
    signature: dict[str, object] = field(default_factory=dict)
    peak_vram_mb: int | None = None
    context: int | None = None
    note: str | None = None

    @property
    def statistic(self) -> Statistic:
        return Statistic(self.samples)

    def to_dict(self) -> dict[str, object]:
        return {
            "at": self.at,
            "samples": list(self.samples),
            "signature": self.signature,
            "peak_vram_mb": self.peak_vram_mb,
            "context": self.context,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> Record:
        return cls(
            at=str(raw.get("at", "")),
            samples=tuple(float(v) for v in raw.get("samples") or ()),
            signature=dict(raw.get("signature") or {}),
            peak_vram_mb=raw.get("peak_vram_mb"),
            context=raw.get("context"),
            note=raw.get("note"),
        )


@dataclass(frozen=True)
class Comparison:
    verdict: Verdict
    detail: str
    change: float | None = None
    p_value: float | None = None
    causes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def actionable(self) -> bool:
        return self.verdict is Verdict.SLOWER


def history_dir(root: Path | None = None) -> Path:
    return (root or profiles_dir().parent) / HISTORY_DIRNAME


def history_path(model_digest: str, context: int, root: Path | None = None) -> Path:
    """Keyed by the model, not by the whole signature.

    The signature changes the moment a driver does, and a history that split on that
    could never show the thing it exists to show: what the driver update cost.
    """
    stem = model_digest.split(":")[-1][:16] or "unknown"
    return history_dir(root) / f"{stem}-c{context}.jsonl"


def record_of(profile: Profile, samples: tuple[float, ...], at: str, **extra) -> Record:
    signature = profile.signature
    return Record(
        at=at,
        samples=samples,
        signature={
            "gpu": signature.gpu,
            "vram_total_mb": signature.vram_total_mb,
            "driver": signature.driver,
            "backend": signature.backend,
            "platform": signature.platform,
        },
        context=profile.target.context,
        **extra,
    )


def append(path: Path, record: Record) -> None:
    """One JSON object per line, so a long history stays cheap to append to."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict()) + "\n")


def load(path: Path) -> list[Record]:
    """Every readable record. A corrupt line is skipped, not fatal."""
    if not path.is_file():
        return []
    records: list[Record] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(Record.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return records


def attribute(before: dict[str, object], after: Signature) -> tuple[str, ...]:
    """What about the machine changed between two checks, in words."""
    current = {
        "gpu": after.gpu,
        "vram_total_mb": after.vram_total_mb,
        "driver": after.driver,
        "backend": after.backend,
        "platform": after.platform,
    }
    labels = {
        "gpu": "GPU",
        "vram_total_mb": "VRAM",
        "driver": "driver",
        "backend": "backend",
        "platform": "platform",
    }
    changes = []
    for key, label in labels.items():
        was, now = before.get(key), current[key]
        if was is not None and was != now:
            changes.append(f"{label} {was} -> {now}")
    return tuple(changes)


def compare(before: Record, after: Record, causes: tuple[str, ...] = ()) -> Comparison:
    """Is the newest reading different from the last one, and enough to act on?

    Two questions, and both have to be answered. Whether the difference is larger than
    the repetitions can explain is a statistical question; whether it is large enough to
    matter is not, and a test that only asks the first will report a real one-percent
    drift as news.
    """
    old, new = before.statistic.median, after.statistic.median
    if old is None or new is None or old <= 0:
        return Comparison(Verdict.UNKNOWN, "there is nothing to compare against", causes=causes)

    change = new / old - 1
    p_value = rank_sum_p(before.samples, after.samples)

    if p_value is None:
        return Comparison(
            Verdict.UNKNOWN,
            "the readings cannot be told apart statistically",
            change,
            None,
            causes,
        )

    if p_value >= ALPHA:
        return Comparison(
            Verdict.SAME,
            f"{change:+.1%}, which the run-to-run spread explains (p = {p_value:.3f})",
            change,
            p_value,
            causes,
        )

    if abs(change) < MIN_MATERIAL_CHANGE:
        return Comparison(
            Verdict.SAME,
            f"{change:+.1%} is measurably real (p = {p_value:.3f}) but too small to act on",
            change,
            p_value,
            causes,
        )

    verdict = Verdict.SLOWER if change < 0 else Verdict.FASTER
    return Comparison(
        verdict,
        f"{change:+.1%}, beyond what the repetitions explain (p = {p_value:.3f})",
        change,
        p_value,
        causes,
    )
