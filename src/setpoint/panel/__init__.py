"""The panel: what setpoint measured, on one screen.

Every figure shown here comes from a command that already exists. The panel adds no
measurement of its own, which is what keeps it a viewer rather than a second way to
produce numbers.

Textual is an optional extra, so importing the widgets is deferred: the rest of setpoint
has to keep working with nothing installed.
"""

from __future__ import annotations

from .data import (
    Card,
    HistoryEntry,
    ProfileView,
    Row,
    Snapshot,
    gather,
    known_models,
    read_budget,
    read_card,
    read_history,
    read_profiles,
)

__all__ = [
    "Card",
    "HistoryEntry",
    "ProfileView",
    "Row",
    "Snapshot",
    "available",
    "gather",
    "known_models",
    "read_budget",
    "read_card",
    "read_history",
    "read_profiles",
    "run",
]


def available() -> bool:
    """Whether the extra is installed."""
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return True


def run(reference: str | None = None, context: int = 4096, interval: float = 2.0) -> int:
    """Start the panel, or explain what to install."""
    if not available():
        raise ImportError(
            "the panel needs the tui extra. Install it with `pip install setpoint[tui]`."
        )
    from .app import Panel

    Panel(reference=reference, context=context, interval=interval).run()
    return 0
