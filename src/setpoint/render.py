"""Terminal rendering.

Colour is opt-out via NO_COLOR and is disabled automatically when stdout is not a
terminal, so piping to a file or to jq stays clean.
"""

from __future__ import annotations

import os
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
BLUE = "\033[34m"
GREY = "\033[90m"

# One accent and one shade of it. The face reads as the brand colour, the bevel a step
# behind it, which is what gives the wordmark depth without introducing a second hue.
ACCENT = "\033[96m"
ACCENT_DIM = "\033[36m"


def color_enabled(stream: object | None = None) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("SETPOINT_FORCE_COLOR"):
        return True
    target = stream if stream is not None else sys.stdout
    return bool(getattr(target, "isatty", lambda: False)())


class Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap(BOLD, text)

    def dim(self, text: str) -> str:
        return self._wrap(DIM, text)

    def red(self, text: str) -> str:
        return self._wrap(RED, text)

    def yellow(self, text: str) -> str:
        return self._wrap(YELLOW, text)

    def green(self, text: str) -> str:
        return self._wrap(GREEN, text)

    def blue(self, text: str) -> str:
        return self._wrap(BLUE, text)

    def grey(self, text: str) -> str:
        return self._wrap(GREY, text)

    def accent(self, text: str) -> str:
        return self._wrap(ACCENT, text)

    def accent_dim(self, text: str) -> str:
        return self._wrap(ACCENT_DIM, text)


def wrap(text: str, width: int, indent: str = "") -> list[str]:
    """Greedy word wrap. Long tokens such as paths are left intact."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = indent + words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = indent + word
    lines.append(current)
    return lines


def gib(count: float) -> str:
    """Gibibytes, the unit every VRAM figure is compared in."""
    return f"{count / 1024**3:.2f} GiB"


def human_bytes(count: float) -> str:
    """Bytes at whichever scale reads naturally. For prose, not for columns."""
    for limit, unit in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if abs(count) >= limit:
            return f"{count / limit:.1f} {unit}"
    return f"{count:.0f} B"


def short_path(path: object, home: str | None = None) -> str:
    """Collapse the home directory to `~` so paths stay readable in a narrow terminal."""
    text = str(path).replace("\\", "/")
    root = (home if home is not None else os.path.expanduser("~")).replace("\\", "/")
    if root and text.startswith(root):
        text = "~" + text[len(root) :]
    head, _, name = text.rpartition("/")
    if len(name) > 28:
        name = f"{name[:16]}..{name[-8:]}"
    return f"{head}/{name}" if head else name


# Terminals that render UTF-8 whatever the console code page says. Windows still
# reports the ANSI code page to Python, so without this the box drawing is unprintable
# on a machine whose terminal displays it perfectly well.
_UTF8_TERMINALS = ("WT_SESSION", "TERM_PROGRAM", "VSCODE_INJECTION")


def enable_unicode_output(stream: object | None = None) -> bool:
    """Switch stdout to UTF-8 where the terminal is known to render it.

    Left alone everywhere else: writing UTF-8 to a console that reads it as a legacy
    code page produces mojibake, which is worse than the ASCII fallback.
    """
    target = stream if stream is not None else sys.stdout
    if (getattr(target, "encoding", "") or "").lower().replace("-", "") == "utf8":
        return True
    if not any(os.environ.get(name) for name in _UTF8_TERMINALS):
        return False
    reconfigure = getattr(target, "reconfigure", None)
    if reconfigure is None:
        return False
    try:
        reconfigure(encoding="utf-8")
    except (OSError, ValueError):
        return False
    return True


def encodable(text: str, stream: object | None = None) -> bool:
    """Whether the output encoding can represent `text`.

    A Windows console on a non-Latin-1 code page cannot print box drawing, and Python
    raises rather than substituting. Anything decorative has to ask first.
    """
    target = stream if stream is not None else sys.stdout
    encoding = getattr(target, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def term_width(default: int = 88) -> int:
    """Width to lay text out in. Capped, because prose stops being readable past it."""
    return min(terminal_columns(default), 100)


def terminal_columns(default: int = 88) -> int:
    """The real width. Anything that must not wrap has to ask for this one."""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default
