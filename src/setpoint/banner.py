"""The wordmark, in whichever size the terminal can hold.

Width is read at render time and the tier is chosen from it. A banner that wraps is
worse than no banner: it destroys the shape it exists to show and leaves a horizontal
scrollbar behind. Nothing here ever emits a line longer than the width it was given.
"""

from __future__ import annotations

from .render import RESET, Style, encodable, terminal_columns

# Face and bevel are separate character sets so the two can be coloured apart. The
# smaller tiers are drawn from half blocks, which belong to the face.
_FACE = frozenset("█▀▄▌▐")
_BEVEL = frozenset("╗╔╝╚║═")

WORDMARK: tuple[str, ...] = (
    "███████╗███████╗████████╗██████╗  ██████╗ ██╗███╗   ██╗████████╗",
    "██╔════╝██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗██║████╗  ██║╚══██╔══╝",
    "███████╗█████╗     ██║   ██████╔╝██║   ██║██║██╔██╗ ██║   ██║   ",
    "╚════██║██╔══╝     ██║   ██╔═══╝ ██║   ██║██║██║╚██╗██║   ██║   ",
    "███████║███████╗   ██║   ██║     ╚██████╔╝██║██║ ╚████║   ██║   ",
    "╚══════╝╚══════╝   ╚═╝   ╚═╝      ╚═════╝ ╚═╝╚═╝  ╚═══╝   ╚═╝   ",
)

# There is no half-height form. One was tried and it read as a cut-off wordmark rather
# than as a smaller one, every time it was shown: block letters drawn at half height
# look like the top of a letter, not like a letter. Below the wordmark the mark takes
# over, because a small complete thing beats a large truncated one.
#
# The mark, then the name alone.
MARK = "▐█▌ setpoint"
NAME = "setpoint"

# What is printed when the output encoding cannot carry box drawing. Not a style
# choice: a Windows console on a Turkish code page raises on the wordmark, which would
# take the whole command down with it.
ASCII_RULE = "-"

TAGLINE = "measurement-driven configuration for local inference"
TAGLINE_SHORT = "measured, not guessed"

RULE = "─"

WORDMARK_WIDTH = max(len(line) for line in WORDMARK)
MARK_WIDTH = len(MARK)

# The mark may use the full width. It is centred, so a terminal exactly as wide as the
# mark still shows the whole of it rather than dropping to a smaller form.
MARGIN = 0

# Two colours, both stated. The face carries the accent; the bevel is bright white.
#
# Neither is left to the terminal. A darker shade of the accent sank into a dark
# background and took the strokes inside the letterforms with it; the theme's own
# foreground turned out to be too close to the accent to read as a second colour. White
# against a coloured face is the one pair that stays legible either way.
# The panel's own scrollbar blue, and white for the strokes inside the letters.
#
# Blue carries only 0.0722 of relative luminance, so a dark one cannot be seen on a dark
# terminal at all: #0000af measures 1.62:1 against black, #0000ff 2.44:1, and the
# scrollbar's resting #003054 only 1.55:1, where a filled block needs about 3:1. The
# scrollbar's lit state, #0178d4, measures 4.64:1 and is the same blue awake.
FACE_COLOUR = 32
BEVEL_COLOUR = 231

HEX: dict[int, str] = {FACE_COLOUR: "#0178d4", BEVEL_COLOUR: "#ffffff"}


def _colour(index: int) -> str:
    return f"\033[38;5;{index}m"


# Largest to smallest. `limit` names the biggest form a caller will accept, which is
# how the panel keeps a working screen from being eaten by a nine-line mark.
FORMS = ("wordmark", "mark", "name")


def tier(width: int, unicode_ok: bool = True, limit: str | None = None) -> str:
    """The largest form that fits whole.

    Every form is a drawing, so it is chosen or it is not; a form is never cut down to
    size. Half a letterform is what a broken logo looks like, and the only string here
    that survives being cut is the name itself.
    """
    if not unicode_ok:
        return "ascii"
    allowed = FORMS[FORMS.index(limit) :] if limit in FORMS else FORMS
    for form, needed in (("wordmark", WORDMARK_WIDTH), ("mark", MARK_WIDTH)):
        if form in allowed and width >= needed + MARGIN:
            return form
    return "name"


def lines(
    width: int | None = None,
    tagline: bool = True,
    unicode_ok: bool | None = None,
    limit: str | None = None,
) -> list[str]:
    """The banner as plain text, guaranteed to fit `width` and to be printable."""
    width = terminal_columns() if width is None else width
    if unicode_ok is None:
        unicode_ok = encodable(WORDMARK[0] + MARK)
    chosen = tier(width, unicode_ok, limit)
    if chosen == "ascii":
        out = [ASCII_RULE * len(NAME), NAME, ASCII_RULE * len(NAME)]
        if tagline:
            out.append(TAGLINE if len(TAGLINE) <= width else TAGLINE_SHORT)
    elif chosen == "wordmark":
        rule = RULE * WORDMARK_WIDTH
        out = [rule, *WORDMARK, rule]
        if tagline:
            out.append(TAGLINE)
    elif chosen == "mark":
        rule = RULE * MARK_WIDTH
        out = [rule, MARK, rule]
        if tagline:
            out.append(TAGLINE if len(TAGLINE) <= width else TAGLINE_SHORT)
    else:
        out = [NAME]
    # The graphic is padded, not trimmed: cutting the trailing space off a block
    # letterform leaves a ragged right edge, which reads as a logo that failed to finish
    # drawing. The tagline is prose and keeps no trailing space.
    # The graphic is padded, not trimmed: cutting the trailing space off a block
    # letterform leaves a ragged right edge. Only the tagline, which is prose, is cut.
    graphic = [line for line in out if line not in (TAGLINE, TAGLINE_SHORT)]
    span = min(max((len(line) for line in graphic), default=0), width)
    return [
        line[:width] if line in (TAGLINE, TAGLINE_SHORT) else line.ljust(span)[:width]
        for line in out
    ]


def centre(body: list[str], width: int) -> list[str]:
    """Centre the graphic as a block, and the tagline under it.

    The block moves as one piece: centring each of its rows separately would shear the
    letterforms, because the rows of a block letter are not all the same length. The
    tagline is prose and is centred on its own.
    """
    graphic = [line for line in body if line not in (TAGLINE, TAGLINE_SHORT)]
    span = max((len(line) for line in graphic), default=0)
    pad = " " * max((width - span) // 2, 0)
    return [
        " " * max((width - len(line)) // 2, 0) + line
        if line in (TAGLINE, TAGLINE_SHORT)
        else pad + line
        for line in body
    ]


def paint(text: str, style: Style) -> str:
    """Colour one row: the face in the accent, the bevel a shade behind it.

    Runs are grouped rather than coloured per character, because a per-character escape
    triples the byte count of every line for no visible difference.
    """
    if not style.enabled:
        return text
    out: list[str] = []
    current: str | None = None
    for ch in text:
        code = (
            _colour(FACE_COLOUR) if ch in _FACE else _colour(BEVEL_COLOUR) if ch in _BEVEL else None
        )
        if code != current:
            if current is not None:
                out.append(RESET)
            if code is not None:
                out.append(code)
            current = code
        out.append(ch)
    if current is not None:
        out.append(RESET)
    return "".join(out)


def render(
    style: Style,
    width: int | None = None,
    tagline: bool = True,
    unicode_ok: bool | None = None,
    limit: str | None = None,
    centred: bool = False,
) -> list[str]:
    """The banner, coloured.

    Not centred by default. A command prints its banner once and the terminal owns the
    lines afterwards: leading padding makes every row longer than the mark, so narrowing
    the window wraps and shatters a block that would otherwise still have fitted. The
    panel redraws on resize and asks for centring.
    """
    width = terminal_columns() if width is None else width
    body = lines(width, tagline, unicode_ok, limit)
    if centred:
        body = centre(body, width)
    out = []
    for line in body:
        if line in (TAGLINE, TAGLINE_SHORT):
            out.append(style.grey(line))
        elif line.strip() and set(line.strip()) <= set(RULE + ASCII_RULE):
            out.append(_colour(FACE_COLOUR) + line + RESET if style.enabled else line)
        else:
            out.append(paint(line, style))
    return out


def markup(
    width: int,
    tagline: bool = True,
    limit: str | None = None,
    centred: bool = False,
    face: str | None = None,
) -> str:
    """The banner as Rich markup, for the panel.

    The terminal inside a TUI is the widget, not the window, so the width comes from the
    caller rather than from the environment.
    """
    face = face or HEX[FACE_COLOUR]
    out: list[str] = []
    body = lines(width, tagline, unicode_ok=True, limit=limit)
    for line in centre(body, width) if centred else body:
        if line in (TAGLINE, TAGLINE_SHORT):
            out.append(f"[dim]{line}[/]")
        else:
            out.append(_markup_run(line, face))
    return "\n".join(out)


def _markup_run(line: str, face: str) -> str:
    """Face in the accent, strokes inside the letters in white."""
    out: list[str] = []
    for ch in line:
        if ch in _FACE or ch == RULE:
            out.append(f"[{face}]{ch}[/]")
        elif ch in _BEVEL:
            out.append(f"[{HEX[BEVEL_COLOUR]}]{ch}[/]")
        else:
            out.append(ch)
    return "".join(out)
