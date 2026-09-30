"""Preview the mark's accent colour in your own terminal.

    .venv\Scripts\python.exe preview_banner.py

Deep blues, with the strokes inside the letters in white. Say which number, and it
goes in. Delete this file once a choice is made.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from setpoint.banner import _FACE, WORDMARK  # noqa: E402

RESET = "\033[0m"
GREY = "\033[90m"

PALETTES: dict[str, int] = {
    "1  navy 17      #00005f  (darkest)": 17,
    "2  navy 18      #000087": 18,
    "3  navy 19      #0000af  (in now)": 19,
    "4  blue 20      #0000d7": 20,
    "5  blue 21      #0000ff": 21,
    "6  steel 25     #005faf": 25,
    "7  azure 26     #005fd7": 26,
    "8  deep teal 23 #005f5f": 23,
    "9  indigo 55    #5f00af": 55,
}


def paint(line: str, accent: int) -> str:
    out: list[str] = []
    inside = False
    for ch in line:
        wants = ch in _FACE
        if wants != inside:
            out.append(f"\033[38;5;{accent}m" if wants else RESET)
            inside = wants
        out.append(ch)
    if inside:
        out.append(RESET)
    return "".join(out)


def main() -> int:
    for label, accent in PALETTES.items():
        print(f"{GREY}{label}{RESET}")
        for row in WORDMARK:
            print(paint(row, accent))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
