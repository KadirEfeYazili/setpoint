"""Model presets for llama-server's own router mode.

The engine grew a router of its own, so the measured profiles have a second mechanism to
drive. The file is an INI: one section per model, and inside it the server's own long
option names without their leading dashes. The schema is not in the upstream README; it
was established against the binary, which names the option it rejects.

Ports are left out on purpose. The router starts each model itself and assigns the port,
which is the difference from a config that spawns the process on the runner's behalf.
"""

from __future__ import annotations

from .llamaswap import ConfigEntry

# The router supplies these itself, and setting them from a preset would either be
# ignored or fight the router for the same resource.
SKIPPED_OPTIONS = ("port", "host", "alias")

COMMENT = "#"


def options_of(entry: ConfigEntry) -> list[tuple[str, str]]:
    """Turn one entry's command line into preset options, in the order they appeared."""
    argv = list(entry.command)[1:]
    out: list[tuple[str, str]] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        name = token[2:]
        follows = argv[index + 1] if index + 1 < len(argv) else None
        if follows is not None and not follows.startswith("--"):
            value, index = follows, index + 2
        else:
            # A switch with no argument. The router's parser wants a value, and the
            # server's own convention for these is "on".
            value, index = "on", index + 1
        if name not in SKIPPED_OPTIONS:
            out.append((name, value))
    return out


def render(entries: list[ConfigEntry], header: list[str] | None = None) -> str:
    """Emit the INI. The evidence stays as comments, the same as the other target."""
    lines: list[str] = []
    for line in header or ():
        lines.append(f"{COMMENT} {line}")
    if lines:
        lines.append("")

    for entry in entries:
        for comment in entry.comments:
            lines.append(f"{COMMENT} {comment}")
        lines.append(f"[{entry.name}]")
        for name, value in options_of(entry):
            lines.append(f"{name} = {value}")
        if entry.aliases:
            # One preset, several names a client may ask for.
            lines.append(f"alias = {','.join(entry.aliases)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
