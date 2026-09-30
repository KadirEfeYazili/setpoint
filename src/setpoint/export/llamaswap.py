"""Turn measured profiles into a llama-swap configuration.

setpoint does not serve HTTP and does not manage processes: a mature tool already does
both, and does them better than a first attempt would. What it does not do is know what
the settings should be - its configuration is hand-written, so the user still has to
guess `-ngl` and the rest. That gap is the whole of setpoint's contribution here.

Every entry carries the measurement that justifies it as a comment. A configuration
line without evidence beside it is exactly the guess this project exists to replace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..backend import server_argv
from ..profile import Profile

# llama-swap substitutes a free port here.
PORT_PLACEHOLDER = "${PORT}"

# How long a model stays loaded with nothing asking for it. A model holding most of the
# card blocks every other model, so it should yield sooner. These are policy, not
# measurement, and `--ttl` overrides them.
TTL_BANDS: tuple[tuple[float, int], ...] = (
    (0.75, 300),
    (0.40, 900),
    (0.00, 3600),
)

# Sleeping comes before unloading and is the cheaper of the two. Measured on a GTX 1650:
# it hands back 98.6% of the model's VRAM, and the next request pays 2.9-4.7 s to wake,
# against 3.8-5.8 s for a cold start. So a model that holds most of the card should
# yield it quickly. Policy, like the bands above; `--sleep-idle` overrides them.
SLEEP_BANDS: tuple[tuple[float, int], ...] = (
    (0.75, 60),
    (0.40, 180),
    (0.00, 600),
)

# Dots survive: a user types "qwen2.5", and turning it into "qwen2-5" makes the
# generated id harder to guess than the name it came from.
_SLUG = re.compile(r"[^a-z0-9.]+")


@dataclass
class ConfigEntry:
    """One model as llama-swap will see it."""

    name: str
    command: list[str]
    ttl: int
    label: str | None = None
    description: str | None = None
    comments: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)


def slug(text: str) -> str:
    """A model id a person can type: lowercase, hyphenated, no surprises."""
    return _SLUG.sub("-", text.lower()).strip("-.") or "model"


def entry_name(profile: Profile) -> str:
    """`qwen2.5-3b-instruct-c8192`. The context belongs in the name.

    A profile is measured for one context, and a different context is a different
    measurement. Hiding that in a shared name would serve one of them wrongly.
    """
    label = profile.model.name or profile.model.architecture or "model"
    return f"{slug(label)}-c{profile.target.context}"


def ttl_for(profile: Profile, vram_total_mb: int | None) -> int:
    """How long to keep an idle model loaded, from how much of the card it holds."""
    peak = profile.measurement.peak_vram_mb
    if not peak or not vram_total_mb:
        return TTL_BANDS[-1][1]
    share = peak / vram_total_mb
    for threshold, seconds in TTL_BANDS:
        if share >= threshold:
            return seconds
    return TTL_BANDS[-1][1]


def sleep_for(profile: Profile, vram_total_mb: int | None, ttl: int) -> int:
    """How long to keep an idle model's weights in VRAM before releasing them.

    Always inside the unload TTL: past that the runner takes the whole process away and
    sleeping first would have bought nothing.
    """
    peak = profile.measurement.peak_vram_mb
    seconds = SLEEP_BANDS[-1][1]
    if peak and vram_total_mb:
        share = peak / vram_total_mb
        for threshold, banded in SLEEP_BANDS:
            if share >= threshold:
                seconds = banded
                break
    return min(seconds, max(1, ttl - 1)) if ttl > 0 else seconds


def headline(profile: Profile) -> str:
    """The measurement in one line, for the runner's own interface to show."""
    stat = profile.measurement.decode_tok_s
    parts = [f"measured {stat.median:.2f} tok/s"]
    baseline = profile.baseline
    if baseline.failed:
        parts.append("the llama.cpp default does not start on this card")
    elif baseline.speedup is not None:
        parts.append(f"{baseline.speedup:.2f}x over the llama.cpp default")
    if profile.measurement.peak_vram_mb:
        parts.append(f"peak {profile.measurement.peak_vram_mb} MiB")
    return ", ".join(parts)


def evidence(profile: Profile) -> list[str]:
    """The measurement behind an entry, as comment lines."""
    stat = profile.measurement.decode_tok_s
    lines = [
        f"measured {stat.median:.2f} tok/s (spread {stat.spread:.1%}) "
        f"over {profile.measurement.runs} runs on {profile.measurement.measured_at}"
    ]
    if profile.measurement.peak_vram_mb:
        lines.append(f"peak VRAM {profile.measurement.peak_vram_mb} MiB")
    baseline = profile.baseline
    if baseline.failed:
        lines.append(f"the {baseline.label} did not start on this card")
    elif baseline.speedup is not None:
        lines.append(f"{baseline.speedup:.2f}x against the {baseline.label}")
    return lines


def build_entries(
    profiles: list[Profile],
    server_binary: str | Path = "llama-server",
    vram_total_mb: int | None = None,
    ttl_override: int | None = None,
    sleep_override: int | None = None,
    devices: tuple[str, ...] = (),
) -> list[ConfigEntry]:
    """One entry per profile, plus a short alias where a model has only one."""
    entries: list[ConfigEntry] = []
    for profile in sorted(profiles, key=entry_name):
        ttl = ttl_override if ttl_override is not None else ttl_for(profile, vram_total_mb)
        sleep = (
            sleep_override if sleep_override is not None else sleep_for(profile, vram_total_mb, ttl)
        )
        extra = ("--port", PORT_PLACEHOLDER)
        comments = evidence(profile)
        if sleep > 0:
            extra += ("--sleep-idle-seconds", str(sleep))
            comments.append(
                f"releases its VRAM after {sleep}s idle and reloads on the next request; "
                f"the runner unloads the process after {ttl}s"
            )
        argv = server_argv(
            server_binary,
            profile.model.path or "",
            profile.config,
            profile.target.context,
            devices=devices,
            extra=extra,
        )
        label = profile.model.name or profile.model.architecture or "model"
        entries.append(
            ConfigEntry(
                name=entry_name(profile),
                command=argv,
                ttl=ttl,
                label=f"{label} ({profile.target.context} ctx)",
                description=headline(profile),
                comments=comments,
            )
        )

    # A model measured at a single context gets its bare name as an alias, so a client
    # can ask for "qwen3-8b" without knowing which contexts were measured.
    stems = [name.rsplit("-c", 1)[0] for name in (e.name for e in entries)]
    for entry in entries:
        stem = entry.name.rsplit("-c", 1)[0]
        if stems.count(stem) == 1:
            entry.aliases.append(stem)
    return entries


def render(entries: list[ConfigEntry], header: list[str] | None = None) -> str:
    """Emit the YAML by hand, because the comments are the point.

    A serialiser would drop them, and an entry whose evidence has been stripped is
    indistinguishable from a guess.

    The command goes in a block scalar, one argument per line. A quoted scalar would
    process backslash escapes, which breaks the moment a Windows path appears; a block
    scalar takes the text verbatim, and it is also how llama-swap's own examples are
    written. `description` carries the headline measurement, so the evidence reaches the
    runner's interface and not only this file.
    """
    lines: list[str] = []
    for line in header or ():
        lines.append(f"# {line}")
    if header:
        lines.append("")
    lines.append("models:")
    if not entries:
        lines.append("  # no profiles match this machine yet. Run `setpoint tune <model>`.")
        return "\n".join(lines) + "\n"

    for entry in entries:
        lines.append("")
        for comment in entry.comments:
            lines.append(f"  # {comment}")
        lines.append(f"  {entry.name}:")
        if entry.label:
            lines.append(f"    name: {_scalar(entry.label)}")
        if entry.description:
            lines.append(f"    description: {_scalar(entry.description)}")
        lines.append("    cmd: |")
        for argument in _command_lines(entry.command):
            lines.append(f"      {argument}")
        lines.append(f"    ttl: {entry.ttl}")
        for alias in entry.aliases:
            if not lines[-1].startswith("    aliases:"):
                lines.append("    aliases:")
            lines.append(f"      - {alias}")
    return "\n".join(lines) + "\n"


def _command_lines(argv: list[str]) -> list[str]:
    """One flag and its value per line, which is how the command reads best."""
    if not argv:
        return []
    out = [_shell(argv[0])]
    index = 1
    while index < len(argv):
        token = argv[index]
        if token.startswith("-") and index + 1 < len(argv) and not argv[index + 1].startswith("-"):
            out.append(f"{token} {_shell(argv[index + 1])}")
            index += 2
        else:
            out.append(_shell(token))
            index += 1
    return out


def _shell(argument: str) -> str:
    """Quote for the command splitter, not for YAML: the block scalar handles YAML."""
    return f'"{argument}"' if " " in argument else argument


def _scalar(text: str) -> str:
    """A single-quoted YAML scalar, which leaves backslashes alone."""
    return "'" + text.replace("'", "''") + "'"
