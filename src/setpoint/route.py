"""Which local model should answer this request, given what it costs to switch.

A router that knows only throughput sends every short request to the fastest model. On a
card that holds one sizeable model at a time that is the wrong answer often enough to
matter: the faster model has to be loaded first, and the load can cost more than the
request saves. The switch cost is measurable and nobody publishes it, so setpoint
measures it and hands the decision over with its arithmetic shown.

This does not serve requests. Classifying a request and routing it are a runner's job,
and llama-server carries both now. What is produced here is the measured input that
neither of them has.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .backend import find_server_binary, server_argv
from .measure import Statistic
from .profile import store

# A load is I/O bound and shares the disk with everything else on the machine, so its
# spread is wider than a decode measurement. Measured 6-13% across three models, which
# is above the gate a profile has to pass; it is reported, not hidden.
DEFAULT_LOAD_RUNS = 3
READY_TIMEOUT_S = 300.0
STOP_TIMEOUT_S = 30.0
POLL_S = 0.05

LOADS_DIRNAME = "loads"


@dataclass(frozen=True)
class LoadCost:
    """What it took to bring one model up, on this machine.

    `first_seconds` is the earliest observation, kept apart from the rest because the
    page cache makes every later load a different measurement. Dropping the cache needs
    privileges setpoint does not ask for, so this is labelled rather than called cold.
    """

    model_digest: str
    seconds: Statistic
    first_seconds: float | None = None
    measured_at: str | None = None

    @property
    def median(self) -> float | None:
        return self.seconds.median

    @property
    def spread(self) -> float | None:
        return self.seconds.spread


@dataclass(frozen=True)
class Candidate:
    """One model that could answer, with what is known about it."""

    name: str
    decode_tok_s: float
    load_seconds: float | None = None
    resident: bool = False
    context: int | None = None

    @property
    def usable(self) -> bool:
        """Whether this candidate can be costed at all."""
        return self.decode_tok_s > 0 and (self.resident or self.load_seconds is not None)


@dataclass(frozen=True)
class Choice:
    """One candidate costed for one request."""

    candidate: Candidate
    decode_seconds: float
    switch_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.decode_seconds + self.switch_seconds


@dataclass(frozen=True)
class Plan:
    """Every candidate costed, and the cheapest one."""

    tokens: int
    choices: tuple[Choice, ...] = field(default_factory=tuple)
    unusable: tuple[Candidate, ...] = field(default_factory=tuple)

    @property
    def best(self) -> Choice | None:
        return min(self.choices, key=lambda c: c.total_seconds) if self.choices else None

    @property
    def resident(self) -> Choice | None:
        return next((c for c in self.choices if c.candidate.resident), None)

    @property
    def saved_seconds(self) -> float | None:
        """What the decision saves against staying where we are."""
        best, staying = self.best, self.resident
        if best is None or staying is None or best is staying:
            return None
        return staying.total_seconds - best.total_seconds

    @property
    def fastest(self) -> Choice | None:
        """What a router that ignored the switch cost would have picked."""
        return max(self.choices, key=lambda c: c.candidate.decode_tok_s) if self.choices else None

    @property
    def switch_changed_the_answer(self) -> bool:
        """Whether counting the switch picked a different model than throughput alone."""
        best, fastest = self.best, self.fastest
        return best is not None and fastest is not None and best is not fastest


def plan(tokens: int, candidates: tuple[Candidate, ...]) -> Plan:
    """Cost every candidate for a request of `tokens` output tokens.

    Only decode is counted. Prompt processing depends on a prompt this function has not
    been given, and guessing it would put an estimate inside a measurement.
    """
    choices = []
    unusable = []
    for candidate in candidates:
        if not candidate.usable:
            unusable.append(candidate)
            continue
        switch = 0.0 if candidate.resident else float(candidate.load_seconds or 0.0)
        choices.append(
            Choice(
                candidate=candidate,
                decode_seconds=tokens / candidate.decode_tok_s,
                switch_seconds=switch,
            )
        )
    return Plan(tokens=tokens, choices=tuple(choices), unusable=tuple(unusable))


def loads_dir(directory: Path | None = None) -> Path:
    return (directory or store.home()) / LOADS_DIRNAME


def loads_path(model_digest: str, directory: Path | None = None) -> Path:
    return loads_dir(directory) / f"{model_digest[:16]}.json"


def save_load(cost: LoadCost, directory: Path | None = None) -> Path:
    path = loads_path(cost.model_digest, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_digest": cost.model_digest,
        "seconds": list(cost.seconds.samples),
        "first_seconds": cost.first_seconds,
        "measured_at": cost.measured_at or store.now(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_load(model_digest: str, directory: Path | None = None) -> LoadCost | None:
    path = loads_path(model_digest, directory)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    samples = tuple(float(s) for s in raw.get("seconds") or () if isinstance(s, int | float))
    if not samples:
        return None
    return LoadCost(
        model_digest=str(raw.get("model_digest") or model_digest),
        seconds=Statistic(samples),
        first_seconds=raw.get("first_seconds"),
        measured_at=raw.get("measured_at"),
    )


def measure_load(
    model_path: str | Path,
    config: object,
    context: int,
    model_digest: str,
    devices: tuple[str, ...] = (),
    runs: int = DEFAULT_LOAD_RUNS,
    binary: str | Path | None = None,
) -> LoadCost | None:
    """Time how long the server takes to answer /health after being started.

    The first load is measured and kept separately, then `runs` more are measured and
    summarised. What is timed is the whole path a switch really costs: process start,
    weights read, and the backend's own setup.
    """
    server = binary or find_server_binary()
    if server is None:
        return None
    argv = server_argv(server, model_path, config, context, devices)
    observed = [_one_load(argv) for _ in range(runs + 1)]
    timings = [t for t in observed if t is not None]
    if len(timings) < 2:
        return None
    return LoadCost(
        model_digest=model_digest,
        seconds=Statistic(tuple(timings[1:])),
        first_seconds=timings[0],
        measured_at=store.now(),
    )


def _one_load(argv: list[str]) -> float | None:
    port = _free_port()
    process = subprocess.Popen(
        [*argv, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    started = time.monotonic()
    try:
        deadline = started + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return None
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=1
                ) as response:
                    if response.status == 200:
                        return time.monotonic() - started
            except (urllib.error.URLError, TimeoutError, OSError):
                time.sleep(POLL_S)
        return None
    finally:
        process.terminate()
        try:
            process.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
