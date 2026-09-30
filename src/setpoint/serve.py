"""Running a llama-server for the length of a block, and talking to it.

setpoint does not serve requests; it starts the engine with a measured configuration and
speaks to it as a client. That distinction is the whole of this module: there is no
routing, no queue and no endpoint of our own, only a process that exists while something
here needs an answer from it.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .backend import BackendDevice, BackendError, LlamaCppBackend, select_device, server_argv

READY_TIMEOUT_S = 180.0
REQUEST_TIMEOUT_S = 600.0
STOP_TIMEOUT_S = 30.0
POLL_S = 1.0


@dataclass(frozen=True)
class Reply:
    """One answer, with what it cost to produce."""

    text: str
    decode_tok_s: float | None = None
    prompt_tok_s: float | None = None
    tokens: int | None = None
    prompt_tokens: int | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.detail is None


@dataclass
class Session:
    """A llama-server started for one configuration and stopped afterwards."""

    binary: str | Path
    model_path: str | Path
    config: object
    context: int
    devices: tuple[str, ...] = ()
    speculator: str | None = None
    port: int | None = None
    extra: tuple[str, ...] = ()
    log: object = None
    argv: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.last: Reply | None = None
        self.port = self.port or free_port()
        flags: tuple[str, ...] = ("--port", str(self.port), "--metrics", *self.extra)
        if self.speculator:
            flags += ("--spec-type", self.speculator)
        self.argv = server_argv(
            self.binary, self.model_path, self.config, self.context, self.devices, flags
        )
        self._process: subprocess.Popen | None = None

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> Session:
        sink = self.log if self.log is not None else subprocess.DEVNULL
        self._process = subprocess.Popen(self.argv, stdout=sink, stderr=subprocess.STDOUT)
        if not self.wait_ready():
            self.__exit__(None, None, None)
            raise BackendError("llama-server did not become ready")
        return self

    def __exit__(self, *_: object) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def wait_ready(self, timeout_s: float = READY_TIMEOUT_S) -> bool:
        """Block until the server answers, or until it gives up starting."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._process is None or self._process.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(f"{self.base}/health", timeout=2) as response:
                    if response.status == 200:
                        return True
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(POLL_S)
        return False

    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> Reply:
        """One answer, and the timings the server reports for it."""
        payload: dict[str, object] = {
            "model": "setpoint",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        try:
            body = self._post("/v1/chat/completions", payload)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return Reply(text="", detail=str(exc))
        return _reply_of(body)

    def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Token fragments as they arrive.

        A local model answers slowly enough that waiting for the whole reply feels like
        a hang, so the pieces are handed over as they come.

        The timings of this reply land in `last`, taken from the stream's own closing
        event. Timing a second request instead would measure that request.
        """
        self.last = None
        payload = {
            "model": "setpoint",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        request = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        text: list[str] = []
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            for raw in response:
                event = _event(raw)
                if event is None:
                    continue
                piece = _content_of(event)
                if piece:
                    text.append(piece)
                    yield piece
                if event.get("timings") or event.get("usage"):
                    self.last = _reply_of({**event, "choices": []})
        if self.last is not None:
            self.last = Reply(
                text="".join(text),
                decode_tok_s=self.last.decode_tok_s,
                prompt_tok_s=self.last.prompt_tok_s,
                tokens=self.last.tokens,
                prompt_tokens=self.last.prompt_tokens,
            )

    def counters(self) -> dict[str, int]:
        """The server's speculation counters, which say whether a drafter fired."""
        try:
            with urllib.request.urlopen(f"{self.base}/metrics", timeout=10) as response:
                return parse_counters(response.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError):
            return {}

    def _post(self, path: str, payload: dict[str, object]) -> dict:
        request = urllib.request.Request(
            f"{self.base}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            return json.loads(response.read())


def _reply_of(body: dict) -> Reply:
    choices = body.get("choices") or [{}]
    message = choices[0].get("message") or {}
    timings = body.get("timings") or {}
    usage = body.get("usage") or {}
    return Reply(
        text=str(message.get("content") or ""),
        decode_tok_s=timings.get("predicted_per_second"),
        prompt_tok_s=timings.get("prompt_per_second"),
        tokens=usage.get("completion_tokens") or timings.get("predicted_n"),
        prompt_tokens=usage.get("prompt_tokens") or timings.get("prompt_n"),
    )


def _event(raw: bytes) -> dict | None:
    """One server-sent event, or nothing for a keepalive or the end marker."""
    line = raw.decode("utf-8", "replace").strip()
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        parsed = json.loads(data)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _content_of(event: dict) -> str:
    delta = (event.get("choices") or [{}])[0].get("delta") or {}
    return str(delta.get("content") or "")


def parse_counters(text: str) -> dict[str, int]:
    """Speculation counters out of the server's Prometheus text.

    The per-position breakdown carries labels and is skipped: the totals are what a
    decision needs, and the labelled lines would collide on name.
    """
    out: dict[str, int] = {}
    for line in text.splitlines():
        if not line.startswith("llamacpp:spec_decode_num") or "{" in line:
            continue
        name, _, value = line.partition(" ")
        try:
            out[name.split(":", 1)[1]] = int(float(value))
        except ValueError:
            continue
    return out


def pin_device(prefer: str | None) -> tuple[tuple[str, ...], BackendDevice | None]:
    """Resolve the card by name and return the flag value to pin it with.

    Device ids are positional and a reboot renumbers them, so the id stored in a
    profile can name different hardware today. With one device there is nothing to
    pin and no flag is returned.
    """
    try:
        listing = LlamaCppBackend().devices()
    except BackendError:
        return (), None
    chosen = select_device(listing, prefer) if listing else None
    if chosen is None or len(listing) < 2:
        return (), chosen
    return (chosen.id,), chosen


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
