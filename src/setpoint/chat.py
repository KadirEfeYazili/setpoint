"""Talking to the model setpoint measured, and showing what each answer cost.

The point is not the conversation. It is that every reply arrives with its own
throughput and peak VRAM beside it, on the configuration a measurement chose, so the
number in the profile stops being a claim and becomes something felt.

No proxy and no endpoint: this is a client to a server started here and stopped when the
conversation ends.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

# Room kept for the answer. A conversation that fills the context leaves the model no
# space to reply, which looks like the model refusing rather than the context running out.
DEFAULT_RESERVE = 512

# Used only when the server cannot be asked. Four characters to a token is the usual
# rule of thumb for English and it is wrong for everything else, so it is labelled an
# estimate wherever it reaches the user.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Turn:
    """One message, and for an answer, what it cost."""

    role: str
    text: str
    decode_tok_s: float | None = None
    tokens: int | None = None
    peak_vram_mib: int | None = None
    speculator: str | None = None

    @property
    def measured(self) -> bool:
        return self.decode_tok_s is not None


@dataclass
class Conversation:
    """The turns so far, and how they are handed to the server."""

    system: str | None = None
    turns: list[Turn] = field(default_factory=list)
    dropped: int = 0

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)

    def messages(self) -> list[dict[str, str]]:
        head = [{"role": "system", "content": self.system}] if self.system else []
        return head + [{"role": turn.role, "content": turn.text} for turn in self.turns]

    @property
    def answers(self) -> list[Turn]:
        return [turn for turn in self.turns if turn.role == "assistant"]

    @property
    def rate(self) -> float | None:
        """Median decode rate across the answers, which is what the profile claims."""
        rates = sorted(t.decode_tok_s for t in self.answers if t.decode_tok_s is not None)
        if not rates:
            return None
        middle = len(rates) // 2
        if len(rates) % 2:
            return rates[middle]
        return (rates[middle - 1] + rates[middle]) / 2

    def clear(self) -> None:
        self.turns.clear()
        self.dropped = 0


def estimate_tokens(text: str) -> int:
    """A token count for when the server cannot be asked for a real one."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def trim(
    conversation: Conversation,
    context: int,
    count: Callable[[str], int] = estimate_tokens,
    reserve: int = DEFAULT_RESERVE,
) -> int:
    """Drop the oldest turns until the rest fits, and say how many went.

    Oldest first, in pairs where it can: dropping a question but keeping its answer
    leaves the model reading its own words with nothing to have prompted them.
    """
    budget = context - reserve - (count(conversation.system) if conversation.system else 0)
    if budget <= 0:
        removed = len(conversation.turns)
        conversation.turns.clear()
        conversation.dropped += removed
        return removed

    removed = 0
    while conversation.turns and sum(count(t.text) for t in conversation.turns) > budget:
        conversation.turns.pop(0)
        removed += 1
        if conversation.turns and conversation.turns[0].role == "assistant":
            conversation.turns.pop(0)
            removed += 1
    conversation.dropped += removed
    return removed


def token_counter(base: str) -> Callable[[str], int]:
    """A counter backed by the server's own tokenizer, falling back to the estimate.

    Asking is worth the round trip: the estimate is off by a factor on a non-English
    conversation, and being wrong here means either wasting context or overrunning it.
    """

    def count(text: str) -> int:
        if not text:
            return 0
        request = urllib.request.Request(
            f"{base}/tokenize",
            data=json.dumps({"content": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                tokens = json.loads(response.read()).get("tokens")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return estimate_tokens(text)
        return len(tokens) if isinstance(tokens, list) else estimate_tokens(text)

    return count
