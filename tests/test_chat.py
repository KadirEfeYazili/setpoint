"""Conversation tests.

The talking needs a server; the bookkeeping around it does not. What is pinned here is
what happens when the context runs out, and that a reply which was never timed is never
reported as having been.
"""

from __future__ import annotations

from setpoint import chat


def words(count: int) -> str:
    return "word " * count


def conversation(pairs: int = 3, size: int = 40, system: str | None = None) -> chat.Conversation:
    talk = chat.Conversation(system=system)
    for i in range(pairs):
        talk.add(chat.Turn("user", words(size)))
        talk.add(chat.Turn("assistant", words(size), decode_tok_s=80.0 + i))
    return talk


def count_words(text: str) -> int:
    """A stand-in tokenizer: one token per word, so the arithmetic is readable."""
    return len(text.split())


class TestMessages:
    def test_the_system_prompt_leads(self):
        talk = conversation(1, system="be brief")
        assert talk.messages()[0] == {"role": "system", "content": "be brief"}

    def test_without_a_system_prompt_nothing_is_invented(self):
        assert all(m["role"] != "system" for m in conversation(1).messages())

    def test_the_turns_keep_their_order(self):
        roles = [m["role"] for m in conversation(2).messages()]
        assert roles == ["user", "assistant", "user", "assistant"]


class TestTrimming:
    def test_a_conversation_that_fits_is_left_alone(self):
        talk = conversation(2, size=10)
        assert chat.trim(talk, context=1000, count=count_words, reserve=100) == 0
        assert len(talk.turns) == 4

    def test_the_oldest_turns_go_first(self):
        talk = conversation(4, size=40)
        talk.turns[-1] = chat.Turn("assistant", "the newest answer")
        chat.trim(talk, context=200, count=count_words, reserve=50)
        assert talk.turns[-1].text == "the newest answer"

    def test_it_drops_until_the_rest_fits(self):
        talk = conversation(4, size=40)
        chat.trim(talk, context=200, count=count_words, reserve=50)
        assert sum(count_words(t.text) for t in talk.turns) <= 150

    def test_an_answer_is_never_left_without_its_question(self):
        # A conversation that starts on an assistant turn reads as the model talking to
        # itself, and some templates refuse it outright.
        talk = conversation(4, size=40)
        chat.trim(talk, context=200, count=count_words, reserve=50)
        assert not talk.turns or talk.turns[0].role == "user"

    def test_what_was_dropped_is_counted(self):
        talk = conversation(4, size=40)
        dropped = chat.trim(talk, context=200, count=count_words, reserve=50)
        assert dropped > 0
        assert talk.dropped == dropped

    def test_dropping_accumulates_across_turns(self):
        talk = conversation(4, size=40)
        chat.trim(talk, context=200, count=count_words, reserve=50)
        first = talk.dropped
        talk.add(chat.Turn("user", words(300)))
        chat.trim(talk, context=200, count=count_words, reserve=50)
        assert talk.dropped > first

    def test_a_system_prompt_eats_into_the_budget(self):
        wide = conversation(3, size=20)
        narrow = conversation(3, size=20, system=words(100))
        kept_wide = len(wide.turns) - chat.trim(wide, 200, count_words, 20)
        kept_narrow = len(narrow.turns) - chat.trim(narrow, 200, count_words, 20)
        assert kept_narrow < kept_wide

    def test_a_context_smaller_than_the_reserve_clears_rather_than_loops(self):
        talk = conversation(2)
        assert chat.trim(talk, context=10, count=count_words, reserve=512) == 4
        assert talk.turns == []


class TestRate:
    def test_the_median_is_taken_over_the_answers(self):
        talk = chat.Conversation()
        for rate in (60.0, 80.0, 100.0):
            talk.add(chat.Turn("assistant", "x", decode_tok_s=rate))
        assert talk.rate == 80.0

    def test_an_even_number_of_answers_averages_the_middle(self):
        talk = chat.Conversation()
        for rate in (60.0, 80.0):
            talk.add(chat.Turn("assistant", "x", decode_tok_s=rate))
        assert talk.rate == 70.0

    def test_answers_the_server_never_timed_are_not_counted_as_zero(self):
        # The chat printed 0.0 t/s once, from a rate it had not measured. A missing
        # timing has to stay missing.
        talk = chat.Conversation()
        talk.add(chat.Turn("assistant", "x", decode_tok_s=None))
        assert talk.rate is None
        assert not talk.answers[0].measured

    def test_questions_do_not_count_towards_the_rate(self):
        talk = chat.Conversation()
        talk.add(chat.Turn("user", "x"))
        talk.add(chat.Turn("assistant", "y", decode_tok_s=50.0))
        assert talk.rate == 50.0

    def test_clearing_forgets_the_turns_and_the_tally(self):
        talk = conversation(2)
        talk.dropped = 3
        talk.clear()
        assert talk.turns == []
        assert talk.dropped == 0


class TestEstimate:
    def test_it_never_returns_zero_for_real_text(self):
        # A zero would make a turn look free and it would never be trimmed.
        assert chat.estimate_tokens("a") >= 1

    def test_it_grows_with_the_text(self):
        assert chat.estimate_tokens("x" * 400) > chat.estimate_tokens("x" * 40)
