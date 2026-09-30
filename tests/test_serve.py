"""Server session tests.

Starting a server and reading its answers is the seam between setpoint and the engine.
The parts that need no engine are pinned here: what the command line comes out as, and
how a reply and the server's counters are read back.
"""

from __future__ import annotations

from setpoint import serve
from setpoint.profile import Config


class TestCounters:
    TEXT = "\n".join(
        [
            "# HELP llamacpp:spec_decode_num_draft_tokens_total Total draft tokens",
            "llamacpp:spec_decode_num_draft_tokens_total 384",
            "llamacpp:spec_decode_num_accepted_tokens_total 260",
            "llamacpp:spec_decode_num_drafts_total 8",
            'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 45',
            "llamacpp:n_decode_total 1200",
        ]
    )

    def test_it_reads_the_totals(self):
        counters = serve.parse_counters(self.TEXT)
        assert counters["spec_decode_num_draft_tokens_total"] == 384
        assert counters["spec_decode_num_accepted_tokens_total"] == 260
        assert counters["spec_decode_num_drafts_total"] == 8

    def test_it_skips_the_labelled_breakdown_and_unrelated_metrics(self):
        counters = serve.parse_counters(self.TEXT)
        assert not any("per_pos" in name for name in counters)
        assert not any("n_decode" in name for name in counters)

    def test_an_empty_body_reads_as_nothing_rather_than_failing(self):
        assert serve.parse_counters("") == {}


class TestSession:
    def _argv(self, speculator: str | None) -> list[str]:
        session = serve.Session(
            binary="llama-server",
            model_path="m.gguf",
            config=Config(n_gpu_layers=27, ubatch_size=128),
            context=4096,
            devices=("Vulkan0",),
            speculator=speculator,
            port=1234,
        )
        return session.argv

    def test_it_asks_for_the_metrics_endpoint(self):
        # Acceptance is read from the counters, so a session without them is useless.
        assert "--metrics" in self._argv("ngram-mod")

    def test_it_names_the_speculator_it_was_asked_for(self):
        argv = self._argv("ngram-mod")
        assert argv[argv.index("--spec-type") + 1] == "ngram-mod"

    def test_the_baseline_session_names_no_speculator(self):
        assert "--spec-type" not in self._argv(None)

    def test_it_carries_the_measured_configuration(self):
        argv = self._argv(None)
        assert argv[argv.index("--n-gpu-layers") + 1] == "27"
        assert argv[argv.index("--ubatch-size") + 1] == "128"
        assert argv[argv.index("--ctx-size") + 1] == "4096"


class TestReply:
    def test_it_reads_the_answer_and_what_it_cost(self):
        body = {
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "timings": {
                "predicted_per_second": 86.71,
                "prompt_per_second": 240.0,
                "predicted_n": 12,
                "prompt_n": 30,
            },
            "usage": {"completion_tokens": 12, "prompt_tokens": 30},
        }
        reply = serve._reply_of(body)
        assert reply.text == "hello"
        assert reply.decode_tok_s == 86.71
        assert reply.tokens == 12
        assert reply.ok

    def test_a_cached_prefix_does_not_look_like_a_processed_one(self):
        # The two numbers are not the same and once were merged into one field:
        # the prompt was 706 tokens long, one of them was processed, and dividing
        # the first by a rate derived from the second reported fifteen seconds of
        # prefill that never happened.
        body = {
            "choices": [{"message": {"content": "hello"}}],
            "timings": {
                "cache_n": 705,
                "prompt_n": 1,
                "prompt_ms": 31.497,
                "prompt_per_second": 31.75,
                "predicted_per_second": 81.95,
            },
            "usage": {
                "prompt_tokens": 706,
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 705},
            },
        }
        reply = serve._reply_of(body)
        assert reply.prompt_tokens == 706
        assert reply.prompt_processed == 1
        assert reply.cached_tokens == 705
        assert reply.prompt_ms == 31.497

    def test_prefill_time_is_taken_from_the_server_not_derived(self):
        # Deriving it as prompt_tokens / prompt_per_second is wrong whenever any of
        # the prompt was cached, which in a conversation is almost always.
        body = {
            "choices": [{"message": {"content": "x"}}],
            "timings": {"prompt_n": 706, "prompt_ms": 1032.801, "prompt_per_second": 683.58},
            "usage": {"prompt_tokens": 706},
        }
        reply = serve._reply_of(body)
        assert reply.prompt_ms == 1032.801
        assert reply.cached_tokens is None

    def test_an_answer_without_timings_still_carries_its_text(self):
        # The measurement is the point, but losing it must not lose the reply.
        reply = serve._reply_of({"choices": [{"message": {"content": "hi"}}]})
        assert reply.text == "hi"
        assert reply.decode_tok_s is None
        assert reply.ok

    def test_a_failure_is_not_an_empty_answer(self):
        # "" and "it did not work" are different things to show a person.
        failed = serve.Reply(text="", detail="connection refused")
        assert not failed.ok


def piece(raw: bytes) -> str:
    event = serve._event(raw)
    return serve._content_of(event) if event else ""


class TestStreamFragments:
    def test_it_reads_a_token_out_of_an_event(self):
        assert piece(b'data: {"choices":[{"delta":{"content":"hel"}}]}') == "hel"

    def test_the_end_marker_is_not_an_event(self):
        assert serve._event(b"data: [DONE]") is None

    def test_a_keepalive_is_not_an_event(self):
        assert serve._event(b"") is None
        assert serve._event(b": ping") is None

    def test_a_malformed_event_is_skipped_rather_than_fatal(self):
        # A stream that dies mid-token would otherwise take the conversation with it.
        assert serve._event(b"data: {not json") is None

    def test_an_event_with_no_content_yields_nothing(self):
        assert piece(b'data: {"choices":[{"delta":{"role":"assistant"}}]}') == ""

    def test_the_closing_event_carries_the_timings(self):
        # Where the rate has to come from. Timing a second request instead reported
        # that request, which is how the chat once printed 0.0 t/s.
        event = serve._event(
            b'data: {"choices":[],"timings":{"predicted_per_second":72.7,"predicted_n":23}}'
        )
        assert event is not None
        assert serve._reply_of(event).decode_tok_s == 72.7


class TestPort:
    def test_it_finds_a_port_nothing_is_using(self):
        assert 1024 < serve.free_port() <= 65535

    def test_two_asks_do_not_collide(self):
        assert serve.free_port() != serve.free_port()
