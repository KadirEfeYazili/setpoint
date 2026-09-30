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


class TestStreamFragments:
    def test_it_reads_a_token_out_of_an_event(self):
        line = b'data: {"choices":[{"delta":{"content":"hel"}}]}'
        assert serve._fragment(line) == "hel"

    def test_the_end_marker_yields_nothing(self):
        assert serve._fragment(b"data: [DONE]") == ""

    def test_a_keepalive_yields_nothing(self):
        assert serve._fragment(b"") == ""
        assert serve._fragment(b": ping") == ""

    def test_a_malformed_event_is_skipped_rather_than_fatal(self):
        # A stream that dies mid-token would otherwise take the conversation with it.
        assert serve._fragment(b"data: {not json") == ""

    def test_an_event_with_no_content_yields_nothing(self):
        assert serve._fragment(b'data: {"choices":[{"delta":{"role":"assistant"}}]}') == ""


class TestPort:
    def test_it_finds_a_port_nothing_is_using(self):
        assert 1024 < serve.free_port() <= 65535

    def test_two_asks_do_not_collide(self):
        assert serve.free_port() != serve.free_port()
