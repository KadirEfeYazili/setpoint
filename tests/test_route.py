"""Routing arithmetic tests.

The interesting cases are the ones where counting the switch changes the answer, because
that is the whole claim: a router that only knows throughput gets short requests wrong on
a card that holds one model at a time. The numbers used here are the ones measured on the
development machine, so a change to the arithmetic has to argue with them.
"""

from __future__ import annotations

from setpoint import route
from setpoint.measure import Statistic

# Measured: qwen2.5:3b 45.84 tok/s and 5.19 s to load, gemma3:1b 86.71 tok/s and 2.62 s.
QWEN = route.Candidate("qwen2.5:3b", decode_tok_s=45.84, load_seconds=5.19)
GEMMA = route.Candidate("gemma3:1b", decode_tok_s=86.71, load_seconds=2.62)


def loaded(candidate: route.Candidate) -> route.Candidate:
    return route.Candidate(
        name=candidate.name,
        decode_tok_s=candidate.decode_tok_s,
        load_seconds=candidate.load_seconds,
        resident=True,
    )


class TestCosting:
    def test_a_loaded_model_pays_nothing_to_switch(self):
        plan = route.plan(200, (loaded(QWEN),))
        assert plan.choices[0].switch_seconds == 0.0

    def test_a_cold_model_pays_its_measured_load(self):
        plan = route.plan(200, (GEMMA,))
        assert plan.choices[0].switch_seconds == 2.62

    def test_decode_time_is_tokens_over_throughput(self):
        plan = route.plan(200, (loaded(GEMMA),))
        assert plan.choices[0].decode_seconds == 200 / 86.71

    def test_a_candidate_with_no_measured_load_cannot_be_costed(self):
        unknown = route.Candidate("mystery", decode_tok_s=60.0)
        plan = route.plan(200, (unknown,))
        assert plan.choices == ()
        assert plan.unusable == (unknown,)

    def test_a_candidate_with_no_throughput_cannot_be_costed(self):
        # A profile that carries no median is not evidence of anything.
        empty = route.Candidate("empty", decode_tok_s=0.0, resident=True)
        assert route.plan(200, (empty,)).unusable == (empty,)


class TestDecision:
    def test_the_switch_cost_changes_the_answer_on_the_measured_pair(self):
        # 200 tokens: gemma3 decodes it in 2.31s against qwen's 4.36s, but the 2.62s
        # load turns that into 4.93s. This is the real case, not a constructed one.
        plan = route.plan(200, (loaded(QWEN), GEMMA))
        assert plan.best is not None
        assert plan.best.candidate.name == "qwen2.5:3b"
        assert plan.fastest is not None
        assert plan.fastest.candidate.name == "gemma3:1b"
        assert plan.switch_changed_the_answer

    def test_a_long_enough_request_pays_the_switch_back(self):
        plan = route.plan(2000, (loaded(QWEN), GEMMA))
        assert plan.best.candidate.name == "gemma3:1b"
        assert not plan.switch_changed_the_answer

    def test_it_reports_what_the_decision_saves(self):
        plan = route.plan(2000, (loaded(QWEN), GEMMA))
        saved = plan.saved_seconds
        assert saved is not None and saved > 0

    def test_staying_put_saves_nothing_to_report(self):
        plan = route.plan(20, (loaded(QWEN), GEMMA))
        assert plan.best is plan.resident
        assert plan.saved_seconds is None

    def test_with_nothing_loaded_the_switch_is_paid_either_way(self):
        plan = route.plan(200, (QWEN, GEMMA))
        assert plan.resident is None
        assert plan.best.candidate.name == "gemma3:1b"

    def test_no_candidates_decides_nothing(self):
        plan = route.plan(200, ())
        assert plan.best is None
        assert not plan.switch_changed_the_answer


class TestLoadCostStore:
    def cost(self) -> route.LoadCost:
        return route.LoadCost(
            model_digest="a" * 64,
            seconds=Statistic((2.60, 2.62, 2.71)),
            first_seconds=2.95,
        )

    def test_it_round_trips(self, tmp_path):
        route.save_load(self.cost(), tmp_path)
        read = route.read_load("a" * 64, tmp_path)
        assert read is not None
        assert read.median == 2.62
        assert read.first_seconds == 2.95

    def test_the_first_load_is_kept_apart_from_the_rest(self):
        # The page cache makes a repeat load a different measurement, so the first one
        # is never folded into the median.
        cost = self.cost()
        assert cost.first_seconds not in cost.seconds.samples

    def test_a_missing_file_reads_as_nothing(self, tmp_path):
        assert route.read_load("b" * 64, tmp_path) is None

    def test_a_corrupt_file_reads_as_nothing(self, tmp_path):
        path = route.loads_path("c" * 64, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        assert route.read_load("c" * 64, tmp_path) is None

    def test_a_file_with_no_samples_reads_as_nothing(self, tmp_path):
        path = route.loads_path("d" * 64, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"seconds": []}', encoding="utf-8")
        assert route.read_load("d" * 64, tmp_path) is None

    def test_it_keys_by_the_model_rather_than_the_whole_signature(self, tmp_path):
        # A driver update changes the signature but not what the file costs to read.
        assert route.loads_path("e" * 64, tmp_path).name.startswith("e" * 16)
