"""Speculation report tests.

The measurements themselves need a server and a card. What is pinned here is the
judgement built on top of them, including the two ways a real run misled us: a drafter
that produced nothing while the timing drifted upwards, and a reading whose spread was
eight times the reliability gate.
"""

from __future__ import annotations

from setpoint import speculate
from setpoint.measure import Statistic

PROSE = speculate.DEFAULT_WORKLOADS[0]


def trial(
    speculator: str | None = "ngram-simple",
    samples: tuple[float, ...] = (100.0, 100.5, 101.0),
    drafted: int = 384,
    accepted: int = 260,
    steps: int = 8,
) -> speculate.Trial:
    return speculate.Trial(
        workload=PROSE.name,
        speculator=speculator,
        throughput=Statistic(samples),
        drafted=drafted,
        accepted=accepted,
        steps=steps,
    )


def result(*trials: speculate.Trial, baseline: float = 80.0) -> speculate.WorkloadResult:
    return speculate.WorkloadResult(
        workload=PROSE,
        baseline=trial(speculator=None, samples=(baseline, baseline, baseline), drafted=0),
        trials=trials,
    )


class TestTrial:
    def test_a_drafter_that_produced_nothing_says_so(self):
        quiet = trial(drafted=0, accepted=0, steps=0)
        assert not quiet.fired
        assert quiet.acceptance is None
        assert quiet.tokens_per_step is None

    def test_acceptance_is_the_share_of_drafts_kept(self):
        assert trial(drafted=400, accepted=100).acceptance == 0.25

    def test_tokens_per_step_counts_the_target_token_too(self):
        # Four accepted drafts over two steps is two per step, plus the model's own.
        assert trial(accepted=4, steps=2).tokens_per_step == 3.0

    def test_a_wide_spread_is_not_reliable(self):
        assert not trial(samples=(60.0, 100.0, 140.0)).reliable

    def test_too_few_runs_is_not_reliable(self):
        assert not trial(samples=(100.0,)).reliable

    def test_a_tight_repeated_reading_is_reliable(self):
        assert trial().reliable


class TestPick:
    def test_it_picks_the_largest_reliable_gain(self):
        weak = trial(speculator="ngram-cache", samples=(90.0, 90.0, 90.0))
        strong = trial(speculator="ngram-simple", samples=(160.0, 160.0, 160.0))
        assert result(weak, strong).best is strong

    def test_a_speculator_that_never_fired_is_never_picked(self):
        # Measured: three speculators drafted nothing at all, and their timings still
        # moved by a percent either way. A speedup they cannot have caused is noise.
        idle = trial(samples=(160.0, 160.0, 160.0), drafted=0, accepted=0, steps=0)
        assert result(idle).best is None

    def test_an_unreliable_reading_is_never_picked(self):
        # Measured: ngram-mod on prose read +73% with a 34% spread.
        noisy = trial(samples=(100.0, 140.0, 180.0))
        assert result(noisy).best is None

    def test_a_gain_too_small_to_act_on_is_not_picked(self):
        assert result(trial(samples=(82.0, 82.0, 82.0))).best is None

    def test_a_loss_is_not_picked(self):
        assert result(trial(samples=(70.0, 70.0, 70.0))).best is None

    def test_gain_is_measured_against_the_baseline_of_its_own_workload(self):
        one = result(trial(samples=(120.0, 120.0, 120.0)), baseline=80.0)
        assert one.gain(one.trials[0]) == 0.5


class TestReport:
    def _report(self, prose_best: bool, edit_best: bool) -> speculate.Report:
        def make(workload, winning):
            samples = (160.0, 160.0, 160.0) if winning else (80.0, 80.0, 80.0)
            return speculate.WorkloadResult(
                workload=workload,
                baseline=trial(speculator=None, samples=(80.0, 80.0, 80.0), drafted=0),
                trials=(trial(samples=samples),),
            )

        return speculate.Report(
            model="m.gguf",
            results=(
                make(speculate.DEFAULT_WORKLOADS[0], prose_best),
                make(speculate.DEFAULT_WORKLOADS[1], edit_best),
            ),
        )

    def test_one_winner_everywhere_is_a_machine_wide_answer(self):
        assert self._report(True, True).agreed == "ngram-simple"

    def test_workloads_that_disagree_have_no_single_answer(self):
        # This is the measured case, not a corner: nothing on prose, +96% on an edit.
        report = self._report(False, True)
        assert report.agreed is None
        assert report.winners == {"prose": None, "edit": "ngram-simple"}

    def test_nothing_anywhere_is_reported_as_nothing(self):
        report = self._report(False, False)
        assert report.agreed is None
        assert not any(report.winners.values())


class TestWorkloads:
    def test_the_defaults_cover_both_ends_of_repetition(self):
        # A single workload would have hidden a factor of twenty in acceptance.
        assert len(speculate.DEFAULT_WORKLOADS) >= 2
        assert {w.name for w in speculate.DEFAULT_WORKLOADS} == {"prose", "edit"}

    def test_the_edit_workload_quotes_its_own_input(self):
        edit = speculate.DEFAULT_WORKLOADS[1]
        assert "Repeat the following passage" in edit.prompt

    def test_every_default_speculator_needs_no_draft_model(self):
        assert all(name.startswith("ngram-") for name in speculate.DRAFT_FREE)
