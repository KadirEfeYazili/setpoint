"""Autotuner tests.

The search is driven by a synthetic landscape with a known optimum, so what is checked
here is the algorithm's behaviour - that it climbs, that it stops, that it survives a
failed measurement, that it never claims an improvement the noise could explain - rather
than any property of a real machine.
"""

from __future__ import annotations

import pytest

from setpoint.profile import Config
from setpoint.tune import (
    MIN_IMPROVEMENT,
    Effort,
    SearchSpace,
    Stage,
    Trial,
    Verdict,
    is_improvement,
    neighbours,
    search,
)

SCREEN = Effort(repetitions=2, n_depth=1024, label="screen")
FULL = Effort(repetitions=5, n_depth=8192, label="full")

SPACE = SearchSpace(max_gpu_layers=37, max_threads=12)

# Where the synthetic optimum sits.
PEAK_LAYERS = 24
PEAK_UBATCH = 256


def config(**overrides) -> Config:
    fields = {
        "n_gpu_layers": PEAK_LAYERS,
        "ubatch_size": 512,
        "batch_size": 2048,
        "threads": 6,
        "flash_attn": True,
    }
    fields.update(overrides)
    return Config(**fields)


class Landscape:
    """A smooth peak, plus a record of what was asked for."""

    def __init__(self, fails: set[int] | None = None, spread: float = 0.005) -> None:
        self.asked: list[tuple[Config, Effort]] = []
        self.fails = fails or set()
        self.spread = spread

    def __call__(self, cfg: Config, effort: Effort) -> Trial:
        self.asked.append((cfg, effort))
        if cfg.n_gpu_layers in self.fails:
            return Trial(cfg, detail="out of memory")
        score = 30.0 - abs(PEAK_LAYERS - (cfg.n_gpu_layers or 0)) * 0.8
        if cfg.ubatch_size == PEAK_UBATCH:
            score += 2.0
        if cfg.flash_attn:
            score += 1.0
        return Trial(cfg, score=score, spread=self.spread, reliable=True)

    @property
    def configs(self) -> list[Config]:
        return [c for c, _ in self.asked]


def seeds(layers: tuple[int, ...] = (18, 22, 24, 30)) -> list[Config]:
    return [config(n_gpu_layers=n) for n in layers]


class TestImprovementRule:
    def test_a_clear_gain_counts(self):
        current = Trial(config(), score=10.0, spread=0.01)
        better = Trial(config(), score=12.0, spread=0.01)
        assert is_improvement(current, better)

    def test_a_gain_under_the_floor_does_not(self):
        current = Trial(config(), score=10.0, spread=0.0)
        marginal = Trial(config(), score=10.0 * (1 + MIN_IMPROVEMENT / 2), spread=0.0)
        assert not is_improvement(current, marginal)

    def test_noisy_measurements_raise_the_bar(self):
        # A 5% gain between two measurements that each wobble 10% proves nothing.
        current = Trial(config(), score=10.0, spread=0.10)
        noisy = Trial(config(), score=10.5, spread=0.10)
        assert not is_improvement(current, noisy)

    def test_a_failed_measurement_is_never_an_improvement(self):
        assert not is_improvement(Trial(config(), score=1.0), Trial(config()))

    def test_anything_beats_nothing(self):
        assert is_improvement(Trial(config()), Trial(config(), score=1.0))


class TestNeighbours:
    def test_layer_moves_come_first(self):
        labels = [label for label, _ in neighbours(config(), SPACE)]
        assert labels[0].startswith("-ngl")

    def test_moves_stay_inside_the_bounds(self):
        edge = config(n_gpu_layers=SPACE.max_gpu_layers, threads=SPACE.max_threads)
        for _, moved in neighbours(edge, SPACE):
            assert 0 <= moved.n_gpu_layers <= SPACE.max_gpu_layers
            assert 1 <= moved.threads <= SPACE.max_threads

    def test_the_context_cache_type_is_never_a_tuning_axis(self):
        # Cache precision follows from the budget the user accepted, not from the search.
        for _, moved in neighbours(config(), SPACE):
            assert moved.cache_type_k == config().cache_type_k
            assert moved.cache_type_v == config().cache_type_v

    def test_expert_placement_only_moves_on_a_moe_model(self):
        dense = neighbours(config(n_cpu_moe=4), SPACE)
        moe = neighbours(config(n_cpu_moe=4), SearchSpace(37, 12, moe=True))
        assert not any(label.startswith("-ncmoe") for label, _ in dense)
        assert any(label.startswith("-ncmoe") for label, _ in moe)

    def test_flash_attention_is_left_alone_when_it_is_unset(self):
        assert not any("flash" in label for label, _ in neighbours(config(flash_attn=None), SPACE))


class TestSearch:
    def test_it_climbs_to_the_optimum(self):
        result = search(seeds(), SPACE, Landscape(), SCREEN, FULL)
        assert result.best.config.n_gpu_layers == PEAK_LAYERS
        assert result.best.config.ubatch_size == PEAK_UBATCH

    def test_it_finds_the_peak_even_when_no_seed_is_on_it(self):
        result = search(seeds((10, 14, 18)), SPACE, Landscape(), SCREEN, FULL)
        assert result.best.config.n_gpu_layers == PEAK_LAYERS

    def test_screening_drops_the_weaker_seeds(self):
        result = search(seeds(), SPACE, Landscape(), SCREEN, FULL)
        dropped = [s for s in result.steps if s.verdict is Verdict.DROPPED]
        assert dropped
        assert all(s.stage is Stage.SCREEN for s in dropped)

    def test_the_winner_is_re_measured_at_full_effort(self):
        # No number that reaches a profile may come from a screening run.
        landscape = Landscape()
        result = search(seeds(), SPACE, landscape, SCREEN, FULL)
        confirm = [s for s in result.steps if s.stage is Stage.CONFIRM]
        assert len(confirm) == 1
        assert confirm[0].config == result.best.config
        assert (result.best.config, FULL) in landscape.asked

    def test_no_configuration_is_measured_twice(self):
        landscape = Landscape()
        search(seeds(), SPACE, landscape, SCREEN, FULL)
        screened = [c for c, e in landscape.asked if e is SCREEN]
        assert len(screened) == len(set(screened))

    def test_it_stops_when_nothing_improves(self):
        landscape = Landscape()
        search([config(ubatch_size=PEAK_UBATCH)], SPACE, landscape, SCREEN, FULL)
        # Starting on the peak, one sweep finds nothing better and the search ends.
        assert len(landscape.asked) < 30

    def test_the_measurement_budget_is_respected(self):
        landscape = Landscape()
        result = search(seeds(), SPACE, landscape, SCREEN, FULL, measurement_budget=6)
        assert result.measurements <= 7  # the confirmation run is always allowed
        assert "stopped after" in result.reason

    def test_every_decision_is_recorded(self):
        result = search(seeds(), SPACE, Landscape(), SCREEN, FULL)
        assert result.steps
        assert {s.stage for s in result.steps} >= {Stage.SCREEN, Stage.CONFIRM}
        assert all(s.verdict in set(Verdict) for s in result.steps)
        assert result.measurements > 0


class TestWarmUp:
    """A GPU reads low while its clocks ramp, so the first measurement is thrown away."""

    def test_the_first_measurement_is_discarded(self):
        landscape = Landscape()
        result = search(seeds(), SPACE, landscape, SCREEN, FULL)
        warmups = [s for s in result.steps if s.stage is Stage.WARMUP]
        assert len(warmups) == 1
        assert result.steps[0].stage is Stage.WARMUP

    def test_the_warm_up_never_becomes_the_answer(self):
        def only_the_first_is_fast(cfg, effort):
            score = 100.0 if not landscape.asked else 10.0
            landscape.asked.append((cfg, effort))
            return Trial(cfg, score=score, spread=0.005, reliable=True)

        landscape = Landscape()
        result = search(seeds(), SPACE, only_the_first_is_fast, SCREEN, FULL)
        assert result.best.score == 10.0

    def test_an_interrupt_never_reports_the_warm_up_as_best(self):
        calls = {"n": 0}

        def fast_then_stop(cfg, effort):
            calls["n"] += 1
            if calls["n"] > 1:
                raise KeyboardInterrupt
            return Trial(cfg, score=999.0, spread=0.005, reliable=True)

        result = search(seeds(), SPACE, fast_then_stop, SCREEN, FULL)
        assert result.interrupted
        assert result.best is None


class TestBaseline:
    def test_the_baseline_is_measured_at_full_effort_and_reported(self):
        landscape = Landscape()
        result = search(seeds(), SPACE, landscape, SCREEN, FULL, baseline=config(n_gpu_layers=37))
        assert result.baseline is not None
        assert (config(n_gpu_layers=37), FULL) in landscape.asked
        assert result.speedup > 1.0

    def test_the_baseline_is_measured_next_to_the_winner_not_first(self):
        # Measured first, the baseline was the coldest reading and inflated the speedup.
        result = search(seeds(), SPACE, Landscape(), SCREEN, FULL, baseline=config(n_gpu_layers=37))
        stages = [s.stage for s in result.steps]
        assert stages.index(Stage.BASELINE) > stages.index(Stage.CONFIRM)

    def test_losing_to_the_baseline_is_reported_not_hidden(self):
        # A flat landscape the tuner cannot climb, with a baseline that already wins.
        winner = config(n_gpu_layers=30)

        def flat(cfg: Config, effort: Effort) -> Trial:
            score = 20.0 if cfg == winner else 10.0
            return Trial(cfg, score=score, spread=0.005, reliable=True)

        result = search(seeds((10, 12)), SPACE, flat, SCREEN, FULL, baseline=winner)
        assert result.speedup == 0.5
        assert "search finished" in result.reason

    def test_without_a_baseline_there_is_no_speedup_to_claim(self):
        assert search(seeds(), SPACE, Landscape(), SCREEN, FULL).speedup is None


class TestFailures:
    def test_a_configuration_that_will_not_run_is_skipped(self):
        result = search(seeds(), SPACE, Landscape(fails={30}), SCREEN, FULL)
        assert result.best is not None
        assert any(s.verdict is Verdict.FAILED for s in result.steps)

    def test_every_seed_failing_yields_no_best_and_says_why(self):
        result = search(seeds((30,)), SPACE, Landscape(fails={30}), SCREEN, FULL)
        assert result.best is None
        assert "no seed configuration produced a measurement" in result.reason

    def test_an_empty_seed_list_is_reported_rather_than_crashing(self):
        result = search([], SPACE, Landscape(), SCREEN, FULL)
        assert result.best is None
        assert result.reason


class TestInterrupt:
    def test_stopping_early_keeps_the_best_result_so_far(self):
        landscape = Landscape()

        def impatient(cfg: Config, effort: Effort) -> Trial:
            if len(landscape.asked) >= 4:
                raise KeyboardInterrupt
            return landscape(cfg, effort)

        result = search(seeds(), SPACE, impatient, SCREEN, FULL)
        assert result.interrupted
        assert result.best is not None
        assert "interrupted" in result.reason

    def test_an_interrupt_before_any_measurement_yields_nothing(self):
        def instant(cfg: Config, effort: Effort) -> Trial:
            raise KeyboardInterrupt

        result = search(seeds(), SPACE, instant, SCREEN, FULL)
        assert result.interrupted
        assert result.best is None


@pytest.mark.parametrize("peak", [8, 16, 24, 32])
def test_the_search_finds_the_peak_wherever_it_sits(monkeypatch, peak):
    monkeypatch.setattr("tests.test_tune.PEAK_LAYERS", peak, raising=False)

    class Shifted(Landscape):
        def __call__(self, cfg: Config, effort: Effort) -> Trial:
            self.asked.append((cfg, effort))
            score = 30.0 - abs(peak - (cfg.n_gpu_layers or 0)) * 0.8
            return Trial(cfg, score=score, spread=0.005, reliable=True)

    result = search(seeds((4, 12, 20, 28, 36)), SPACE, Shifted(), SCREEN, FULL)
    assert result.best.config.n_gpu_layers == peak
