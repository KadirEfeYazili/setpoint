"""Tests for the piece that joins the search to the machine.

The backend and NVML are stood in for, so what is checked is the wiring: that a
configuration becomes the right command, that each objective scores what it claims to
score, that a run taken on a throttling card is labelled rather than silently kept, and
that losing the GPU sampler never costs the measurement it accompanied.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from setpoint.backend import BackendBuild, BackendError, BenchRun, BenchSample, MeasurementKind
from setpoint.hardware.watch import GpuWatch
from setpoint.measure import Statistic
from setpoint.profile import Config, Objective
from setpoint.tune import Effort
from setpoint.tune.measure import BackendMeasure, run_spec

MODEL = Path("model.gguf")
EFFORT = Effort(repetitions=5, n_depth=8192, label="full")
MIB = 1024 * 1024


def bench_run(decode: float = 12.0, prefill: float = 300.0, spread: float = 0.004) -> BenchRun:
    def stat(value: float) -> Statistic:
        step = value * spread / 2
        return Statistic((value - step, value, value, value, value + step))

    samples = (
        BenchSample(MeasurementKind.PREFILL, 512, 0, 8192, stat(prefill), stat(1e9)),
        BenchSample(MeasurementKind.DECODE, 0, 128, 8192, stat(decode), stat(1e10)),
    )
    return BenchRun(spec=None, build=BackendBuild("llama.cpp", number=10850), samples=samples)


class FakeBackend:
    def __init__(self, run: BenchRun | None = None, error: str | None = None) -> None:
        self.run_result = run if run is not None else bench_run()
        self.error = error
        self.specs = []

    def run(self, spec, **kwargs):
        self.specs.append(spec)
        if self.error:
            raise BackendError(self.error)
        return self.run_result


@pytest.fixture(autouse=True)
def _quiet_gpu(monkeypatch):
    """No NVML in the tests: a clean card, and a watcher that reports nothing."""
    monkeypatch.setattr("setpoint.tune.measure.read_throttle", lambda *a, **k: ())
    monkeypatch.setattr("setpoint.tune.measure.wait_until_settled", lambda *a, **k: ())
    monkeypatch.setattr("setpoint.tune.measure.GpuWatcher", _watcher(GpuWatch()))


def _watcher(result: GpuWatch):
    class Stub:
        def __init__(self, *_a, **_k) -> None:
            self.result = result

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    return Stub


def measurer(monkeypatch=None, watch: GpuWatch | None = None, **kwargs) -> BackendMeasure:
    if watch is not None and monkeypatch is not None:
        monkeypatch.setattr("setpoint.tune.measure.GpuWatcher", _watcher(watch))
    kwargs.setdefault("backend", FakeBackend())
    return BackendMeasure(model_path=MODEL, **kwargs)


class TestCommandMapping:
    def test_the_configuration_becomes_the_run(self):
        spec = run_spec(Config(n_gpu_layers=24, threads=6, ubatch_size=256), EFFORT, MODEL)
        assert spec.n_gpu_layers == 24
        assert spec.threads == 6
        assert spec.ubatch_size == 256

    def test_the_effort_sets_the_depth_and_the_repetitions(self):
        spec = run_spec(Config(), Effort(repetitions=2, n_depth=1024), MODEL)
        assert spec.n_depth == 1024
        assert spec.repetitions == 2

    def test_the_device_is_carried_through(self):
        spec = run_spec(Config(), EFFORT, MODEL, devices=("Vulkan1",))
        assert spec.devices == ("Vulkan1",)

    def test_the_backend_is_actually_asked(self):
        backend = FakeBackend()
        BackendMeasure(backend=backend, model_path=MODEL)(Config(n_gpu_layers=20), EFFORT)
        assert backend.specs[0].n_gpu_layers == 20


class TestScoring:
    def test_speed_scores_the_decode_rate(self):
        trial = measurer()(Config(), EFFORT)
        assert trial.score == pytest.approx(12.0)

    def test_latency_scores_the_prefill_rate(self):
        trial = measurer(objective=Objective.LATENCY)(Config(), EFFORT)
        assert trial.score == pytest.approx(300.0)

    def test_efficiency_divides_the_rate_by_the_power(self, monkeypatch):
        trial = measurer(
            monkeypatch, GpuWatch(samples=4, average_power_w=60.0), objective=Objective.EFFICIENCY
        )(Config(), EFFORT)
        assert trial.score == pytest.approx(0.2)

    def test_efficiency_without_a_power_reading_scores_nothing(self, monkeypatch):
        # Reporting a made-up efficiency is worse than reporting none.
        trial = measurer(monkeypatch, GpuWatch(samples=4), objective=Objective.EFFICIENCY)(
            Config(), EFFORT
        )
        assert trial.score is None
        assert "power could not be read" in trial.detail

    def test_headroom_rejects_a_configuration_that_overran_the_ceiling(self, monkeypatch):
        trial = measurer(
            monkeypatch,
            GpuWatch(samples=4, peak_vram_bytes=3600 * MIB),
            objective=Objective.HEADROOM,
            vram_ceiling_bytes=3000 * MIB,
        )(Config(), EFFORT)
        assert trial.score is None
        assert "past the ceiling" in trial.detail

    def test_headroom_accepts_a_configuration_that_stayed_inside_it(self, monkeypatch):
        trial = measurer(
            monkeypatch,
            GpuWatch(samples=4, peak_vram_bytes=2000 * MIB),
            objective=Objective.HEADROOM,
            vram_ceiling_bytes=3000 * MIB,
        )(Config(), EFFORT)
        assert trial.score == pytest.approx(12.0)

    def test_headroom_without_a_ceiling_says_it_enforced_nothing(self, monkeypatch):
        trial = measurer(
            monkeypatch, GpuWatch(samples=4, peak_vram_bytes=1), objective=Objective.HEADROOM
        )(Config(), EFFORT)
        assert trial.score == pytest.approx(12.0)
        assert "not enforced" in trial.detail

    def test_the_spread_reported_belongs_to_the_scored_side(self):
        speed = measurer()(Config(), EFFORT)
        latency = measurer(objective=Objective.LATENCY)(Config(), EFFORT)
        assert speed.spread is not None
        assert latency.spread is not None


class TestThermalState:
    def test_a_card_still_throttling_when_the_run_starts_is_recorded(self, monkeypatch):
        monkeypatch.setattr(
            "setpoint.tune.measure.wait_until_settled", lambda *a, **k: ("sw_thermal_slowdown",)
        )
        trial = measurer()(Config(), EFFORT)
        assert "throttling before the run" in trial.detail

    def test_throttling_during_the_run_disqualifies_the_result(self, monkeypatch):
        # The number may be real, but it does not describe an unthrottled machine.
        trial = measurer(
            monkeypatch, GpuWatch(samples=8, throttle_reasons=("hw_thermal_slowdown",))
        )(Config(), EFFORT)
        assert trial.score is not None
        assert not trial.reliable
        assert "throttled during the run" in trial.detail

    def test_a_clean_run_is_reliable(self):
        assert measurer()(Config(), EFFORT).reliable


class TestFailures:
    def test_a_backend_error_becomes_an_unusable_trial_not_an_exception(self):
        trial = BackendMeasure(backend=FakeBackend(error="out of memory"), model_path=MODEL)(
            Config(n_gpu_layers=99), EFFORT
        )
        assert not trial.usable
        assert "out of memory" in trial.detail

    def test_losing_the_gpu_sampler_does_not_lose_the_measurement(self, monkeypatch):
        trial = measurer(monkeypatch, GpuWatch(detail="no GPU samples were taken"))(
            Config(), EFFORT
        )
        assert trial.score == pytest.approx(12.0)
        assert "no GPU samples" in trial.detail

    def test_backend_notes_survive_into_the_trial(self):
        run = bench_run()
        noted = BenchRun(
            spec=run.spec, build=run.build, samples=run.samples, notes=("device mismatch",)
        )
        trial = BackendMeasure(backend=FakeBackend(noted), model_path=MODEL)(Config(), EFFORT)
        assert "device mismatch" in trial.detail


class TestFreeMemoryDrift:
    """Free VRAM moves while a desktop runs, and a plan drawn on a quiet card can fail."""

    def _with_free(self, monkeypatch, free_bytes: int | None) -> None:
        monkeypatch.setattr("setpoint.tune.measure.free_vram", lambda *a, **k: free_bytes)

    def test_a_material_drop_since_the_plan_is_reported(self, monkeypatch):
        self._with_free(monkeypatch, 3000 * MIB)
        trial = measurer(expected_free_bytes=3500 * MIB)(Config(), EFFORT)
        assert "500 MiB less free" in trial.detail

    def test_extra_room_is_reported_too(self, monkeypatch):
        self._with_free(monkeypatch, 3800 * MIB)
        trial = measurer(expected_free_bytes=3500 * MIB)(Config(), EFFORT)
        assert "300 MiB more free" in trial.detail

    def test_a_small_wobble_is_not_worth_mentioning(self, monkeypatch):
        self._with_free(monkeypatch, 3500 * MIB - 8 * MIB)
        trial = measurer(expected_free_bytes=3500 * MIB)(Config(), EFFORT)
        assert "free than" not in trial.detail

    def test_an_unreadable_reading_says_the_drift_is_unknown(self, monkeypatch):
        self._with_free(monkeypatch, None)
        trial = measurer(expected_free_bytes=3500 * MIB)(Config(), EFFORT)
        assert "drift since the plan is unknown" in trial.detail

    def test_without_an_expectation_nothing_is_checked(self, monkeypatch):
        self._with_free(monkeypatch, 1)
        assert "free" not in measurer()(Config(), EFFORT).detail
