"""llama.cpp adapter tests.

Command line construction and output parsing are pure, so they are covered here without
llama-bench installed. `run` is covered by standing in for the subprocess, which leaves
only the real binary's behaviour unverified.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from setpoint.backend import (
    BackendError,
    LlamaCppBackend,
    MeasurementKind,
    RunSpec,
    parse_devices,
    parse_output,
)
from setpoint.backend.llamacpp import BINARY_ENV_VAR, find_binary

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "llama_bench_run.json"
DEVICES = FIXTURES / "llama_bench_devices.txt"


@pytest.fixture
def output() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture
def backend(tmp_path, monkeypatch) -> LlamaCppBackend:
    binary = tmp_path / "llama-bench"
    binary.write_text("", encoding="utf-8")
    monkeypatch.setenv(BINARY_ENV_VAR, str(binary))
    return LlamaCppBackend()


def spec(**overrides) -> RunSpec:
    fields = {"model_path": Path("model.gguf"), "n_gpu_layers": 24, "n_depth": 8192}
    fields.update(overrides)
    return RunSpec(**fields)


class TestDiscovery:
    def test_the_environment_variable_wins(self, tmp_path, monkeypatch):
        binary = tmp_path / "llama-bench"
        binary.write_text("", encoding="utf-8")
        monkeypatch.setenv(BINARY_ENV_VAR, str(binary))
        assert find_binary() == binary

    def test_a_path_that_does_not_exist_resolves_to_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv(BINARY_ENV_VAR, str(tmp_path / "missing"))
        assert find_binary() is None

    def test_a_missing_binary_makes_the_backend_unavailable(self, monkeypatch):
        monkeypatch.delenv(BINARY_ENV_VAR, raising=False)
        monkeypatch.setattr("setpoint.backend.llamacpp.shutil.which", lambda _: None)
        assert not LlamaCppBackend().available

    def test_building_a_command_without_a_binary_is_an_error(self, monkeypatch):
        monkeypatch.delenv(BINARY_ENV_VAR, raising=False)
        monkeypatch.setattr("setpoint.backend.llamacpp.shutil.which", lambda _: None)
        with pytest.raises(BackendError):
            LlamaCppBackend().build_argv(spec())


class TestCommandLine:
    def test_it_asks_for_json_and_repeats(self, backend):
        argv = backend.build_argv(spec(repetitions=7))
        assert argv[argv.index("-o") + 1] == "json"
        assert argv[argv.index("-r") + 1] == "7"

    def test_warmup_is_never_skipped(self, backend):
        # The first pass of a configuration is not a measurement.
        assert "--no-warmup" not in backend.build_argv(spec())

    def test_the_target_context_becomes_depth(self, backend):
        argv = backend.build_argv(spec(n_depth=16384))
        assert argv[argv.index("-d") + 1] == "16384"

    def test_depth_is_omitted_when_it_is_zero(self, backend):
        assert "-d" not in backend.build_argv(spec(n_depth=0))

    def test_flash_attention_uses_the_modern_spelling(self, backend):
        assert backend.build_argv(spec(flash_attn=True))[-1] == "on"
        assert backend.build_argv(spec(flash_attn=False))[-1] == "off"

    def test_flash_attention_is_left_to_the_backend_when_unset(self, backend):
        assert "-fa" not in backend.build_argv(spec(flash_attn=None))

    def test_unset_options_are_left_out_rather_than_defaulted(self, backend):
        argv = backend.build_argv(spec(batch_size=None, threads=None, n_cpu_moe=None))
        assert "-b" not in argv
        assert "-t" not in argv
        assert "-ncmoe" not in argv

    def test_every_tensor_override_is_passed(self, backend):
        argv = backend.build_argv(spec(tensor_overrides=("exps=CPU", "attn=CUDA0")))
        assert argv.count("-ot") == 2
        assert "exps=CPU" in argv

    def test_a_named_device_is_pinned(self, backend):
        argv = backend.build_argv(spec(devices=("Vulkan1",)))
        assert argv[argv.index("-dev") + 1] == "Vulkan1"

    def test_several_devices_are_joined_the_way_the_backend_expects(self, backend):
        argv = backend.build_argv(spec(devices=("CUDA0", "CUDA1")))
        assert argv[argv.index("-dev") + 1] == "CUDA0/CUDA1"

    def test_no_device_is_pinned_when_none_was_named(self, backend):
        assert "-dev" not in backend.build_argv(spec())

    def test_cache_types_are_always_explicit(self, backend):
        argv = backend.build_argv(spec(cache_type_k="q8_0", cache_type_v="q8_0"))
        assert argv[argv.index("-ctk") + 1] == "q8_0"
        assert argv[argv.index("-ctv") + 1] == "q8_0"


class TestParsing:
    def test_prefill_and_decode_are_told_apart(self, output):
        run = parse_output(output, spec())
        assert {s.kind for s in run.samples} == {MeasurementKind.PREFILL, MeasurementKind.DECODE}
        assert run.sample_of(MeasurementKind.DECODE).n_gen == 128
        assert run.sample_of(MeasurementKind.PREFILL).n_prompt == 512

    def test_the_median_comes_from_the_repetitions_not_the_reported_mean(self, output):
        decode = parse_output(output, spec()).sample_of(MeasurementKind.DECODE)
        assert decode.throughput.runs == 5
        assert decode.tokens_per_second == 10.77
        assert decode.reported_mean_ts == 10.76
        assert decode.throughput.median != decode.throughput.samples[0]

    def test_a_tight_run_is_reliable(self, output):
        assert parse_output(output, spec()).reliable

    def test_the_build_identifies_itself(self, output):
        build = parse_output(output, spec()).build
        assert build.number == 7412
        assert build.commit == "1f2e3d4c"
        assert build.accelerators == "CUDA"
        assert "b7412" in str(build)

    def test_noise_around_the_json_is_ignored(self, output):
        run = parse_output(f"loading model...\n{output}\nllama_perf: done\n", spec())
        assert len(run.samples) == 2

    def test_missing_repetitions_fall_back_and_say_so(self, output):
        records = json.loads(output)
        for record in records:
            record.pop("samples_ts")
            record.pop("samples_ns")
        run = parse_output(json.dumps(records), spec())
        assert not run.reliable
        assert any("per-repetition" in note for note in run.notes)

    def test_running_on_a_different_device_than_asked_for_is_flagged(self, output):
        # A measurement filed under the wrong card is worse than no measurement.
        records = json.loads(output)
        for record in records:
            record["devices"] = "Vulkan0"
        run = parse_output(json.dumps(records), spec(devices=("Vulkan1",)))
        assert any("may not describe the hardware" in note for note in run.notes)

    def test_matching_devices_raise_no_complaint(self, output):
        records = json.loads(output)
        for record in records:
            record["devices"] = "Vulkan1"
        run = parse_output(json.dumps(records), spec(devices=("Vulkan1",)))
        assert run.notes == ()
        assert run.devices == "Vulkan1"

    def test_an_old_build_is_flagged(self, output):
        records = json.loads(output)
        for record in records:
            record["build_number"] = 3000
        assert any("older than" in note for note in parse_output(json.dumps(records), spec()).notes)

    def test_output_without_json_is_an_error(self):
        with pytest.raises(BackendError):
            parse_output("could not load model\n", spec())

    def test_broken_json_is_an_error(self):
        with pytest.raises(BackendError):
            parse_output("[{oops}]", spec())

    def test_an_empty_result_list_is_an_error(self):
        with pytest.raises(BackendError):
            parse_output("[]", spec())


class TestRun:
    def _stub(self, monkeypatch, **result):
        captured: dict[str, object] = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            if "raises" in result:
                raise result["raises"]
            return subprocess.CompletedProcess(
                argv,
                result.get("returncode", 0),
                result.get("stdout", ""),
                result.get("stderr", ""),
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        return captured

    def test_a_successful_run_is_parsed_and_timed(self, backend, monkeypatch, output):
        captured = self._stub(monkeypatch, stdout=output)
        run = backend.run(spec())
        assert run.decode_tokens_per_second == 10.77
        assert run.duration_s is not None
        assert run.command == tuple(captured["argv"])

    def test_a_failing_run_surfaces_the_backend_message(self, backend, monkeypatch):
        self._stub(monkeypatch, returncode=1, stderr="error: failed to load model\n")
        with pytest.raises(BackendError, match="failed to load model"):
            backend.run(spec())

    def test_a_timeout_is_reported_as_one(self, backend, monkeypatch):
        self._stub(monkeypatch, raises=subprocess.TimeoutExpired("llama-bench", 1.0))
        with pytest.raises(BackendError, match="did not finish"):
            backend.run(spec(), timeout_s=1.0)

    def test_a_binary_that_will_not_start_is_reported(self, backend, monkeypatch):
        self._stub(monkeypatch, raises=OSError("permission denied"))
        with pytest.raises(BackendError, match="could not be started"):
            backend.run(spec())


class TestDeviceListing:
    """Parsed from output a real llama-bench produced, not from a guess at its shape."""

    @pytest.fixture
    def listing(self) -> str:
        return DEVICES.read_text(encoding="utf-8")

    def test_both_accelerators_are_found(self, listing):
        devices = parse_devices(listing)
        assert [d.id for d in devices] == ["Vulkan0", "Vulkan1"]
        assert devices[1].name == "NVIDIA GeForce GTX 1650"

    def test_memory_is_read_where_the_backend_reports_it(self, listing):
        discrete = parse_devices(listing)[1]
        assert discrete.total_mib == 4176
        assert discrete.free_mib == 3581

    def test_the_backend_kind_comes_off_the_device_id(self, listing):
        assert {d.kind for d in parse_devices(listing)} == {"Vulkan"}

    def test_the_chatter_before_the_listing_is_ignored(self, listing):
        # Every line before "Available devices:" mentions devices too.
        assert len(parse_devices(listing)) == 2

    def test_a_device_without_a_memory_report_still_parses(self):
        devices = parse_devices("Available devices:\n  CPU: 12th Gen Intel Core i5\n")
        assert devices[0].id == "CPU"
        assert devices[0].total_mib is None

    def test_output_without_a_listing_yields_nothing(self):
        assert parse_devices("error: no backends loaded\n") == ()
