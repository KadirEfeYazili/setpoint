"""Runner configuration tests.

The output is a file another program parses, so the tests that matter most are that it
parses at all and that the evidence survives. A configuration that a runner rejects is
worse than no configuration, and one whose measurement has been stripped is
indistinguishable from the guess setpoint exists to replace.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from setpoint import export
from setpoint.measure import Statistic
from setpoint.profile import (
    Baseline,
    Config,
    Measurement,
    ModelRef,
    Objective,
    Profile,
    Signature,
    Target,
)

TIGHT = Statistic((52.6, 52.77, 52.77, 52.8, 52.9))


def profile(
    name: str = "Qwen2.5 3B Instruct",
    context: int = 1024,
    peak_mb: int | None = 2560,
    path: str = "C:/models/qwen.gguf",
    failed_baseline: bool = False,
) -> Profile:
    baseline = (
        Baseline("llama.cpp default", Config(n_gpu_layers=99), failed=True, detail="out of memory")
        if failed_baseline
        else Baseline("llama.cpp default", Config(n_gpu_layers=99), 50.77, 1.04)
    )
    return Profile(
        signature=Signature(
            model_digest="sha256:" + "a" * 64,
            model_digest_kind="header",
            model_size_bytes=1,
            gpu="NVIDIA GeForce GTX 1650",
            vram_total_mb=4096,
            driver="512.89",
            backend="llama.cpp b10850",
            platform="windows/amd64",
        ),
        model=ModelRef(name, "qwen2", "Q4_K_M", path),
        target=Target(context=context, optimize=Objective.SPEED),
        config=Config(n_gpu_layers=37, ubatch_size=128, flash_attn=True),
        measurement=Measurement.from_statistics(
            decode=TIGHT, measured_at="2026-09-08T18:34:19Z", peak_vram_mb=peak_mb
        ),
        baseline=baseline,
        created="2026-09-08T18:34:19Z",
    )


def rendered(*profiles: Profile, **kwargs) -> str:
    entries = export.build_entries(list(profiles), vram_total_mb=4096, **kwargs)
    return export.render(entries, ["header line"])


class TestItParses:
    def test_the_output_is_valid_yaml(self):
        data = yaml.safe_load(rendered(profile()))
        assert "models" in data

    def test_a_windows_path_does_not_break_it(self):
        # A quoted scalar would read the backslashes as escapes and fail to parse.
        text = rendered(profile(path=r"C:\Users\x\.ollama\blobs\sha256-abc"))
        data = yaml.safe_load(text)
        cmd = next(iter(data["models"].values()))["cmd"]
        assert r"C:\Users\x\.ollama\blobs\sha256-abc" in cmd

    def test_a_path_with_a_space_survives_both_layers(self):
        text = rendered(profile(path=r"C:\Program Files\models\m.gguf"))
        cmd = yaml.safe_load(text)["models"]["qwen2.5-3b-instruct-c1024"]["cmd"]
        assert '"C:\\Program Files\\models\\m.gguf"' in cmd

    def test_the_port_placeholder_is_left_for_the_runner(self):
        cmd = yaml.safe_load(rendered(profile()))["models"]["qwen2.5-3b-instruct-c1024"]["cmd"]
        assert export.PORT_PLACEHOLDER in cmd

    def test_an_empty_config_still_parses_and_says_what_to_do(self):
        text = export.render([], ["header"])
        assert yaml.safe_load(text) == {"models": None}
        assert "setpoint tune" in text


class TestEvidenceSurvives:
    def test_the_measurement_is_in_the_comments(self):
        text = rendered(profile())
        assert "52.77 tok/s" in text
        assert "2026-09-08T18:34:19Z" in text

    def test_the_headline_reaches_a_field_the_runner_can_show(self):
        data = yaml.safe_load(rendered(profile()))
        entry = data["models"]["qwen2.5-3b-instruct-c1024"]
        assert "52.77 tok/s" in entry["description"]
        assert "1.04x" in entry["description"]

    def test_a_baseline_that_never_started_is_said_plainly(self):
        text = rendered(profile(failed_baseline=True))
        assert "does not start on this card" in text
        assert "1.04x" not in text

    def test_the_human_label_carries_the_context(self):
        entry = yaml.safe_load(rendered(profile()))["models"]["qwen2.5-3b-instruct-c1024"]
        assert entry["name"] == "Qwen2.5 3B Instruct (1024 ctx)"


class TestNaming:
    def test_dots_survive_the_slug(self):
        assert export.slug("Qwen2.5 3B") == "qwen2.5-3b"

    def test_the_context_is_part_of_the_name(self):
        assert export.entry_name(profile(context=8192)).endswith("-c8192")

    def test_a_nameless_model_falls_back_to_its_architecture(self):
        assert export.entry_name(profile(name="")).startswith("qwen2")

    def test_one_context_gets_a_bare_alias(self):
        entry = yaml.safe_load(rendered(profile()))["models"]["qwen2.5-3b-instruct-c1024"]
        assert entry["aliases"] == ["qwen2.5-3b-instruct"]

    def test_several_contexts_get_no_bare_alias(self):
        # A bare name would have to pick one context, and picking silently is wrong.
        data = yaml.safe_load(rendered(profile(context=1024), profile(context=8192)))
        assert len(data["models"]) == 2
        assert all("aliases" not in entry for entry in data["models"].values())


class TestUnloadPolicy:
    def test_a_model_holding_most_of_the_card_yields_soonest(self):
        hungry = export.ttl_for(profile(peak_mb=3900), 4096)
        modest = export.ttl_for(profile(peak_mb=2000), 4096)
        tiny = export.ttl_for(profile(peak_mb=500), 4096)
        assert hungry < modest < tiny

    def test_an_unknown_peak_keeps_the_model_longest(self):
        assert export.ttl_for(profile(peak_mb=None), 4096) == export.TTL_BANDS[-1][1]

    def test_an_unknown_card_size_keeps_the_model_longest(self):
        assert export.ttl_for(profile(), None) == export.TTL_BANDS[-1][1]

    def test_an_override_beats_the_policy(self):
        data = yaml.safe_load(rendered(profile(peak_mb=3900), ttl_override=42))
        assert next(iter(data["models"].values()))["ttl"] == 42


class TestSleepPolicy:
    def test_a_model_holding_most_of_the_card_releases_it_soonest(self):
        hungry = export.sleep_for(profile(peak_mb=3900), 4096, ttl=3600)
        modest = export.sleep_for(profile(peak_mb=2000), 4096, ttl=3600)
        tiny = export.sleep_for(profile(peak_mb=500), 4096, ttl=3600)
        assert hungry < modest < tiny

    def test_sleeping_always_happens_before_the_unload(self):
        # Past the TTL the runner takes the process away, so sleeping first would have
        # bought nothing.
        assert export.sleep_for(profile(peak_mb=500), 4096, ttl=30) < 30

    def test_an_unknown_peak_holds_the_vram_longest(self):
        assert export.sleep_for(profile(peak_mb=None), 4096, ttl=3600) == export.SLEEP_BANDS[-1][1]

    def test_the_flag_reaches_the_command_with_the_reason_beside_it(self):
        text = rendered(profile(peak_mb=3900))
        assert "--sleep-idle-seconds" in text
        assert "releases its VRAM after" in text

    def test_zero_turns_it_off_rather_than_sleeping_instantly(self):
        assert "--sleep-idle-seconds" not in rendered(profile(), sleep_override=0)

    def test_an_override_beats_the_policy(self):
        assert "--sleep-idle-seconds 42" in rendered(profile(peak_mb=3900), sleep_override=42)

    def test_the_preset_target_carries_it_too(self):
        entries = export.build_entries([profile(peak_mb=3900)], vram_total_mb=4096)
        text = export.llamaserver.render(entries)
        assert "sleep-idle-seconds = " in text


class TestDevicePinning:
    def test_a_named_device_reaches_the_command(self):
        text = rendered(profile(), devices=("Vulkan0",))
        assert "--device Vulkan0" in text

    def test_no_device_named_means_none_pinned(self):
        assert "--device" not in rendered(profile())


class TestWrittenFile:
    def test_it_round_trips_through_disk(self, tmp_path: Path):
        target = tmp_path / "config.yaml"
        target.write_text(rendered(profile()), encoding="utf-8")
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert "qwen2.5-3b-instruct-c1024" in data["models"]

    @pytest.mark.parametrize("context", [256, 1024, 8192, 131072])
    def test_every_context_yields_a_parseable_entry(self, context):
        data = yaml.safe_load(rendered(profile(context=context)))
        assert len(data["models"]) == 1
