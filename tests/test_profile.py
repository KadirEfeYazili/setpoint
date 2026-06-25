"""Profile format tests.

Two rules from the spec carry the format's credibility and are pinned hardest: an
unreliable measurement cannot be written, and the `reliable` flag in a file is never
trusted on the way back in.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from setpoint import profile as prof
from setpoint.measure import Statistic
from setpoint.profile import store
from setpoint.profile.signature import platform_tag, signature_id

TIGHT = Statistic((10.6, 10.77, 10.76, 10.78, 10.84))
NOISY = Statistic((10.0, 14.0, 10.0, 15.0, 10.0))


def signature(**overrides) -> prof.Signature:
    fields = {
        "model_digest": "sha256:" + "a" * 64,
        "model_digest_kind": prof.DIGEST_HEADER,
        "model_size_bytes": 5225374496,
        "gpu": "NVIDIA GeForce GTX 1650",
        "vram_total_mb": 4096,
        "driver": "512.89",
        "backend": "llama.cpp b7412 (Vulkan)",
        "platform": "windows/amd64",
    }
    fields.update(overrides)
    return prof.Signature(**fields)


def measurement(decode: Statistic = TIGHT, **overrides) -> prof.Measurement:
    return prof.Measurement.from_statistics(
        decode=decode,
        measured_at="2026-09-07T21:14:31Z",
        prefill=Statistic((360.4, 361.0, 359.8, 362.4, 360.1)),
        **overrides,
    )


def profile(**overrides) -> prof.Profile:
    fields = {
        "signature": signature(),
        "target": prof.Target(context=8192, optimize=prof.Objective.SPEED),
        "config": prof.Config(n_gpu_layers=24, flash_attn=True, threads=6),
        "measurement": measurement(),
        "baseline": prof.Baseline(
            label="auto (-ngl 99)",
            config=prof.Config(n_gpu_layers=99),
            decode_tok_s=3.59,
            speedup=3.0,
        ),
        "created": "2026-09-07T21:14:31Z",
    }
    fields.update(overrides)
    return prof.Profile(**fields)


class TestEvidenceRule:
    def test_a_noisy_measurement_cannot_back_a_profile(self, tmp_path):
        noisy = profile(measurement=measurement(decode=NOISY))
        assert not noisy.writable
        with pytest.raises(prof.ProfileError, match="spread"):
            store.save(noisy, tmp_path)
        assert list(tmp_path.glob("*.yaml")) == []

    def test_too_few_runs_cannot_back_a_profile(self, tmp_path):
        thin = profile(measurement=measurement(decode=Statistic((10.0, 10.0))))
        assert not thin.writable
        with pytest.raises(prof.ProfileError, match="runs"):
            store.save(thin, tmp_path)

    def test_a_noisy_prefill_disqualifies_a_tight_decode(self):
        mixed = prof.Measurement.from_statistics(
            decode=TIGHT, measured_at="2026-09-07T21:14:31Z", prefill=NOISY
        )
        assert not mixed.reliable

    def test_a_slower_than_baseline_result_is_still_written(self, tmp_path):
        # A negative result is a result. Only unreliable measurements are refused.
        lost = profile(
            baseline=prof.Baseline("auto", prof.Config(n_gpu_layers=99), 20.0, 0.54),
        )
        path = store.save(lost, tmp_path)
        assert store.load(path).baseline.speedup == 0.54
        assert not store.load(path).baseline.improved


class TestRoundTrip:
    def test_a_profile_survives_a_write_and_a_read(self, tmp_path):
        original = profile()
        assert store.load(store.save(original, tmp_path)) == original

    def test_the_schema_is_the_first_field(self, tmp_path):
        text = store.dumps(profile())
        assert text.splitlines()[0] == "schema: setpoint/v1"

    def test_notes_are_kept_when_present_and_omitted_when_not(self, tmp_path):
        assert "notes" not in yaml.safe_load(store.dumps(profile()))
        annotated = profile(notes=("measured while the display was idle",))
        assert store.load(store.save(annotated, tmp_path)).notes == annotated.notes

    def test_unset_config_entries_stay_unset_rather_than_becoming_zero(self, tmp_path):
        sparse = profile(config=prof.Config(n_gpu_layers=24))
        restored = store.load(store.save(sparse, tmp_path))
        assert restored.config.n_cpu_moe is None
        assert restored.config.threads is None

    def test_mean_and_standard_deviation_are_not_stored(self):
        payload = yaml.safe_load(store.dumps(profile()))
        assert set(payload["measurement"]["decode_tok_s"]) == {"median", "iqr", "spread"}


class TestReadingRules:
    def _write(self, tmp_path: Path, mutate) -> Path:
        payload = yaml.safe_load(store.dumps(profile()))
        mutate(payload)
        path = tmp_path / "hand-edited.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def test_an_unknown_schema_is_refused(self, tmp_path):
        path = self._write(tmp_path, lambda p: p.update(schema="setpoint/v2"))
        with pytest.raises(prof.ProfileError, match="schema"):
            store.load(path)

    def test_a_reliable_flag_cannot_be_edited_into_truth(self, tmp_path):
        # The flag in the file is a claim; the numbers next to it are the fact.
        def widen(payload):
            payload["measurement"]["decode_tok_s"]["spread"] = 0.4
            payload["measurement"]["reliable"] = True

        with pytest.raises(prof.ProfileError, match="spread"):
            store.load(self._write(tmp_path, widen))

    def test_a_missing_required_field_is_refused(self, tmp_path):
        path = self._write(tmp_path, lambda p: p["signature"].pop("driver"))
        with pytest.raises(prof.ProfileError, match="driver"):
            store.load(path)

    def test_an_unknown_optimize_target_is_refused(self, tmp_path):
        path = self._write(tmp_path, lambda p: p["target"].update(optimize="vibes"))
        with pytest.raises(prof.ProfileError, match="optimize"):
            store.load(path)

    def test_unknown_fields_are_ignored_for_forward_compatibility(self, tmp_path):
        path = self._write(tmp_path, lambda p: p.update(future_field={"added": "later"}))
        assert store.load(path).target.context == 8192

    def test_a_file_that_is_not_a_profile_is_refused(self, tmp_path):
        path = tmp_path / "junk.yaml"
        path.write_text("just a string\n", encoding="utf-8")
        with pytest.raises(prof.ProfileError):
            store.load(path)


class TestLookup:
    def test_a_profile_is_found_by_its_exact_signature(self, tmp_path):
        store.save(profile(), tmp_path)
        assert store.find(signature(), tmp_path) is not None

    def test_a_driver_update_invalidates_the_profile(self, tmp_path):
        store.save(profile(), tmp_path)
        assert store.find(signature(driver="580.10"), tmp_path) is None

    def test_a_different_backend_build_invalidates_the_profile(self, tmp_path):
        store.save(profile(), tmp_path)
        assert store.find(signature(backend="llama.cpp b7500 (CUDA)"), tmp_path) is None

    def test_a_different_model_invalidates_the_profile(self, tmp_path):
        store.save(profile(), tmp_path)
        assert store.find(signature(model_digest="sha256:" + "b" * 64), tmp_path) is None

    def test_the_same_world_always_yields_the_same_id(self):
        assert signature_id(signature()) == signature_id(signature())
        assert signature_id(signature()) != signature_id(signature(driver="580.10"))

    def test_listing_skips_files_it_cannot_read(self, tmp_path):
        store.save(profile(), tmp_path)
        (tmp_path / "broken.yaml").write_text("schema: setpoint/v9\n", encoding="utf-8")
        assert len(store.load_all(tmp_path)) == 1

    def test_an_empty_directory_lists_nothing(self, tmp_path):
        assert store.load_all(tmp_path / "nothing-here") == []


class TestLocation:
    def test_the_home_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv(prof.HOME_ENV_VAR, str(tmp_path))
        assert prof.profiles_dir() == tmp_path / "profiles"

    def test_the_platform_tag_is_lowercase_and_normalised(self):
        tag = platform_tag()
        assert tag == tag.lower()
        assert "/" in tag
