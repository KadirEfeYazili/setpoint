"""Profile sharing tests.

A profile is a claim about one machine. Sharing one is only useful when the receiving
machine is the same machine in every way that mattered, so the tests that matter are
the refusals: wrong hardware, and a model this machine does not have.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from setpoint.measure import Statistic
from setpoint.model import ResolvedModel
from setpoint.profile import (
    Baseline,
    Config,
    MachineFacts,
    Measurement,
    ModelRef,
    Objective,
    Profile,
    Signature,
    Target,
    adopt,
    describes_machine,
    find_model,
    for_sharing,
    machine_mismatch,
)

HERE = MachineFacts(
    gpu="NVIDIA GeForce GTX 1650",
    vram_total_mb=4096,
    driver="512.89",
    platform="windows/amd64",
)


def signature(**overrides) -> Signature:
    fields = {
        "model_digest": "sha256:" + "a" * 64,
        "model_digest_kind": "header",
        "model_size_bytes": 1929903008,
        "gpu": HERE.gpu,
        "vram_total_mb": HERE.vram_total_mb,
        "driver": HERE.driver,
        "backend": "llama.cpp b10850",
        "platform": HERE.platform,
    }
    fields.update(overrides)
    return Signature(**fields)


def profile(**overrides) -> Profile:
    fields = {
        "signature": signature(),
        "model": ModelRef("Qwen2.5 3B", "qwen2", "Q4_K_M", r"C:\someone-else\model.gguf"),
        "target": Target(context=1024, optimize=Objective.SPEED),
        "config": Config(n_gpu_layers=37),
        "measurement": Measurement.from_statistics(
            decode=Statistic((52.6, 52.77, 52.77, 52.8, 52.9)),
            measured_at="2026-09-08T18:34:19Z",
        ),
        "baseline": Baseline("default", Config(n_gpu_layers=99), 50.77, 1.04),
        "created": "2026-09-08T18:34:19Z",
    }
    fields.update(overrides)
    return Profile(**fields)


class TestWhatLeaves:
    def test_the_file_path_is_removed(self):
        # It names a directory on someone's machine and means nothing on another.
        assert for_sharing(profile()).model.path is None

    def test_what_identifies_the_model_stays(self):
        shared = for_sharing(profile())
        assert shared.signature.model_digest == profile().signature.model_digest
        assert shared.model.name == "Qwen2.5 3B"
        assert shared.model.file_type == "Q4_K_M"

    def test_the_measurement_is_untouched(self):
        shared = for_sharing(profile())
        assert shared.measurement == profile().measurement
        assert shared.baseline == profile().baseline

    def test_sharing_does_not_change_the_original(self):
        original = profile()
        for_sharing(original)
        assert original.model.path is not None


class TestWhoItApplies:
    def test_the_same_machine_is_recognised(self):
        assert describes_machine(signature(), HERE)
        assert machine_mismatch(signature(), HERE) == ()

    @pytest.mark.parametrize(
        ("field", "value", "word"),
        [
            ("gpu", "NVIDIA GeForce RTX 4090", "GPU"),
            ("vram_total_mb", 24564, "VRAM"),
            ("driver", "580.10", "driver"),
            ("platform", "linux/amd64", "platform"),
        ],
    )
    def test_each_difference_is_named(self, field, value, word):
        # "It is for other hardware" is not useful; which field differs is.
        differences = machine_mismatch(signature(**{field: value}), HERE)
        assert len(differences) == 1
        assert differences[0].startswith(word)
        assert not describes_machine(signature(**{field: value}), HERE)

    def test_several_differences_are_all_reported(self):
        other = signature(gpu="NVIDIA GeForce RTX 4090", driver="580.10")
        assert len(machine_mismatch(other, HERE)) == 2

    def test_the_backend_build_is_not_part_of_the_machine_check(self):
        # Only a run reports it, and an importer has not run anything yet.
        assert describes_machine(signature(backend="llama.cpp b99999"), HERE)


class TestFindingTheModel:
    def _candidate(self, tmp_path: Path, name: str, body: bytes) -> ResolvedModel:
        path = tmp_path / name
        path.write_bytes(body)
        return ResolvedModel(path, name, "path")

    def test_it_matches_on_the_digest_not_the_name(self, tmp_path, monkeypatch):
        wanted = self._candidate(tmp_path, "renamed.gguf", b"x")
        other = self._candidate(tmp_path, "model.gguf", b"y")
        monkeypatch.setattr(
            "setpoint.profile.share.model_digest",
            lambda path, kind: (
                ("sha256:" + "a" * 64, 1929903008)
                if Path(path).name == "renamed.gguf"
                else ("sha256:" + "b" * 64, 5)
            ),
        )
        assert find_model(profile(), [other, wanted]) is wanted

    def test_a_matching_digest_with_a_different_size_is_not_it(self, tmp_path, monkeypatch):
        candidate = self._candidate(tmp_path, "model.gguf", b"x")
        monkeypatch.setattr(
            "setpoint.profile.share.model_digest",
            lambda path, kind: ("sha256:" + "a" * 64, 999),
        )
        assert find_model(profile(), [candidate]) is None

    def test_an_unreadable_candidate_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        candidate = self._candidate(tmp_path, "broken.gguf", b"x")

        def explode(path, kind):
            raise OSError("unreadable")

        monkeypatch.setattr("setpoint.profile.share.model_digest", explode)
        assert find_model(profile(), [candidate]) is None

    def test_nothing_local_means_nothing_found(self):
        assert find_model(profile(), []) is None


class TestAdoption:
    def test_the_local_path_is_attached(self, tmp_path):
        adopted = adopt(for_sharing(profile()), tmp_path / "local.gguf")
        assert adopted.model.path == str(tmp_path / "local.gguf")

    def test_what_the_sender_knew_is_kept(self, tmp_path):
        adopted = adopt(for_sharing(profile()), tmp_path / "local.gguf")
        assert adopted.model.name == "Qwen2.5 3B"
        assert adopted.model.file_type == "Q4_K_M"

    def test_an_unreadable_file_still_yields_a_usable_profile(self, tmp_path):
        # analyze() failing must not lose the profile; the path is what was needed.
        adopted = adopt(for_sharing(profile()), tmp_path / "not-a-gguf")
        assert adopted.model.path.endswith("not-a-gguf")
        assert adopted.measurement == profile().measurement
