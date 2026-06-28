"""Runtime allowance tests.

The constants in `overhead.py` were fitted to measurements taken on real hardware. What
these tests pin is the shape the measurements showed - linear in the microbatch, driven
by the vocabulary rather than the embedding width - so that a later change to the
formula has to argue with the readings rather than slip past them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from setpoint.budget import overhead
from setpoint.model.types import Attention, ModelInfo, Weights

MIB = 1024 * 1024


def model(vocab: int | None = 151936, embedding: int = 2048) -> ModelInfo:
    return ModelInfo(
        path=Path("model.gguf"),
        file_bytes=0,
        architecture="qwen2",
        name="Test",
        block_count=36,
        embedding_length=embedding,
        attention=Attention((16,) * 36, (2,) * 36, 128, 128),
        weights=Weights((0,) * 36, (0,) * 36, 0, 0),
        parameter_count=0,
        quant_mix=(),
        vocab_size=vocab,
    )


class TestShape:
    def test_it_grows_in_step_with_the_microbatch(self):
        small = overhead.runtime_allowance(model(), 128).total_bytes - overhead.BASE_BYTES
        large = overhead.runtime_allowance(model(), 512).total_bytes - overhead.BASE_BYTES
        assert large == pytest.approx(4 * small)

    def test_the_vocabulary_dominates_not_the_embedding_width(self):
        # Measured: doubling the embedding width left the per-token cost within 3%.
        narrow = overhead.runtime_allowance(model(embedding=2048), 512).total_bytes
        wide = overhead.runtime_allowance(model(embedding=4096), 512).total_bytes
        assert wide / narrow < 1.05

    def test_a_larger_vocabulary_costs_proportionally_more(self):
        small = overhead.runtime_allowance(model(vocab=32000), 512).logits_bytes
        large = overhead.runtime_allowance(model(vocab=64000), 512).logits_bytes
        assert large == 2 * small

    def test_the_parts_add_up_to_the_whole(self):
        allowance = overhead.runtime_allowance(model(), 512)
        assert (
            allowance.logits_bytes + allowance.graph_bytes + allowance.base_bytes
            == allowance.total_bytes
        )


class TestAgainstMeasurements:
    """Values read off a GTX 1650 running the Vulkan backend."""

    @pytest.mark.parametrize(("ubatch", "measured_mib"), [(128, 84), (256, 162), (512, 318)])
    def test_it_matches_what_the_card_actually_used(self, ubatch, measured_mib):
        predicted = overhead.runtime_allowance(model(), ubatch).total_bytes / MIB
        assert predicted == pytest.approx(measured_mib, abs=5)

    def test_it_errs_towards_reserving_too_much(self):
        # Under-reserving spills into system RAM; over-reserving only costs a layer.
        for ubatch, measured in ((128, 84), (256, 162), (512, 318)):
            predicted = overhead.runtime_allowance(model(), ubatch).total_bytes / MIB
            assert predicted >= measured


class TestUnknownModel:
    def test_a_model_without_a_vocabulary_falls_back_and_says_so(self):
        allowance = overhead.runtime_allowance(model(vocab=None), 512)
        assert allowance.total_bytes == overhead.FALLBACK_BYTES
        assert not allowance.calibrated
        assert "vocabulary" in allowance.detail

    def test_a_calculated_allowance_names_what_it_was_fitted_on(self):
        assert overhead.CALIBRATION in overhead.runtime_allowance(model(), 512).detail
