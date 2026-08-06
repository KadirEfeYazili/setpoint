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
    """Peak VRAM read off a GTX 1650 on the Vulkan backend, full offload.

    Two architectures with different vocabularies, each measured against a baseline taken
    immediately before the run, because the desktop's own VRAM use drifts.
    """

    QWEN = (model(vocab=151936, embedding=2048), [(128, 92), (256, 166), (512, 322)])
    GEMMA = (model(vocab=262144, embedding=1152), [(128, 149), (256, 274), (512, 534)])

    @pytest.mark.parametrize("case", [QWEN, GEMMA], ids=["qwen2-152k", "gemma3-262k"])
    def test_it_reserves_more_than_the_card_used_but_not_much_more(self, case):
        # One-sided on purpose. Under-reserving spills into system RAM or refuses to
        # start; over-reserving only risks a layer, so the band is asymmetric.
        info, readings = case
        for ubatch, measured in readings:
            predicted = overhead.runtime_allowance(info, ubatch).total_bytes / MIB
            assert 0 <= predicted - measured <= 20

    def test_the_per_token_slope_tracks_the_vocabulary_across_the_two_models(self):
        # The discriminating measurement: 1.73 times the vocabulary moved the measured
        # slope by 1.69, which the embedding term cannot account for.
        slopes = []
        for info, _ in (self.QWEN, self.GEMMA):
            allowance = overhead.runtime_allowance(info, 512)
            slopes.append((allowance.total_bytes - overhead.BASE_BYTES) / 512)
        assert slopes[1] / slopes[0] == pytest.approx(1.69, abs=0.05)


class TestUnknownModel:
    def test_a_model_without_a_vocabulary_falls_back_and_says_so(self):
        allowance = overhead.runtime_allowance(model(vocab=None), 512)
        assert allowance.total_bytes == overhead.FALLBACK_BYTES
        assert not allowance.calibrated
        assert "vocabulary" in allowance.detail

    def test_a_calculated_allowance_names_what_it_was_fitted_on(self):
        assert overhead.CALIBRATION in overhead.runtime_allowance(model(), 512).detail
