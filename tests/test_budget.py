"""Budgeter tests.

The KV formula and the offload split are the two places a wrong number turns into a
wrong recommendation, so both are pinned against hand-computed values.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from setpoint import budget
from setpoint.budget import kv, planner, vram
from setpoint.budget.types import VramBudget
from setpoint.hardware.types import (
    DriverInfo,
    GpuSample,
    GpuStatic,
    HardwareSnapshot,
    HostInfo,
    ProbeStatus,
)
from setpoint.model.types import Attention, Experts, ModelInfo, Weights

MIB = 1024 * 1024
GIB = 1024 * MIB


def model(
    block_count: int = 4,
    block_bytes: int = 100 * MIB,
    head_count_kv: int = 2,
    architecture: str = "llama",
    sliding_window: int | None = None,
    experts: Experts | None = None,
    tied_embedding: bool = False,
) -> ModelInfo:
    return ModelInfo(
        path=Path("model.gguf"),
        file_bytes=block_count * block_bytes,
        architecture=architecture,
        name="Test",
        block_count=block_count,
        embedding_length=64,
        attention=Attention(
            head_count=(8,) * block_count,
            head_count_kv=(head_count_kv,) * block_count,
            key_length=128,
            value_length=128,
            sliding_window=sliding_window,
        ),
        weights=Weights(
            block_bytes=(block_bytes,) * block_count,
            expert_bytes=(0,) * block_count,
            input_bytes=10 * MIB,
            output_bytes=20 * MIB,
            tied_embedding=tied_embedding,
        ),
        parameter_count=1_000_000,
        quant_mix=(),
        train_context=8192,
        experts=experts,
    )


def tight_budget(total: int = 300 * MIB) -> VramBudget:
    """A ceiling that is exactly `total`, so the arithmetic in the tests stays readable."""
    return vram.assumed(total, fragmentation_pct=0.0, runtime_allowance_bytes=0)


def snapshot(total: int = 4 * GIB, free: int | None = 3 * GIB) -> HardwareSnapshot:
    return HardwareSnapshot(
        host=HostInfo("Linux", "6.0", "x86_64", "3.12.0", total_ram_bytes=16 * GIB),
        driver=DriverInfo(ProbeStatus.OK, driver_version="550.00"),
        gpus=(GpuStatic(index=0, name="Test GPU", uuid=None, vram_total_bytes=total),),
        samples=(GpuSample(index=0, vram_free_bytes=free),),
    )


class TestCacheTypes:
    def test_f16_costs_two_bytes_per_element(self):
        assert kv.bytes_per_element("f16") == 2.0

    def test_q8_0_carries_its_block_scale(self):
        assert kv.bytes_per_element("q8_0") == pytest.approx(34 / 32)

    def test_an_unknown_type_is_rejected(self):
        with pytest.raises(ValueError):
            kv.bytes_per_element("q3_k_m")


class TestKvEstimate:
    def test_size_matches_the_hand_computed_value(self):
        # 4 blocks * 2 kv heads * (128 + 128) elements * 2 bytes * 1024 tokens.
        estimate = kv.estimate(model(), 1024)
        assert estimate.total_bytes == 4 * 2 * 256 * 2 * 1024
        assert estimate.bytes_per_token == 4 * 2 * 256 * 2

    def test_halving_the_kv_heads_halves_the_cache(self):
        wide = kv.estimate(model(head_count_kv=8), 1024).total_bytes
        narrow = kv.estimate(model(head_count_kv=4), 1024).total_bytes
        assert wide == 2 * narrow

    def test_quantizing_the_cache_shrinks_it(self):
        f16 = kv.estimate(model(), 1024).total_bytes
        q8 = kv.estimate(model(), 1024, "q8_0", "q8_0").total_bytes
        assert q8 < f16

    def test_a_quantized_cache_says_it_needs_flash_attention(self):
        estimate = kv.estimate(model(), 1024, "q8_0", "q8_0")
        assert any("flash attention" in note for note in estimate.notes)

    def test_a_sliding_window_makes_the_figure_an_upper_bound(self):
        estimate = kv.estimate(model(sliding_window=512), 4096)
        assert estimate.upper_bound
        assert any("Sliding-window" in note for note in estimate.notes)

    def test_a_context_inside_the_window_is_exact(self):
        assert not kv.estimate(model(sliding_window=4096), 1024).upper_bound

    def test_latent_attention_gets_no_estimate(self):
        # Not being able to model it and modelling it as zero are different claims.
        assert kv.estimate(model(architecture="deepseek2"), 1024) is None


class TestVramBudget:
    def test_the_ceiling_subtracts_every_claim(self):
        result = vram.from_snapshot(
            snapshot(), reserve_bytes=100 * MIB, fragmentation_pct=10.0, runtime_allowance_bytes=0
        )
        assert result.free_bytes == 3 * GIB
        assert result.fragmentation_bytes == int(0.1 * 3 * GIB)
        assert result.ceiling_bytes == 3 * GIB - int(0.1 * 3 * GIB) - 100 * MIB

    def test_an_unreadable_free_reading_is_marked_as_assumed(self):
        result = vram.from_snapshot(snapshot(free=None))
        assert not result.measured
        assert result.detail
        assert result.free_bytes == 4 * GIB

    def test_no_gpu_yields_no_budget(self):
        empty = HardwareSnapshot(
            host=HostInfo("Linux", "6.0", "x86_64", "3.12.0"),
            driver=DriverInfo(ProbeStatus.UNAVAILABLE),
        )
        assert vram.from_snapshot(empty) is None

    def test_a_hypothetical_card_never_claims_measurement(self):
        assert not vram.assumed(24 * 1024 * MIB).measured

    def test_the_ceiling_never_goes_negative(self):
        result = vram.from_snapshot(snapshot(), reserve_bytes=99 * GIB)
        assert result.ceiling_bytes == 0


class TestFit:
    def test_everything_fits_and_the_output_head_joins(self):
        info = model()
        placed = planner.fit(info, None, ceiling_bytes=10 * GIB)
        assert placed.n_gpu_layers == 5
        assert placed.output_on_gpu
        assert placed.fits_fully
        assert placed.cpu_bytes == info.weights.input_bytes

    def test_a_tight_ceiling_places_only_what_it_can_hold(self):
        placed = planner.fit(model(), None, ceiling_bytes=250 * MIB)
        assert placed.n_gpu_layers == 2
        assert not placed.output_on_gpu
        assert placed.weights_on_gpu_bytes == 200 * MIB

    def test_the_kv_cache_competes_with_the_weights(self):
        info = model(head_count_kv=16)
        estimate = kv.estimate(info, 8192)
        without = planner.fit(info, None, ceiling_bytes=250 * MIB).n_gpu_layers
        with_cache = planner.fit(info, estimate, ceiling_bytes=250 * MIB).n_gpu_layers
        assert with_cache < without

    def test_the_output_head_stays_behind_when_it_does_not_fit(self):
        # Room for all four blocks but not the head: llama.cpp reports -ngl 4, not 5.
        placed = planner.fit(model(), None, ceiling_bytes=410 * MIB)
        assert placed.n_gpu_layers == 4
        assert not placed.output_on_gpu

    def test_the_cost_of_the_next_block_is_reported(self):
        # It is also how much desktop drift the plan can absorb before it is wrong.
        placed = planner.fit(model(), None, ceiling_bytes=250 * MIB)
        assert placed.next_block_bytes == 100 * MIB

    def test_a_plan_that_fits_has_no_next_block(self):
        assert planner.fit(model(), None, ceiling_bytes=10 * GIB).next_block_bytes is None

    def test_nothing_fits_under_a_zero_ceiling(self):
        placed = planner.fit(model(), None, ceiling_bytes=0)
        assert placed.n_gpu_layers == 0
        assert placed.gpu_bytes == 0


class TestTiedEmbedding:
    """Measured: on a tied-embedding model those bytes sit on the GPU at every split.

    The token embedding is also the output projection there, and the matmul runs where
    the blocks run. Counting them CPU-side made the plan two blocks optimistic.
    """

    def test_a_tied_embedding_is_spent_before_any_block(self):
        tied = planner.fit(model(tied_embedding=True), None, ceiling_bytes=250 * MIB)
        loose = planner.fit(model(tied_embedding=False), None, ceiling_bytes=250 * MIB)
        assert tied.weights_on_gpu_bytes > loose.weights_on_gpu_bytes - 10 * MIB
        assert tied.n_gpu_layers <= loose.n_gpu_layers

    def test_it_counts_towards_the_gpu_side_once_a_block_lands(self):
        placed = planner.fit(model(tied_embedding=True), None, ceiling_bytes=250 * MIB)
        assert placed.n_gpu_layers >= 1
        assert placed.weights_on_gpu_bytes >= 10 * MIB

    def test_an_untied_model_leaves_the_embedding_on_the_cpu(self):
        # Refuted by measurement on a model with a separate output projection: those
        # bytes were not resident, and assuming otherwise over-reserved by their size.
        info = model(tied_embedding=False)
        placed = planner.fit(info, None, ceiling_bytes=10 * GIB)
        assert placed.cpu_bytes == info.weights.input_bytes

    def test_nothing_on_the_gpu_means_nothing_resident(self):
        placed = planner.fit(model(tied_embedding=True), None, ceiling_bytes=0)
        assert placed.weights_on_gpu_bytes == 0


class TestMaxContext:
    def test_it_is_capped_by_the_trained_context(self):
        assert planner.max_context(model(), 10 * GIB, "f16", "f16") == 8192

    def test_no_room_for_the_weights_means_no_context(self):
        assert planner.max_context(model(), 10 * MIB, "f16", "f16") == 0


class TestPlan:
    def test_alternatives_only_appear_when_they_win_blocks(self):
        result = budget.plan(model(head_count_kv=16), 8192, tight_budget())
        assert not result.offload.fits_fully
        assert result.alternatives
        for option in result.alternatives:
            assert option.n_gpu_layers > result.offload.n_gpu_layers

    def test_a_fitting_plan_offers_a_longer_context(self):
        result = budget.plan(model(), 1024, vram.assumed(10 * GIB, runtime_allowance_bytes=0))
        assert result.offload.fits_fully
        assert result.alternatives[0].change == "-c 8192"

    def test_seeds_bracket_the_static_estimate(self):
        result = budget.plan(model(head_count_kv=16), 8192, tight_budget())
        fitted = result.offload.n_gpu_layers
        layers = [c.n_gpu_layers for c in result.candidates]
        assert min(layers) < fitted < max(layers)

    def test_an_unmodelled_cache_is_declared_in_the_notes(self):
        result = budget.plan(model(architecture="deepseek2"), 8192, vram.assumed(4 * GIB))
        assert result.kv is None
        assert any("no KV cache model" in note for note in result.notes)

    def test_ram_pressure_is_judged_against_what_is_free(self):
        # The figure that matters is RAM available now, not RAM installed.
        result = budget.plan(
            model(block_count=40, block_bytes=GIB),
            8192,
            vram.assumed(4 * GIB),
            host_ram_bytes=8 * GIB,
        )
        assert any("free right now" in note for note in result.notes)

    def test_plenty_of_free_ram_raises_no_note(self):
        result = budget.plan(
            model(block_count=4, block_bytes=100 * MIB),
            1024,
            vram.assumed(4 * GIB),
            host_ram_bytes=8 * GIB,
        )
        assert not any("free right now" in note for note in result.notes)

    def test_the_json_payload_carries_the_derived_numbers(self):
        payload = budget.plan(model(), 4096, vram.assumed(4 * GIB)).to_dict()
        assert payload["vram"]["ceiling_bytes"] > 0
        assert payload["plan"]["n_gpu_layers"] >= 0
        assert payload["kv_cache"]["cache_type_k"] == "f16"
        assert payload["request"]["context"] == 4096
