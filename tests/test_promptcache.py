"""Prompt cache budgeting tests.

The cache is a claim on host RAM that the VRAM budget never counted. What is pinned
here is the arithmetic and, more importantly, when the warning stays quiet: a ceiling
is a permission, and reporting it as a cost would fire on every machine.
"""

from __future__ import annotations

from setpoint.budget import promptcache
from setpoint.budget.types import KvEstimate

GIB = 1024**3
MIB = 1024**2


def kv(total_bytes: int, context: int = 4096, upper_bound: bool = False) -> KvEstimate:
    return KvEstimate(
        context=context,
        cache_type_k="f16",
        cache_type_v="f16",
        bytes_per_block=(total_bytes,),
        upper_bound=upper_bound,
    )


class TestEstimate:
    def test_one_conversation_costs_what_its_context_costs(self):
        estimate = promptcache.estimate(kv(512 * MIB))
        assert estimate.bytes_per_conversation == 512 * MIB

    def test_the_ceiling_says_how_many_fit(self):
        estimate = promptcache.estimate(kv(512 * MIB), cache_ram_mib=4096)
        assert estimate.conversations_under_ceiling == 8

    def test_the_default_ceiling_is_the_servers_own(self):
        assert promptcache.DEFAULT_CACHE_RAM_MIB == 8192

    def test_turning_the_cache_off_leaves_nothing_to_budget(self):
        assert promptcache.estimate(kv(512 * MIB), cache_ram_mib=0) is None

    def test_an_unmodelled_kv_cache_cannot_be_turned_into_a_ram_figure(self):
        # Guessing here would put an invented number next to measured ones.
        assert promptcache.estimate(None) is None

    def test_an_unlimited_ceiling_says_so(self):
        estimate = promptcache.estimate(kv(512 * MIB), cache_ram_mib=-1)
        assert any("unlimited" in note for note in estimate.notes)

    def test_the_sliding_window_discount_is_not_taken(self):
        # The host copy measured larger than the undiscounted formula, so shrinking it
        # would understate the claim on RAM.
        estimate = promptcache.estimate(kv(512 * MIB, upper_bound=True))
        assert estimate.bytes_per_conversation == 512 * MIB
        assert any("not reduced" in note for note in estimate.notes)

    def test_what_fits_is_bounded_by_the_ram_as_well_as_the_ceiling(self):
        estimate = promptcache.estimate(kv(1 * GIB), cache_ram_mib=8192)
        assert estimate.conversations_under_ceiling == 8
        assert estimate.conversations_within(3 * GIB) == 3

    def test_no_room_means_none_fit_rather_than_a_negative_count(self):
        estimate = promptcache.estimate(kv(1 * GIB))
        assert estimate.conversations_within(-1) == 0


class TestPressure:
    def test_it_is_quiet_when_the_model_fits_on_the_card(self):
        # Nothing of the model is in RAM, so the cache is not competing with it.
        estimate = promptcache.estimate(kv(512 * MIB))
        assert promptcache.pressure_note(estimate, cpu_bytes=0, host_ram_bytes=8 * GIB) is None

    def test_it_is_quiet_when_there_is_room_for_plenty(self):
        estimate = promptcache.estimate(kv(64 * MIB))
        assert (
            promptcache.pressure_note(estimate, cpu_bytes=1 * GIB, host_ram_bytes=8 * GIB) is None
        )

    def test_it_speaks_when_a_partial_offload_and_the_cache_want_the_same_ram(self):
        estimate = promptcache.estimate(kv(1 * GIB))
        note = promptcache.pressure_note(estimate, cpu_bytes=3 * GIB, host_ram_bytes=int(3.5 * GIB))
        assert note is not None
        assert "--cache-ram" in note

    def test_it_says_none_fit_rather_than_going_quiet_when_ram_is_already_gone(self):
        estimate = promptcache.estimate(kv(1 * GIB))
        note = promptcache.pressure_note(estimate, cpu_bytes=3 * GIB, host_ram_bytes=int(3.2 * GIB))
        assert "0 conversation(s)" in note

    def test_without_a_ram_reading_it_says_nothing(self):
        estimate = promptcache.estimate(kv(1 * GIB))
        assert promptcache.pressure_note(estimate, cpu_bytes=3 * GIB, host_ram_bytes=None) is None
