"""Hardware layer tests.

These cover pure logic only, so they run on machines without an NVIDIA GPU.
"""

from __future__ import annotations

from setpoint.hardware.nvml import ACTIVE_THROTTLE_REASONS, decode_throttle_reasons
from setpoint.hardware.types import AdapterMemory, DriverInfo, ProbeStatus
from setpoint.hardware.wddm import _counter_kind, match_adapter

GIB = 1024**3


class TestDriverTuple:
    def test_parses_two_part_version(self):
        assert DriverInfo(ProbeStatus.OK, driver_version="512.89").driver_tuple == (512, 89)

    def test_orders_across_the_sysmem_fallback_gate(self):
        old = DriverInfo(ProbeStatus.OK, driver_version="512.89").driver_tuple
        new = DriverInfo(ProbeStatus.OK, driver_version="536.40").driver_tuple
        assert old < (536, 40)
        assert new >= (536, 40)

    def test_missing_version_is_none(self):
        assert DriverInfo(ProbeStatus.UNAVAILABLE).driver_tuple is None

    def test_non_numeric_suffix_is_truncated_not_fatal(self):
        assert DriverInfo(ProbeStatus.OK, driver_version="550.12.beta").driver_tuple == (550, 12)


class TestCudaVersion:
    def test_encodes_major_and_minor(self):
        info = DriverInfo(ProbeStatus.OK, cuda_driver_major=11, cuda_driver_minor=6)
        assert info.cuda_driver_version == "11.6"

    def test_absent_when_unknown(self):
        assert DriverInfo(ProbeStatus.OK).cuda_driver_version is None


class TestThrottleReasons:
    def test_idle_bit_decodes(self):
        assert decode_throttle_reasons(0x1) == ("gpu_idle",)

    def test_multiple_bits_decode_in_order(self):
        assert decode_throttle_reasons(0x4 | 0x40) == ("sw_power_cap", "hw_thermal_slowdown")

    def test_clear_mask_is_empty(self):
        assert decode_throttle_reasons(0) == ()

    def test_idle_is_not_treated_as_active_throttling(self):
        # An idle GPU reports gpu_idle. Flagging that would be a false alarm.
        reasons = decode_throttle_reasons(0x1 | 0x2)
        assert not [r for r in reasons if r in ACTIVE_THROTTLE_REASONS]

    def test_thermal_slowdown_is_active(self):
        reasons = decode_throttle_reasons(0x20)
        assert [r for r in reasons if r in ACTIVE_THROTTLE_REASONS] == ["sw_thermal_slowdown"]


class TestCounterKind:
    def test_recognises_each_counter(self):
        assert _counter_kind(r"\gpu adapter memory(x)\Dedicated Usage") == "dedicated"
        assert _counter_kind(r"\gpu adapter memory(x)\Shared Usage") == "shared"
        assert _counter_kind(r"\gpu adapter memory(x)\Total Committed") == "committed"

    def test_unknown_counter_is_ignored(self):
        assert _counter_kind(r"\processor(_total)\% processor time") is None


class TestMatchAdapter:
    def test_matches_on_dedicated_usage(self):
        adapters = (
            AdapterMemory("igpu", dedicated_bytes=64 * 1024 * 1024),
            AdapterMemory("dgpu", dedicated_bytes=3 * GIB),
        )
        adapter, confident = match_adapter(adapters, nvml_used_bytes=3 * GIB)
        assert adapter is not None
        assert adapter.instance == "dgpu"
        assert confident

    def test_falls_back_to_largest_when_no_close_match(self):
        adapters = (
            AdapterMemory("a", dedicated_bytes=1 * GIB),
            AdapterMemory("b", dedicated_bytes=2 * GIB),
        )
        adapter, confident = match_adapter(adapters, nvml_used_bytes=8 * GIB)
        assert adapter is not None
        assert adapter.instance == "b"
        assert not confident

    def test_reports_low_confidence_without_an_nvml_reading(self):
        adapters = (AdapterMemory("only", dedicated_bytes=GIB),)
        _, confident = match_adapter(adapters, nvml_used_bytes=None)
        assert not confident

    def test_no_usable_adapter_returns_none(self):
        adapter, confident = match_adapter((AdapterMemory("x"),), nvml_used_bytes=GIB)
        assert adapter is None
        assert not confident
