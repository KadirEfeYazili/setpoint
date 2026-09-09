"""Panel data tests.

The panel's binding rule is that it measures nothing, so what these pin is the reading
and the formatting: a figure that was never measured has to read as absent rather than
as zero, and every store the panel touches has to be redirectable so a test never reads
the developer's own profiles.
"""

from __future__ import annotations

from setpoint import profile as prof
from setpoint import sentinel
from setpoint.hardware.types import (
    DriverInfo,
    GpuSample,
    GpuStatic,
    HardwareSnapshot,
    HostInfo,
    ProbeStatus,
)
from setpoint.measure import Statistic
from setpoint.panel import data
from setpoint.profile import store
from setpoint.route import LoadCost, save_load

TIGHT = Statistic((86.6, 86.71, 86.7, 86.8, 86.9))


def signature(**overrides) -> prof.Signature:
    fields = {
        "model_digest": "sha256:" + "a" * 64,
        "model_digest_kind": prof.DIGEST_HEADER,
        "model_size_bytes": 815_000_000,
        "gpu": "NVIDIA GeForce GTX 1650",
        "vram_total_mb": 4096,
        "driver": "512.89",
        "backend": "llama.cpp b10850 (Vulkan)",
        "platform": "windows/amd64",
    }
    fields.update(overrides)
    return prof.Signature(**fields)


def profile(**overrides) -> prof.Profile:
    fields = {
        "signature": signature(),
        "target": prof.Target(context=4096, optimize=prof.Objective.SPEED),
        "config": prof.Config(
            n_gpu_layers=27, ubatch_size=128, flash_attn=True, spec_type="ngram-simple"
        ),
        "measurement": prof.Measurement.from_statistics(
            decode=TIGHT, measured_at="2026-09-09T12:23:24Z", peak_vram_mb=1335
        ),
        "baseline": prof.Baseline(
            label="auto (-ngl 99)",
            config=prof.Config(n_gpu_layers=99),
            decode_tok_s=82.54,
            speedup=1.05,
        ),
        "model": prof.ModelRef(name="gemma3", architecture="gemma3", file_type="Q4_K_M"),
        "created": "2026-09-09T12:23:24Z",
    }
    fields.update(overrides)
    return prof.Profile(**fields)


def snapshot(free_mib: int = 3102, used_mib: int = 993) -> HardwareSnapshot:
    return HardwareSnapshot(
        host=HostInfo(os="Windows", os_release="11", arch="amd64", python_version="3.12.3"),
        driver=DriverInfo(status=ProbeStatus.OK, driver_version="512.89"),
        gpus=(
            GpuStatic(
                index=0,
                name="NVIDIA GeForce GTX 1650",
                uuid="GPU-test",
                vram_total_bytes=4096 * 1024**2,
            ),
        ),
        samples=(
            GpuSample(
                index=0,
                vram_free_bytes=free_mib * 1024**2,
                vram_used_bytes=used_mib * 1024**2,
                utilization_pct=4,
                temperature_c=50,
            ),
        ),
    )


class TestCard:
    def test_it_reports_what_the_card_holds_right_now(self):
        card = data.read_card(snapshot())
        values = {row.label: row.value for row in card.rows}
        assert values["vram free"] == "3102 MiB"
        assert values["vram in use"] == "993 MiB"
        assert values["utilisation"] == "4%"

    def test_a_machine_with_no_gpu_says_so_rather_than_showing_zeroes(self):
        empty = HardwareSnapshot(
            host=HostInfo(os="Linux", os_release="6.8", arch="amd64", python_version="3.12"),
            driver=DriverInfo(status=ProbeStatus.UNSUPPORTED),
        )
        card = data.read_card(empty)
        assert card.detail
        assert card.free_mib is None

    def test_a_quiet_card_shows_no_throttle_row(self):
        assert not any(row.label == "throttling" for row in data.read_card(snapshot()).rows)

    def test_a_throttling_card_shows_one(self):
        base = snapshot()
        hot = HardwareSnapshot(
            host=base.host,
            driver=base.driver,
            gpus=base.gpus,
            samples=(
                GpuSample(
                    index=0,
                    vram_free_bytes=1,
                    vram_used_bytes=1,
                    throttle_reasons=("hw_thermal_slowdown",),
                ),
            ),
        )
        assert any(row.label == "throttling" for row in data.read_card(hot).rows)


class TestProfiles:
    def test_it_reads_a_stored_profile(self, tmp_path):
        store.save(profile(), tmp_path)
        views = data.read_profiles(tmp_path, history_root=tmp_path, loads_dir=tmp_path)
        assert len(views) == 1
        assert views[0].decode_tok_s == 86.71
        assert views[0].speculator == "ngram-simple"

    def test_a_switch_cost_that_was_never_measured_reads_as_absent(self, tmp_path):
        store.save(profile(), tmp_path)
        view = data.read_profiles(tmp_path, history_root=tmp_path, loads_dir=tmp_path)[0]
        assert view.switch_seconds is None
        assert {r.label: r.value for r in view.rows}["switch cost"] == "not measured"

    def test_a_measured_switch_cost_is_shown_beside_the_profile(self, tmp_path):
        one = profile()
        store.save(one, tmp_path)
        save_load(
            LoadCost(model_digest=one.signature.model_digest, seconds=Statistic((2.6, 2.62, 2.7))),
            tmp_path,
        )
        view = data.read_profiles(tmp_path, history_root=tmp_path, loads_dir=tmp_path)[0]
        assert view.switch_seconds == 2.62

    def test_it_counts_the_regression_checks(self, tmp_path):
        one = profile()
        store.save(one, tmp_path)
        path = sentinel.history_path(one.signature.model_digest, one.target.context, tmp_path)
        for at in ("2026-09-09T10:00:00Z", "2026-09-09T11:00:00Z"):
            sentinel.append(path, sentinel.record_of(one, (86.0, 86.5, 87.0), at))
        view = data.read_profiles(tmp_path, history_root=tmp_path, loads_dir=tmp_path)[0]
        assert view.checks == 2

    def test_no_profiles_is_not_an_error(self, tmp_path):
        assert data.read_profiles(tmp_path, history_root=tmp_path, loads_dir=tmp_path) == ()


class TestHistory:
    def test_it_reads_the_checks_for_one_profile(self, tmp_path):
        one = profile()
        store.save(one, tmp_path)
        path = sentinel.history_path(one.signature.model_digest, one.target.context, tmp_path)
        sentinel.append(
            path, sentinel.record_of(one, (86.0, 86.5, 87.0), "2026-09-09T10:00:00Z", note="first")
        )
        entries = data.read_history(
            prof.signature_id(one.signature), tmp_path, history_root=tmp_path
        )
        assert len(entries) == 1
        assert entries[0].decode_tok_s == 86.5
        assert entries[0].note == "first"

    def test_an_unknown_profile_has_no_history(self, tmp_path):
        assert data.read_history("nope", tmp_path, history_root=tmp_path) == ()


class TestFormatting:
    def test_a_missing_figure_reads_as_a_dash_not_a_zero(self):
        assert data._mib(None) == "-"
        assert data._tok_s(None) == "-"

    def test_a_missing_switch_cost_says_what_is_missing(self):
        # "not measured" and "0.00s" would send a router in opposite directions.
        assert data._seconds(None) == "not measured"
        assert data._seconds(2.625) == "2.62s"

    def test_spread_is_shown_as_a_percentage_of_the_median(self):
        assert data._spread(0.007) == "spread 0.7%"
        assert data._spread(None) == ""
