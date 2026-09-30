"""Residency view tests.

The point of this view is what it refuses to say. The driver on the development machine
lists every process holding the card and attributes memory to none of them, so the
dangerous failure is not a missing number but an invented one: dividing the total up,
or reporting a model as resident because a profile exists for it.
"""

from __future__ import annotations

from setpoint.hardware.types import (
    DriverInfo,
    GpuProcess,
    GpuSample,
    GpuStatic,
    HardwareSnapshot,
    HostInfo,
    ProbeStatus,
)
from setpoint.panel import data


def snapshot(*processes: GpuProcess, gpus: bool = True) -> HardwareSnapshot:
    static = (
        GpuStatic(
            index=0,
            name="NVIDIA GeForce GTX 1650",
            uuid=None,
            vram_total_bytes=4096 * 1024 * 1024,
        ),
    )
    return HardwareSnapshot(
        host=HostInfo(os="windows", os_release="11", arch="amd64", python_version="3.12"),
        driver=DriverInfo(status=ProbeStatus.OK),
        gpus=static if gpus else (),
        samples=(
            (
                GpuSample(
                    index=0,
                    vram_used_bytes=1577 * 1024 * 1024,
                    vram_free_bytes=2518 * 1024 * 1024,
                ),
            )
            if gpus
            else ()
        ),
        processes=processes,
    )


def view(*processes: GpuProcess, **kwargs) -> data.ResidencyView:
    return data.read_residency(snapshot(*processes), **kwargs)


def labels(rows) -> str:
    return " | ".join(f"{r.label}={r.value}({r.note})" for r in rows)


class TestTheCard:
    def test_it_reports_what_every_process_holds_together(self):
        assert "1577 MiB" in labels(view().rows)

    def test_a_machine_without_a_gpu_says_so_rather_than_showing_zeroes(self):
        assert data.read_residency(snapshot(gpus=False)).detail is not None

    def test_an_unattributed_driver_is_named_as_such(self):
        # The row exists so that "we did not look" and "nothing was there" stay apart.
        rows = view(GpuProcess(pid=1, name="x.exe")).rows
        assert "not attributed" in labels(rows)

    def test_a_driver_that_does_attribute_memory_is_not_contradicted(self):
        rows = view(GpuProcess(pid=1, name="x.exe", vram_mib=512)).rows
        assert "not attributed" not in labels(rows)

    def test_no_processes_at_all_does_not_claim_the_driver_is_hiding_them(self):
        assert "not attributed" not in labels(view().rows)


class TestProcesses:
    def test_an_inference_process_is_marked(self):
        rows = view(GpuProcess(pid=1, name="C:/llama/llama-server.exe")).processes
        assert rows[0].label == "llama-server.exe"
        assert rows[0].note == "inference"

    def test_other_processes_are_listed_without_the_mark(self):
        rows = view(GpuProcess(pid=1, name="C:/x/chrome.exe")).processes
        assert rows[0].note == ""

    def test_the_full_path_is_not_shown(self):
        # It is a personal path, and the binary is the part that identifies it.
        rows = view(GpuProcess(pid=1, name="C:/Users/someone/llama/llama-server.exe")).processes
        assert "Users" not in rows[0].label

    def test_an_unnamed_process_is_still_counted(self):
        rows = view(GpuProcess(pid=1, name=None)).processes
        assert rows[0].label == "unknown"

    def test_a_process_the_driver_did_not_size_shows_no_figure(self):
        rows = view(GpuProcess(pid=1, name="llama-server.exe")).processes
        assert rows[0].value == "-"
