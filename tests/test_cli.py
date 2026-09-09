"""CLI plumbing tests.

Only the pure parts are covered here: where the boundary between setpoint's own flags
and the backend's is drawn, and what a profile turns into on a command line. The
commands themselves talk to hardware and are exercised by hand.
"""

from __future__ import annotations

import pytest

from setpoint.backend import server_argv
from setpoint.cli import REGRESSION_TOLERANCE, build_parser, split_forwarded
from setpoint.profile import Config


class TestForwarding:
    def test_arguments_after_the_separator_belong_to_the_backend(self):
        own, forwarded = split_forwarded(["run", "model", "--print-only", "--", "--port", "9090"])
        assert own == ["run", "model", "--print-only"]
        assert forwarded == ("--port", "9090")

    def test_without_a_separator_nothing_is_forwarded(self):
        own, forwarded = split_forwarded(["run", "model", "--print-only"])
        assert own == ["run", "model", "--print-only"]
        assert forwarded == ()

    def test_a_flag_of_ours_after_the_separator_is_not_reclaimed(self):
        # The whole point: setpoint must not eat what it was told to pass on.
        _, forwarded = split_forwarded(["run", "m", "--", "--print-only"])
        assert forwarded == ("--print-only",)

    def test_only_the_first_separator_splits(self):
        _, forwarded = split_forwarded(["run", "m", "--", "-a", "--", "-b"])
        assert forwarded == ("-a", "--", "-b")

    def test_a_trailing_separator_forwards_nothing(self):
        own, forwarded = split_forwarded(["run", "m", "--"])
        assert own == ["run", "m"]
        assert forwarded == ()


class TestServerCommand:
    def config(self, **overrides) -> Config:
        fields = {"n_gpu_layers": 28, "ubatch_size": 128, "flash_attn": True}
        fields.update(overrides)
        return Config(**fields)

    def test_the_context_comes_from_the_profile_not_the_config(self):
        argv = server_argv("llama-server", "m.gguf", self.config(), 4096)
        assert argv[argv.index("--ctx-size") + 1] == "4096"

    def test_long_flag_names_are_used_throughout(self):
        # Short forms differ between llama.cpp's tools; the long ones have held still.
        argv = server_argv("llama-server", "m.gguf", self.config(threads=6), 1024)
        assert "--n-gpu-layers" in argv
        assert "--ubatch-size" in argv
        assert "--threads" in argv
        assert not any(a in ("-ngl", "-ub", "-t") for a in argv)

    def test_unset_settings_are_left_out(self):
        argv = server_argv("llama-server", "m.gguf", self.config(threads=None), 1024)
        assert "--threads" not in argv
        assert "--batch-size" not in argv

    def test_flash_attention_is_spelled_the_modern_way(self):
        on = server_argv("llama-server", "m.gguf", self.config(flash_attn=True), 1024)
        off = server_argv("llama-server", "m.gguf", self.config(flash_attn=False), 1024)
        assert on[on.index("--flash-attn") + 1] == "on"
        assert off[off.index("--flash-attn") + 1] == "off"

    def test_the_device_is_named(self):
        argv = server_argv("llama-server", "m.gguf", self.config(), 1024, devices=("Vulkan0",))
        assert argv[argv.index("--device") + 1] == "Vulkan0"

    def test_forwarded_arguments_come_last(self):
        argv = server_argv("llama-server", "m.gguf", self.config(), 1024, extra=("--port", "9090"))
        assert argv[-2:] == ["--port", "9090"]

    def test_cache_types_are_always_stated(self):
        argv = server_argv(
            "llama-server", "m.gguf", self.config(cache_type_k="q8_0", cache_type_v="q8_0"), 1024
        )
        assert argv[argv.index("--cache-type-k") + 1] == "q8_0"
        assert argv[argv.index("--cache-type-v") + 1] == "q8_0"


class TestParser:
    @pytest.mark.parametrize(
        "command", ["doctor", "budget", "tune", "run", "bench", "profile", "hardware"]
    )
    def test_every_documented_command_parses(self, command):
        argv = [command] if command in ("doctor", "profile", "hardware") else [command, "model"]
        assert build_parser().parse_args(argv).command == command

    def test_a_regression_tolerance_wider_than_the_observed_spread(self):
        # Run-to-run spread measured at 0.56% on real hardware; the gate has to clear it.
        assert 0.01 < REGRESSION_TOLERANCE < 0.2


class TestPanelContract:
    """The panel may show nothing that a script cannot also read.

    Written as a test because the rule is easy to break by accident: a pane gets a new
    figure, and the only place it exists is on screen.
    """

    def test_the_panel_shows_nothing_the_json_does_not_carry(self):
        from setpoint.panel import data

        # Every field the profile pane puts on screen, and the command that carries it.
        shown = {
            "decode_tok_s",
            "spread",
            "speedup",
            "n_gpu_layers",
            "ubatch_size",
            "speculator",
            "peak_vram_mib",
            "switch_seconds",
            "checks",
        }
        assert shown <= set(data.ProfileView.__dataclass_fields__)

    def test_profile_show_carries_the_history_and_the_switch_cost(self):
        from setpoint import cli

        # These two are the ones with no home of their own: the history lives in the
        # sentinel's files and the switch cost in route's, so `profile show --json` is
        # where they have to surface.
        assert hasattr(cli, "_history_mapping")
        assert hasattr(cli, "_switch_seconds")

    def test_the_panel_command_can_report_without_opening_a_screen(self):
        parser = build_parser()
        args = parser.parse_args(["panel", "--check"])
        assert args.check is True

    def test_importing_the_cli_does_not_pull_in_the_extra(self):
        # `doctor` and `budget` have to work with nothing installed, so the widgets are
        # imported inside `panel.run()` rather than at module level. Checked here
        # because the failure mode is silent: it only shows on a machine without it.
        import subprocess
        import sys

        code = "import setpoint.cli, sys; print('textual' in sys.modules)"
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert done.stdout.strip() == "False"
