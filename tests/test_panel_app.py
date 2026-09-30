"""Panel widget tests.

Skipped when the tui extra is absent, which is the point of it being an extra: the rest
of the suite has to pass with nothing installed beyond the two runtime dependencies.

The app is driven headless through Textual's own harness. What is checked is that it
composes, fills from the data layer, and survives the interactions a person performs
first: switching model, switching tab, refreshing.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textual", reason="the panel needs the tui extra")

from textual.widgets import DataTable, Input, Label, RichLog, Select  # noqa: E402

from setpoint import serve  # noqa: E402
from setpoint.panel import data  # noqa: E402
from setpoint.panel.app import Panel  # noqa: E402

CARD = data.Card(name="Test GPU", total_mib=4096, free_mib=3000, used_mib=1096)
BUDGET = data.BudgetView(
    model="gemma3:1b",
    context=4096,
    rows=(data.Row("free", "2.93 GiB", "measured now"), data.Row("safe ceiling", "2.13 GiB")),
    plan_rows=(data.Row("-ngl", "27", "26 of 26"),),
    alternatives=(data.Row("-c 32768", "-ngl 27", "the longest context that still fits"),),
)
PROFILE = data.ProfileView(
    signature_id="a748e45f1a843e88",
    model="gemma3 Q4_K_M",
    context=4096,
    decode_tok_s=86.71,
    spread=0.007,
    speedup=1.05,
    n_gpu_layers=27,
    ubatch_size=128,
    speculator="ngram-simple",
    peak_vram_mib=1335,
    switch_seconds=2.62,
    checks=3,
    created="2026-09-09T12:23:24Z",
)
TARGET = data.ChatTarget(
    reference="gemma3:1b",
    label="gemma3 Q4_K_M",
    signature_id="a748e45f1a843e88",
    context=4096,
    claimed_tok_s=86.71,
    speculator="ngram-simple",
    model_path="gemma3.gguf",
    config=object(),
    gpu="Test GPU",
)
SNAPSHOT = data.Snapshot(
    card=CARD,
    models=("gemma3:1b", "qwen2.5:3b"),
    profiles=(PROFILE,),
    budget=BUDGET,
)


@pytest.fixture(autouse=True)
def _no_machine(monkeypatch):
    """No hardware and no stores: the widgets are what is under test."""
    monkeypatch.setattr(data, "gather", lambda *a, **k: SNAPSHOT)
    monkeypatch.setattr(data, "read_card", lambda *a, **k: CARD)
    monkeypatch.setattr(data, "read_history", lambda *a, **k: ())
    monkeypatch.setattr(data, "read_chat_target", lambda *a, **k: TARGET)


def drive(coro):
    return asyncio.run(coro)


class TestComposition:
    def test_it_fills_every_pane_from_the_snapshot(self):
        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                assert app.query_one("#card", DataTable).row_count == len(CARD.rows)
                assert app.query_one("#terms", DataTable).row_count == len(BUDGET.rows)
                assert app.query_one("#plan", DataTable).row_count == len(BUDGET.plan_rows)
                assert app.query_one("#profiles-table", DataTable).row_count == 1
                assert app.query_one("#profile-detail", DataTable).row_count == len(PROFILE.rows)

        drive(go())

    def test_the_model_list_comes_from_what_is_installed(self):
        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                assert app.query_one("#model", Select).value == "gemma3:1b"

        drive(go())

    def test_with_no_model_given_it_takes_the_first(self):
        async def go():
            app = Panel(interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                assert app.reference == "gemma3:1b"

        drive(go())


class TestInteraction:
    def test_choosing_a_model_re_reads_the_budget(self):
        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                app.query_one("#model", Select).value = "qwen2.5:3b"
                await pilot.pause()
                assert app.reference == "qwen2.5:3b"

        drive(go())

    def test_refreshing_keeps_the_panes_filled(self):
        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                await pilot.press("r")
                await pilot.pause()
                assert app.query_one("#terms", DataTable).row_count == len(BUDGET.rows)

        drive(go())

    def test_every_tab_can_be_opened(self):
        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                for target in ("profiles", "search", "budget"):
                    app.query_one("TabbedContent").active = target
                    await pilot.pause()

        drive(go())


class TestResizing:
    @pytest.mark.parametrize(
        "size",
        [(140, 44), (100, 44), (64, 44), (63, 44), (45, 30), (29, 24), (20, 24), (12, 20)],
    )
    def test_it_mounts_at_any_size(self, size):
        # The mark changes form with the width and the height, and every one of those
        # transitions used to be a chance to raise inside an event handler, which stalls
        # the loop rather than failing loudly.
        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                assert app.query("#banner")

        drive(go())

    def test_highlighting_a_row_before_the_panes_exist_does_not_raise(self):
        # Found by mounting at many sizes: the row-highlighted event arrived while
        # `#profile-detail` was not there, and `query_one` raised inside the handler.
        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                app._show_profile(PROFILE)
                await pilot.pause()

        drive(go())


class TestEmptyMachine:
    def test_a_machine_with_nothing_measured_still_opens(self, monkeypatch):
        # The first thing a new user sees. It must not need a profile to exist.
        monkeypatch.setattr(
            data, "gather", lambda *a, **k: data.Snapshot(card=CARD, models=(), profiles=())
        )

        async def go():
            app = Panel(interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                assert app.query_one("#profiles-table", DataTable).row_count == 0
                assert app.query_one("#terms", DataTable).row_count == 1

        drive(go())

    def test_a_model_that_cannot_be_read_says_why(self, monkeypatch):
        broken = data.Snapshot(
            card=CARD,
            models=("broken:1b",),
            budget=data.BudgetView(model="broken:1b", context=4096, detail="not a GGUF file"),
        )
        monkeypatch.setattr(data, "gather", lambda *a, **k: broken)

        async def go():
            app = Panel(interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                table = app.query_one("#terms", DataTable)
                assert table.row_count == 1
                assert "not a GGUF file" in str(table.get_row_at(0))

        drive(go())


class FakeSession:
    """A server that never starts. What is under test is the pane, not llama.cpp."""

    def __init__(self, reply="an answer", timing=True, fails=False):
        self.reply = reply
        self.base = "http://127.0.0.1:1"
        self.last = serve.Reply(text=reply, decode_tok_s=81.9, tokens=61) if timing else None
        self.fails = fails
        self.sent: list[list[dict]] = []
        self.closed = False

    def stream(self, messages, **_):
        self.sent.append(messages)
        if self.fails:
            raise OSError("connection reset")
        yield self.reply

    def __exit__(self, *_):
        self.closed = True


def talking(monkeypatch, session):
    """Wire the app to a fake server, so nothing loads a model."""
    monkeypatch.setattr(Panel, "_start_session", lambda self, target: session)


async def ask(app, pilot, text: str) -> None:
    """Type into the chat box the way a person does, and wait for the answer.

    The pane has to be the visible one: a widget that was never shown has no size and
    a log written to it renders nothing, which is a property of the harness rather
    than of the panel.
    """
    app.query_one("TabbedContent").active = "chat"
    await pilot.pause()
    box = app.query_one("#say", Input)
    box.focus()
    box.value = text
    await pilot.pause()
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


def lines(app) -> str:
    log = app.query_one("#transcript", RichLog)
    return "\n".join(str(line) for line in log.lines)


def status(app) -> str:
    return str(app.query_one("#chat-status", Label).content)


class TestChat:
    def test_the_pane_says_what_it_would_run_on_before_anything_starts(self):
        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                assert "86.71" in status(app)
                assert "starts on the first message" in status(app)

        drive(go())

    def test_a_model_with_no_profile_offers_the_measurement_instead(self, monkeypatch):
        monkeypatch.setattr(
            data,
            "read_chat_target",
            lambda *a, **k: data.ChatTarget(reference="x:1b", detail="nothing measured"),
        )

        async def go():
            app = Panel(reference="x:1b", interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                assert "press t" in status(app)

        drive(go())

    def test_an_answer_lands_with_what_it_cost(self, monkeypatch):
        talking(monkeypatch, FakeSession())

        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                await ask(app, pilot, "hello")
                text = lines(app)
                assert "an answer" in text
                assert "81.9 t/s" in text
                assert "-6% on the profile" in text

        drive(go())

    def test_the_question_reaches_the_server_and_the_box_is_emptied(self, monkeypatch):
        session = FakeSession()
        talking(monkeypatch, session)

        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                await ask(app, pilot, "why measure")
                assert session.sent[0][-1] == {"role": "user", "content": "why measure"}
                assert app.query_one("#say", Input).value == ""

        drive(go())

    def test_an_answer_the_server_never_timed_is_not_reported_as_zero(self, monkeypatch):
        talking(monkeypatch, FakeSession(timing=False))

        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                await ask(app, pilot, "hello")
                text = lines(app)
                assert "0.0 t/s" not in text
                assert "rate not reported" in text

        drive(go())

    def test_a_server_that_stops_answering_is_said_so_and_dropped(self, monkeypatch):
        session = FakeSession(fails=True)
        talking(monkeypatch, session)

        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                app._session = session
                await ask(app, pilot, "hello")
                assert "stopped answering" in lines(app)
                assert app._session is None
                assert session.closed

        drive(go())

    def test_an_empty_question_is_not_sent(self, monkeypatch):
        session = FakeSession()
        talking(monkeypatch, session)

        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                await ask(app, pilot, "   ")
                assert session.sent == []

        drive(go())

    def test_switching_model_stops_the_server_running_the_old_one(self):
        session = FakeSession()

        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                app._session = session
                app.query_one("#model", Select).value = "qwen2.5:3b"
                await pilot.pause()
                assert session.closed
                assert app._session is None

        drive(go())

    def test_leaving_the_panel_stops_the_server(self):
        session = FakeSession()

        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                app._session = session
            assert session.closed

        drive(go())

    def test_clearing_empties_the_transcript_and_the_history(self, monkeypatch):
        talking(monkeypatch, FakeSession())

        async def go():
            app = Panel(reference="gemma3:1b", interval=60.0)
            async with app.run_test(size=(120, 44)) as pilot:
                await pilot.pause()
                await ask(app, pilot, "hello")
                app.action_clear_chat()
                await pilot.pause()
                assert app.conversation.turns == []
                assert lines(app).strip() == ""

        drive(go())
