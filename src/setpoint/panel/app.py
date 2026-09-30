"""The panel's widgets.

Imported only when the tui extra is installed, so nothing here may be referenced from
the rest of setpoint. The app holds no measurement logic: it polls `data.gather` for
what is already known and, when asked, runs the existing commands rather than
reimplementing them -- `tune` as a subprocess, and the chat as a client to a server
started from the stored profile.
"""

from __future__ import annotations

import subprocess
import sys
import time

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    DataTable,
    Footer,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from .. import banner, chat, serve
from ..backend import BackendError, find_server_binary
from ..hardware import GpuWatcher
from . import data

CSS = """
Screen { layout: vertical; }
/* Docked, so it stays put while the panes below it scroll. A mark that scrolls out
   of view is a mark that is not there. */
#banner { dock: top; height: auto; padding: 1 0 0 0; }
#controls { height: auto; padding: 0 1; }
#controls Label { padding: 1 1 0 0; }
#controls Select { width: 34; }
#controls Input { width: 12; }
DataTable { height: auto; max-height: 60%; }
.pane { padding: 0 1; }
.heading { text-style: bold; padding: 1 0 0 0; }
.note { color: $text-muted; }
RichLog { height: 1fr; border: round $panel; }
#transcript { height: 1fr; }
#live { color: $text-muted; padding: 0 1; height: auto; }
#say { dock: bottom; }
.chat { padding: 0 1; height: 1fr; }
"""

# The stream is copied into a widget, and repainting per token would spend more time
# drawing than the model spends generating.
LIVE_REFRESH_S = 0.08


class Panel(App):
    """One screen for what setpoint measured, and one key to measure more."""

    CSS = CSS
    TITLE = "setpoint"
    BINDINGS = [
        ("q", "quit", "quit"),
        ("r", "refresh", "refresh"),
        ("t", "tune", "run tune"),
        ("ctrl+l", "clear_chat", "clear chat"),
    ]

    def __init__(
        self, reference: str | None = None, context: int = 4096, interval: float = 2.0
    ) -> None:
        super().__init__()
        self.reference = reference
        self.context = context
        self.interval = max(0.5, interval)
        self.snapshot: data.Snapshot | None = None
        self._listed: tuple[str, ...] = ()
        self.conversation = chat.Conversation()
        self.target: data.ChatTarget | None = None
        self._session: serve.Session | None = None
        self._count = chat.estimate_tokens

    def compose(self) -> ComposeResult:
        yield Static(id="banner")
        with TabbedContent(initial="budget"):
            with TabPane("budget", id="budget"), VerticalScroll(classes="pane"):
                with Horizontal(id="controls"):
                    yield Label("model")
                    yield Select([], id="model", allow_blank=True, prompt="pick a model")
                    yield Label("context")
                    yield Input(str(self.context), id="context", type="integer")
                yield Label("card", classes="heading")
                yield DataTable(id="card", show_header=False, cursor_type="none")
                yield Label("budget", classes="heading")
                yield DataTable(id="terms", show_header=False, cursor_type="none")
                yield Label("plan", classes="heading")
                yield DataTable(id="plan", show_header=False, cursor_type="none")
                yield Label("instead", classes="heading")
                yield DataTable(id="alternatives", show_header=False, cursor_type="none")
            with TabPane("profiles", id="profiles"), VerticalScroll(classes="pane"):
                yield DataTable(id="profiles-table", cursor_type="row")
                yield Label("detail", classes="heading")
                yield DataTable(id="profile-detail", show_header=False, cursor_type="none")
                yield Label("history", classes="heading")
                yield DataTable(id="history", cursor_type="none")
            with TabPane("residency", id="residency"), VerticalScroll(classes="pane"):
                yield Label("on the card now", classes="heading")
                yield DataTable(id="residency-card", show_header=False, cursor_type="none")
                yield Label("holding it", classes="heading")
                yield DataTable(id="residency-processes", show_header=False, cursor_type="none")
                yield Label("what the policy would do", classes="heading")
                yield DataTable(id="residency-models", show_header=False, cursor_type="none")
            with TabPane("chat", id="chat"), Vertical(classes="chat"):
                yield Label("", id="chat-status", classes="note")
                yield RichLog(id="transcript", markup=True, wrap=True)
                yield Static("", id="live")
                yield Input(placeholder="ask it something", id="say")
            with TabPane("search", id="search"), VerticalScroll(classes="pane"):
                yield Label(
                    "press t to run `setpoint tune` for the model above. "
                    "the panel measures nothing itself; this runs the command.",
                    classes="note",
                )
                yield RichLog(id="log", markup=False, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.paint_banner()
        self.query_one("#profiles-table", DataTable).add_columns(
            "id", "model", "ctx", "-ngl", "decode", "vs base", "speculator"
        )
        self.query_one("#history", DataTable).add_columns("when", "decode", "note")
        self.refresh_data()
        self.set_interval(self.interval, self.poll_card)

    # --- gathering -----------------------------------------------------------------

    def action_refresh(self) -> None:
        self.refresh_data()

    def refresh_data(self) -> None:
        snapshot = data.gather(self.reference, self.context)
        self.snapshot = snapshot
        if self.reference is None and snapshot.budget is not None:
            self.reference = snapshot.budget.model
        self._fill_models(snapshot)
        self._fill_rows("#card", snapshot.card.rows)
        self._fill_budget(snapshot)
        self._fill_profiles(snapshot)
        self._fill_residency()
        self._fill_chat_status()

    def on_resize(self, _: object) -> None:
        """The mark is chosen from the width, so it is chosen again when that changes."""
        self.paint_banner()

    def paint_banner(self) -> None:
        """The whole mark, docked at the top left.

        Whole, at whatever size fits: a narrower window drops to the small mark rather
        than to a shortened wordmark, because a wordmark missing its lower half reads as
        broken rather than as small. Left aligned, so it sits in the same place at every
        width instead of sliding as the window changes.

        A resize arrives before the widget is mounted, so this asks rather than assumes:
        raising inside an event handler stalls the whole loop.
        """
        found = self.query("#banner")
        if not found:
            return
        found.first(Static).update(
            banner.markup(max(self.size.width, 8), tagline=False, face=self.mark_colour())
        )

    def mark_colour(self) -> str:
        """The scrollbar's lit blue, read from the theme so the mark tracks it.

        The resting scrollbar colour is too dark to fill a letterform with: it measures
        about 1.5:1 against the background where a solid block needs three.
        """
        variables = getattr(self, "theme_variables", None) or {}
        return str(variables.get("scrollbar-active") or banner.HEX[banner.FACE_COLOUR])

    def poll_card(self) -> None:
        """Only the card is re-read on the timer. The rest changes when a command runs."""
        self._fill_rows("#card", data.read_card().rows)
        if self.query("TabbedContent").first(TabbedContent).active == "residency":
            # Re-read only while it is the pane being looked at: the process list costs
            # a second NVML round trip and nothing else on screen needs it.
            self._fill_residency()

    # --- filling -------------------------------------------------------------------

    def _fill_models(self, snapshot: data.Snapshot) -> None:
        found = self.query("#model")
        if not found:
            return
        select = found.first(Select)
        # Tracked here rather than read back off the widget: the widget's own list is
        # private and has moved between releases.
        if snapshot.models != self._listed:
            select.set_options([(name, name) for name in snapshot.models])
            self._listed = snapshot.models
        if self.reference in snapshot.models:
            select.value = self.reference

    def _fill_rows(self, selector: str, rows: tuple[data.Row, ...]) -> None:
        # Asked for, not assumed. An event can arrive before the widget is mounted or
        # after it is gone, and raising inside a handler stalls the whole loop.
        found = self.query(selector)
        if not found:
            return
        table = found.first(DataTable)
        if not table.columns:
            table.add_columns("label", "value", "note")
        table.clear()
        for row in rows:
            table.add_row(row.label, row.value, row.note)

    def _fill_budget(self, snapshot: data.Snapshot) -> None:
        view = snapshot.budget
        if view is None:
            self._fill_rows("#terms", (data.Row("budget", "-", "no model chosen"),))
            self._fill_rows("#plan", ())
            self._fill_rows("#alternatives", ())
            return
        if view.detail:
            self._fill_rows("#terms", (data.Row("budget", "-", view.detail),))
            self._fill_rows("#plan", ())
            self._fill_rows("#alternatives", ())
            return
        self._fill_rows("#terms", view.rows)
        self._fill_rows("#plan", view.plan_rows)
        self._fill_rows("#alternatives", view.alternatives)

    def _fill_profiles(self, snapshot: data.Snapshot) -> None:
        found = self.query("#profiles-table")
        if not found:
            return
        table = found.first(DataTable)
        table.clear()
        for view in snapshot.profiles:
            table.add_row(
                view.signature_id[:12],
                view.model[:28],
                str(view.context),
                str(view.n_gpu_layers) if view.n_gpu_layers is not None else "-",
                f"{view.decode_tok_s:.2f}" if view.decode_tok_s else "-",
                f"{view.speedup:.2f}x" if view.speedup else "-",
                view.speculator or "-",
                key=view.signature_id,
            )
        if snapshot.profiles:
            self._show_profile(snapshot.profiles[0])

    def _show_profile(self, view: data.ProfileView) -> None:
        self._fill_rows("#profile-detail", view.rows)
        found = self.query("#history")
        if not found:
            return
        table = found.first(DataTable)
        table.clear()
        for entry in data.read_history(view.signature_id):
            table.add_row(
                entry.when,
                f"{entry.decode_tok_s:.2f}" if entry.decode_tok_s else "-",
                entry.note,
            )

    # --- events --------------------------------------------------------------------

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "model" or event.value is Select.BLANK:
            return
        if str(event.value) != self.reference:
            # The open server is running the previous model's configuration, so it is
            # the wrong thing to keep talking to.
            self.close_session()
        self.reference = str(event.value)
        self.refresh_data()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "say":
            self.on_said(event)
            return
        if event.input.id != "context":
            return
        try:
            self.context = max(1, int(event.value))
        except ValueError:
            return
        self.refresh_data()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "profiles-table" or self.snapshot is None:
            return
        key = event.row_key.value
        view = next((v for v in self.snapshot.profiles if v.signature_id == key), None)
        if view is not None:
            self._show_profile(view)

    def _fill_residency(self) -> None:
        view = data.read_residency()
        if view.detail:
            self._fill_rows("#residency-card", (data.Row("card", "-", view.detail),))
            self._fill_rows("#residency-processes", ())
            self._fill_rows("#residency-models", ())
            return
        self._fill_rows("#residency-card", view.rows)
        self._fill_rows("#residency-processes", view.processes)
        self._fill_rows("#residency-models", view.models)

    # --- talking to it -------------------------------------------------------------

    def _fill_chat_status(self) -> None:
        """What the conversation would run on, before anything is started."""
        found = self.query("#chat-status")
        if not found:
            return
        self.target = data.read_chat_target(self.reference) if self.reference else None
        found.first(Label).update(self._chat_status())

    def _chat_status(self) -> str:
        target = self.target
        if target is None:
            return "pick a model first"
        if not target.ready:
            return f"{target.reference}: {target.detail}. press t to measure it"
        state = "running" if self._session is not None else "starts on the first message"
        claimed = f"{target.claimed_tok_s:.2f} t/s" if target.claimed_tok_s else "unmeasured"
        speculator = f", {target.speculator}" if target.speculator else ""
        return (
            f"{target.label}   c{target.context}{speculator}   profile says {claimed}   [{state}]"
        )

    def on_said(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if self.target is None or not self.target.ready:
            self.write_chat("[dim]there is no measured profile to talk on[/dim]")
            return
        event.input.value = ""
        self.write_chat(f"[bold]you[/bold]  {text}")
        self.say(text)

    def action_clear_chat(self) -> None:
        self.conversation.clear()
        found = self.query("#transcript")
        if found:
            found.first(RichLog).clear()

    def write_chat(self, line: str) -> None:
        found = self.query("#transcript")
        if found:
            found.first(RichLog).write(line)

    def set_live(self, text: str) -> None:
        found = self.query("#live")
        if found:
            found.first(Static).update(text)

    def close_session(self) -> None:
        session, self._session = self._session, None
        self.conversation.clear()
        self._count = chat.estimate_tokens
        if session is not None:
            session.__exit__(None, None, None)

    def on_unmount(self) -> None:
        self.close_session()

    @work(thread=True, exclusive=True)
    def say(self, text: str) -> None:
        """Send one message and stream the answer back into the transcript."""
        target = self.target
        if target is None:
            return
        session = self._session or self._start_session(target)
        if session is None:
            return

        self.conversation.add(chat.Turn("user", text))
        dropped = chat.trim(self.conversation, target.context, self._count)
        if dropped:
            self.call_from_thread(
                self.write_chat, f"[dim]dropped {dropped} older turn(s) to stay in context[/dim]"
            )

        watcher = GpuWatcher()
        pieces: list[str] = []
        painted = 0.0
        try:
            with watcher:
                for piece in session.stream(self.conversation.messages()):
                    pieces.append(piece)
                    now = time.monotonic()
                    if now - painted >= LIVE_REFRESH_S:
                        painted = now
                        self.call_from_thread(self.set_live, "".join(pieces))
        except (OSError, ValueError) as exc:
            self.call_from_thread(self.set_live, "")
            self.call_from_thread(
                self.write_chat, f"[dim]the server stopped answering: {exc}[/dim]"
            )
            self.close_session()
            return

        answer = "".join(pieces)
        measured = session.last
        turn = chat.Turn(
            role="assistant",
            text=answer,
            decode_tok_s=measured.decode_tok_s if measured else None,
            tokens=(measured.tokens if measured and measured.tokens else self._count(answer)),
            peak_vram_mib=watcher.result.peak_vram_mib,
            speculator=target.speculator,
        )
        self.conversation.add(turn)
        self.call_from_thread(self.set_live, "")
        self.call_from_thread(self.write_chat, answer)
        cost = chat.cost_line(turn, target.claimed_tok_s)
        if cost:
            self.call_from_thread(self.write_chat, f"[dim]{cost}[/dim]")

    def _start_session(self, target: data.ChatTarget) -> serve.Session | None:
        """Start the server on the stored configuration, saying so while it loads."""
        binary = find_server_binary()
        if binary is None:
            self.call_from_thread(self.write_chat, "[dim]llama-server was not found[/dim]")
            return None
        devices, chosen = serve.pin_device(target.gpu)
        if chosen is not None:
            # Printed because ids are positional: a reboot renumbers them and a wrong
            # card is otherwise invisible until the rate collapses.
            self.call_from_thread(
                self.write_chat, f"[dim]starting on {chosen.id} -- {chosen.name}[/dim]"
            )
        session = serve.Session(
            binary=binary,
            model_path=target.model_path,
            config=target.config,
            context=target.context,
            devices=devices,
        )
        try:
            session.__enter__()
        except BackendError as exc:
            self.call_from_thread(self.write_chat, f"[dim]{exc}[/dim]")
            return None
        self._session = session
        self._count = chat.token_counter(session.base)
        self.call_from_thread(self._fill_chat_status)
        return session

    # --- running the real command --------------------------------------------------

    def action_tune(self) -> None:
        if not self.reference:
            return
        self.query_one(TabbedContent).active = "search"
        self.stream_tune(self.reference, self.context)

    @work(thread=True, exclusive=True)
    def stream_tune(self, reference: str, context: int) -> None:
        """Run `setpoint tune` and copy its output into the log as it arrives."""
        argv = [sys.executable, "-m", "setpoint", "tune", reference, "-c", str(context)]
        log = self.query_one("#log", RichLog)
        self.call_from_thread(log.write, "$ " + " ".join(argv[2:]))
        try:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.call_from_thread(log.write, f"could not start it: {exc}")
            return
        assert process.stdout is not None
        for line in process.stdout:
            self.call_from_thread(log.write, line.rstrip())
        process.wait()
        self.call_from_thread(log.write, f"[exit {process.returncode}]")
        self.call_from_thread(self.refresh_data)
