"""The panel's widgets.

Imported only when the tui extra is installed, so nothing here may be referenced from
the rest of setpoint. The app holds no measurement logic: it polls `data.gather` for
what is already known and, when asked, runs the existing `tune` command as a subprocess
and streams what it prints.
"""

from __future__ import annotations

import subprocess
import sys

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    TabbedContent,
    TabPane,
)

from . import data

CSS = """
Screen { layout: vertical; }
#controls { height: auto; padding: 0 1; }
#controls Label { padding: 1 1 0 0; }
#controls Select { width: 34; }
#controls Input { width: 12; }
DataTable { height: auto; max-height: 60%; }
.pane { padding: 0 1; }
.heading { text-style: bold; padding: 1 0 0 0; }
.note { color: $text-muted; }
RichLog { height: 1fr; border: round $panel; }
"""


class Panel(App):
    """One screen for what setpoint measured, and one key to measure more."""

    CSS = CSS
    TITLE = "setpoint"
    BINDINGS = [
        ("q", "quit", "quit"),
        ("r", "refresh", "refresh"),
        ("t", "tune", "run tune"),
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

    def compose(self) -> ComposeResult:
        yield Header()
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
            with TabPane("search", id="search"), VerticalScroll(classes="pane"):
                yield Label(
                    "press t to run `setpoint tune` for the model above. "
                    "the panel measures nothing itself; this runs the command.",
                    classes="note",
                )
                yield RichLog(id="log", markup=False, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
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

    def poll_card(self) -> None:
        """Only the card is re-read on the timer. The rest changes when a command runs."""
        self._fill_rows("#card", data.read_card().rows)

    # --- filling -------------------------------------------------------------------

    def _fill_models(self, snapshot: data.Snapshot) -> None:
        select = self.query_one("#model", Select)
        # Tracked here rather than read back off the widget: the widget's own list is
        # private and has moved between releases.
        if snapshot.models != self._listed:
            select.set_options([(name, name) for name in snapshot.models])
            self._listed = snapshot.models
        if self.reference in snapshot.models:
            select.value = self.reference

    def _fill_rows(self, selector: str, rows: tuple[data.Row, ...]) -> None:
        table = self.query_one(selector, DataTable)
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
        table = self.query_one("#profiles-table", DataTable)
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
        table = self.query_one("#history", DataTable)
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
        self.reference = str(event.value)
        self.refresh_data()

    def on_input_submitted(self, event: Input.Submitted) -> None:
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
