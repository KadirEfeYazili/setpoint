"""Whether speculative decoding pays on this machine, and for which kind of work.

Speculation trades a batch of drafted tokens verified in one step against the same
tokens decoded one at a time. Two things decide whether that trade wins, and neither is
a property of the speculator: how much cheaper a batch is on this card, and how much of
each draft the target model accepts for the work at hand. The second one moves by a
factor of twenty between prose and an edit, so a single number for a machine would be
worse than no number at all.

Measurement goes through llama-server, because llama-bench cannot speculate. Acceptance
is read from the server's own counters rather than inferred from the timing: a drafter
that never fires looks exactly like one that fires and gets rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .backend import BackendError, find_server_binary
from .measure import MAX_RELIABLE_SPREAD, MIN_RELIABLE_RUNS, Statistic
from .serve import Session

# Speculators that need no draft model, so none of them costs VRAM. On a card that is
# already full, this is the only kind that can be turned on at all.
DRAFT_FREE: tuple[str, ...] = (
    "ngram-simple",
    "ngram-map-k",
    "ngram-map-k4v",
    "ngram-mod",
    "ngram-cache",
)

DEFAULT_MAX_TOKENS = 200
DEFAULT_RUNS = 3
READY_TIMEOUT_S = 180.0
REQUEST_TIMEOUT_S = 300.0
STOP_TIMEOUT_S = 30.0

# Below this a gain is not worth a configuration change: the run-to-run spread on real
# hardware sits near a percent, and a couple of percent moves with the weather.
MIN_WORTH = 0.05

_PASSAGE = (
    "A measurement is a claim about the world that someone else can check. An estimate "
    "is a claim about your own confidence. The two are often written in the same "
    "notation, which is how a guess ends up in a specification. On constrained hardware "
    "the difference is not academic: a budget drawn from an estimate either refuses to "
    "start or silently spills, and both failures look like the tool being wrong."
)


@dataclass(frozen=True)
class Workload:
    """A prompt that stands for a kind of work, and what it is meant to represent."""

    name: str
    prompt: str
    about: str


# Two, because n-gram speculation is a bet on repetition. Prose repeats little; an edit
# repeats almost everything. Anything in between falls between these two numbers.
DEFAULT_WORKLOADS: tuple[Workload, ...] = (
    Workload(
        name="prose",
        prompt="Write a detailed paragraph explaining why measurement beats estimation.",
        about="fresh text, little repetition",
    ),
    Workload(
        name="edit",
        prompt="Repeat the following passage exactly, then add one closing sentence.\n\n"
        + _PASSAGE,
        about="output that quotes its input, as rewriting and refactoring do",
    ),
)


@dataclass(frozen=True)
class Trial:
    """One speculator measured on one workload, or the baseline when `speculator` is None."""

    workload: str
    speculator: str | None
    throughput: Statistic
    drafted: int = 0
    accepted: int = 0
    steps: int = 0
    detail: str | None = None

    @property
    def fired(self) -> bool:
        """Whether the drafter produced anything at all. Zero is a real outcome."""
        return self.drafted > 0

    @property
    def acceptance(self) -> float | None:
        return self.accepted / self.drafted if self.drafted else None

    @property
    def tokens_per_step(self) -> float | None:
        """Tokens the target emitted per verification step, the accepted ones plus its own."""
        return self.accepted / self.steps + 1 if self.steps else None

    @property
    def reliable(self) -> bool:
        spread = self.throughput.spread
        return (
            self.throughput.runs >= MIN_RELIABLE_RUNS
            and spread is not None
            and spread <= MAX_RELIABLE_SPREAD
        )


@dataclass(frozen=True)
class WorkloadResult:
    """Every speculator on one workload, against a baseline measured in the same session."""

    workload: Workload
    baseline: Trial
    trials: tuple[Trial, ...] = field(default_factory=tuple)

    def gain(self, trial: Trial) -> float | None:
        mine, base = trial.throughput.median, self.baseline.throughput.median
        if mine is None or base is None or base == 0:
            return None
        return mine / base - 1

    @property
    def best(self) -> Trial | None:
        """The speculator worth turning on, or `None` when none of them is.

        A drafter that never fired is not eligible whatever its timing says, because a
        speculator that produced nothing cannot have caused a speedup. Unreliable
        readings are not eligible either, however good they look: a 43% spread was
        measured on this hardware, and recommending from it would be recommending noise.
        """
        eligible = [
            (self.gain(t), t)
            for t in self.trials
            if t.fired and t.reliable and (self.gain(t) or 0) > MIN_WORTH
        ]
        if not eligible:
            return None
        return max(eligible, key=lambda pair: pair[0] or 0)[1]


@dataclass(frozen=True)
class Report:
    """What speculation is worth here, per kind of work."""

    model: str
    results: tuple[WorkloadResult, ...] = field(default_factory=tuple)
    detail: str | None = None

    @property
    def winners(self) -> dict[str, str | None]:
        return {r.workload.name: (r.best.speculator if r.best else None) for r in self.results}

    @property
    def agreed(self) -> str | None:
        """The one speculator that won every workload, if there is one.

        When the workloads disagree there is no machine-wide answer, and saying so is
        the finding rather than a failure to conclude.
        """
        picks = set(self.winners.values())
        return picks.pop() if len(picks) == 1 else None


def measure_trial(
    binary: str | Path,
    model_path: str | Path,
    config: object,
    context: int,
    workload: Workload,
    speculator: str | None,
    devices: tuple[str, ...] = (),
    runs: int = DEFAULT_RUNS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Trial:
    """Start a server, discard one warm-up request, measure `runs` more, stop."""
    try:
        with Session(binary, model_path, config, context, devices, speculator) as session:
            session.complete(
                [{"role": "user", "content": workload.prompt}],
                max_tokens=max_tokens,
                temperature=0,
                seed=7,
            )
            samples = [
                rate
                for rate in (
                    session.complete(
                        [{"role": "user", "content": workload.prompt}],
                        max_tokens=max_tokens,
                        temperature=0,
                        seed=7,
                    ).decode_tok_s
                    for _ in range(runs)
                )
                if rate
            ]
            stats = session.counters() if speculator else {}
    except BackendError as exc:
        return Trial(workload.name, speculator, Statistic(()), detail=str(exc))

    return Trial(
        workload=workload.name,
        speculator=speculator,
        throughput=Statistic(tuple(samples)),
        drafted=stats.get("spec_decode_num_draft_tokens_total", 0),
        accepted=stats.get("spec_decode_num_accepted_tokens_total", 0),
        steps=stats.get("spec_decode_num_drafts_total", 0),
        detail=None if samples else "no request returned a timing",
    )


def measure(
    model_path: str | Path,
    config: object,
    context: int,
    devices: tuple[str, ...] = (),
    speculators: tuple[str, ...] = DRAFT_FREE,
    workloads: tuple[Workload, ...] = DEFAULT_WORKLOADS,
    runs: int = DEFAULT_RUNS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    binary: str | Path | None = None,
    on_trial: object = None,
) -> Report:
    """Measure every speculator on every workload, baseline last within each workload."""
    server = binary or find_server_binary()
    if server is None:
        raise BackendError("llama-server was not found")

    results = []
    for workload in workloads:
        trials = []
        for speculator in speculators:
            trial = measure_trial(
                server,
                model_path,
                config,
                context,
                workload,
                speculator,
                devices,
                runs,
                max_tokens,
            )
            trials.append(trial)
            if callable(on_trial):
                on_trial(trial)
        baseline = measure_trial(
            server, model_path, config, context, workload, None, devices, runs, max_tokens
        )
        if callable(on_trial):
            on_trial(baseline)
        results.append(WorkloadResult(workload=workload, baseline=baseline, trials=tuple(trials)))
    return Report(model=str(model_path), results=tuple(results))
