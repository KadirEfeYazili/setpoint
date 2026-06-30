"""The search.

Not a blind Bayesian sweep. The budgeter already knows roughly where the answer is, so
the search starts from its seeds, screens them cheaply, and then walks one parameter at
a time from the survivor. Every step is recorded, and interrupting the search keeps
whatever it had found.

    warm up -> successive halving -> coordinate descent -> confirm, then baseline
"""

# A GPU ramps its clocks over the first minute of sustained load, so the first
# measurements of a session read low. llama-bench discards a warmup pass inside each
# invocation, which does not help across them; this is the session-level equivalent.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from ..profile import Config
from .types import (
    Effort,
    Measure,
    SearchResult,
    SearchSpace,
    Stage,
    Step,
    Trial,
    Verdict,
)

# A move has to beat the noise to count as a move. The floor applies when both
# measurements were tight; otherwise their spread sets the bar.
MIN_IMPROVEMENT = 0.02

# Safety stop for coordinate descent, not a tuning knob: the descent ends when a full
# sweep of the neighbourhood improves nothing. This only bounds a pathological landscape.
MAX_MOVES = 32

# The search is meant to take minutes. Past this many measurements it stops and reports
# the best it has, rather than running all night.
DEFAULT_MEASUREMENT_BUDGET = 60


def is_improvement(current: Trial, candidate: Trial) -> bool:
    """Whether the candidate is better by more than the two measurements can explain."""
    if candidate.score is None:
        return False
    if current.score is None or current.score <= 0:
        return True
    gain = candidate.score / current.score - 1
    noise = ((current.spread or 0.0) + (candidate.spread or 0.0)) / 2
    return gain > max(MIN_IMPROVEMENT, noise)


def neighbours(config: Config, space: SearchSpace) -> list[tuple[str, Config]]:
    """Configurations one move away, in the order the descent tries them.

    Layer count comes first because it dominates: on a card that cannot hold the whole
    model, everything else is a second-order effect.
    """
    moves: list[tuple[str, Config]] = []

    if config.n_gpu_layers is not None:
        for delta in (1, -1, 2, -2):
            layers = config.n_gpu_layers + delta
            if 0 <= layers <= space.max_gpu_layers:
                moves.append((f"-ngl {layers}", replace(config, n_gpu_layers=layers)))

    if space.moe and config.n_cpu_moe is not None:
        for delta in (1, -1):
            moe = config.n_cpu_moe + delta
            if 0 <= moe <= space.max_gpu_layers:
                moves.append((f"-ncmoe {moe}", replace(config, n_cpu_moe=moe)))

    for size in space.ubatch_sizes:
        if size != config.ubatch_size:
            moves.append((f"-ub {size}", replace(config, ubatch_size=size)))

    for size in space.batch_sizes:
        if size != config.batch_size:
            moves.append((f"-b {size}", replace(config, batch_size=size)))

    if config.threads is not None:
        for delta in (2, -2):
            threads = config.threads + delta
            if 1 <= threads <= space.max_threads:
                moves.append((f"-t {threads}", replace(config, threads=threads)))

    if space.tune_flash_attn and config.flash_attn is not None:
        flipped = not config.flash_attn
        moves.append(
            (f"flash attention {'on' if flipped else 'off'}", replace(config, flash_attn=flipped))
        )

    return moves


def search(
    seeds: list[Config],
    space: SearchSpace,
    measure: Measure,
    screen_effort: Effort,
    full_effort: Effort,
    baseline: Config | None = None,
    measurement_budget: int = DEFAULT_MEASUREMENT_BUDGET,
    on_step: Callable[[Step], None] | None = None,
) -> SearchResult:
    """Run the whole search. Interrupting it returns the best result found so far.

    `on_step` sees each decision as it is made. A search takes minutes, and a caller
    that cannot show progress leaves the user watching a blank terminal.
    """
    steps: list[Step] = []
    state = _State(measure=measure, steps=steps, budget=measurement_budget, on_step=on_step)

    if not seeds:
        return SearchResult(None, None, tuple(steps), 0, False, "no seed configurations")

    try:
        state.run(baseline or seeds[0], screen_effort, Stage.WARMUP, force=True)

        survivor = _screen(seeds, screen_effort, state)
        if survivor is None:
            return SearchResult(
                None,
                None,
                tuple(steps),
                state.count,
                False,
                "no seed configuration produced a measurement",
            )

        best = _descend(survivor, space, screen_effort, state)
        confirmed = state.run(best.config, full_effort, Stage.CONFIRM, force=True)
        winner = confirmed if confirmed.usable else best

        # The baseline is measured last, next to the winner, so the two numbers that
        # form the speedup claim were taken in the same thermal state.
        baseline_trial = None
        if baseline is not None:
            baseline_trial = state.run(baseline, full_effort, Stage.BASELINE, force=True)
        if not confirmed.usable:
            reason = "the winning configuration failed its confirmation run"
        elif state.exhausted:
            reason = f"stopped after {state.count} measurements"
        else:
            reason = "search finished"
        return SearchResult(winner, baseline_trial, tuple(steps), state.count, False, reason)

    except KeyboardInterrupt:
        # Accepting the best result so far is the point of allowing the interrupt.
        best = state.best_so_far()
        return SearchResult(
            best,
            state.baseline,
            tuple(steps),
            state.count,
            True,
            "interrupted; reporting the best configuration measured so far",
        )


def _screen(seeds: list[Config], effort: Effort, state: _State) -> Trial | None:
    """Successive halving: measure the pool, keep the better half, repeat."""
    pool = [state.run(config, effort, Stage.SCREEN) for config in seeds]
    pool = [t for t in pool if t.usable]
    if not pool:
        return None

    while len(pool) > 1:
        pool.sort(key=lambda t: t.score or 0.0, reverse=True)
        keep = max(1, len(pool) // 2)
        for dropped in pool[keep:]:
            state.note(Stage.SCREEN, dropped, Verdict.DROPPED, "slower than the survivors")
        pool = pool[:keep]

    state.note(Stage.SCREEN, pool[0], Verdict.KEPT, "best of the seeds")
    return pool[0]


def _descend(start: Trial, space: SearchSpace, effort: Effort, state: _State) -> Trial:
    """Shift one parameter at a time, keeping any move that beats the noise."""
    current = start
    for _ in range(MAX_MOVES):
        improved = False
        for label, candidate_config in neighbours(current.config, space):
            if state.exhausted:
                return current
            if state.already_tried(candidate_config, effort):
                continue
            candidate = state.run(candidate_config, effort, Stage.DESCEND, quiet=True)
            if is_improvement(current, candidate):
                state.note(Stage.DESCEND, candidate, Verdict.IMPROVED, label)
                current = candidate
                improved = True
                break
            state.note(Stage.DESCEND, candidate, Verdict.NO_BETTER, label)
        if not improved:
            break
    return current


class _State:
    """Bookkeeping shared by the phases: the trace, the trial cache, the running best."""

    def __init__(
        self,
        measure: Measure,
        steps: list[Step],
        budget: int,
        on_step: Callable[[Step], None] | None = None,
    ) -> None:
        self._measure = measure
        self._steps = steps
        self._budget = budget
        self._on_step = on_step
        # Keyed by effort as well as configuration: the same settings measured briefly
        # and measured properly are two different results, and only one may be reported.
        self._seen: dict[tuple[Config, Effort], Trial] = {}
        self.discarded: set[tuple[Config, Effort]] = set()
        self.count = 0
        self.baseline: Trial | None = None

    @property
    def exhausted(self) -> bool:
        return self.count >= self._budget

    def run(
        self,
        config: Config,
        effort: Effort,
        stage: Stage,
        quiet: bool = False,
        force: bool = False,
    ) -> Trial:
        key = (config, effort)
        cached = self._seen.get(key)
        if cached is not None and not force:
            return cached
        trial = self._measure(config, effort)
        self.count += 1
        self._seen[key] = trial
        if stage is Stage.BASELINE:
            self.baseline = trial
        if stage is Stage.WARMUP:
            self.discarded.add(key)
        if not quiet:
            verdict = Verdict.KEPT if trial.usable else Verdict.FAILED
            self.note(stage, trial, verdict, trial.detail)
        return trial

    def already_tried(self, config: Config, effort: Effort) -> bool:
        return (config, effort) in self._seen

    def note(self, stage: Stage, trial: Trial, verdict: Verdict, note: str = "") -> None:
        step = Step(stage, trial.config, trial.score, verdict, note)
        self._steps.append(step)
        if self._on_step is not None:
            self._on_step(step)

    def best_so_far(self) -> Trial | None:
        usable = [
            trial
            for key, trial in self._seen.items()
            if trial.usable and trial is not self.baseline and key not in self.discarded
        ]
        return max(usable, key=lambda t: t.score or 0.0) if usable else None
