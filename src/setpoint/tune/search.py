"""The search.

Not a blind Bayesian sweep. The budgeter already knows roughly where the answer is, so
the search starts from its seeds, screens them cheaply, and then walks one parameter at
a time from the survivor. Every step is recorded, and interrupting the search keeps
whatever it had found.

    seeds -> successive halving -> coordinate descent -> confirm at full effort
"""

from __future__ import annotations

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

# Coordinate descent stops when a whole pass changes nothing, or at this many passes.
MAX_PASSES = 4


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
        moves.append((f"flash attention {'on' if flipped else 'off'}",
                      replace(config, flash_attn=flipped)))

    return moves


def search(
    seeds: list[Config],
    space: SearchSpace,
    measure: Measure,
    screen_effort: Effort,
    full_effort: Effort,
    baseline: Config | None = None,
) -> SearchResult:
    """Run the whole search. Interrupting it returns the best result found so far."""
    steps: list[Step] = []
    state = _State(measure=measure, steps=steps)

    if not seeds:
        return SearchResult(None, None, tuple(steps), 0, False, "no seed configurations")

    try:
        baseline_trial = None
        if baseline is not None:
            baseline_trial = state.run(baseline, full_effort, Stage.BASELINE)

        survivor = _screen(seeds, screen_effort, state)
        if survivor is None:
            return SearchResult(
                None, baseline_trial, tuple(steps), state.count, False,
                "no seed configuration produced a measurement",
            )

        best = _descend(survivor, space, screen_effort, state)
        confirmed = state.run(best.config, full_effort, Stage.CONFIRM)
        winner = confirmed if confirmed.usable else best
        reason = "search finished" if confirmed.usable else "final measurement failed"
        return SearchResult(winner, baseline_trial, tuple(steps), state.count, False, reason)

    except KeyboardInterrupt:
        # Accepting the best result so far is the point of allowing the interrupt.
        best = state.best_so_far()
        return SearchResult(
            best, state.baseline, tuple(steps), state.count, True,
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
    for _ in range(MAX_PASSES):
        improved = False
        for label, candidate_config in neighbours(current.config, space):
            if state.already_tried(candidate_config):
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

    def __init__(self, measure: Measure, steps: list[Step]) -> None:
        self._measure = measure
        self._steps = steps
        self._seen: dict[Config, Trial] = {}
        self.count = 0
        self.baseline: Trial | None = None

    def run(self, config: Config, effort: Effort, stage: Stage, quiet: bool = False) -> Trial:
        cached = self._seen.get(config)
        if cached is not None:
            return cached
        trial = self._measure(config, effort)
        self.count += 1
        self._seen[config] = trial
        if stage is Stage.BASELINE:
            self.baseline = trial
        if not quiet:
            verdict = Verdict.KEPT if trial.usable else Verdict.FAILED
            self.note(stage, trial, verdict, trial.detail)
        return trial

    def already_tried(self, config: Config) -> bool:
        return config in self._seen

    def note(self, stage: Stage, trial: Trial, verdict: Verdict, note: str = "") -> None:
        self._steps.append(Step(stage, trial.config, trial.score, verdict, note))

    def best_so_far(self) -> Trial | None:
        usable = [t for t in self._seen.values() if t.usable and t is not self.baseline]
        return max(usable, key=lambda t: t.score or 0.0) if usable else None
