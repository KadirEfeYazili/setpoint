"""Chunking measurement tests.

The chunking itself belongs to a library and is not retested here. What is pinned is
the harness around it, and in particular the two ways it was caught lying: charging a
strategy for the instrument's cost, and charging a strategy for VRAM it never touched.
"""

from __future__ import annotations

from pathlib import Path

from setpoint import chunk


def corpus_of(*texts: str, root: Path | None = None) -> chunk.Corpus:
    base = root or Path("memory")
    return chunk.Corpus(
        documents=tuple(
            chunk.Document(path=base / f"{i}.txt", text=text) for i, text in enumerate(texts)
        )
    )


def result(name: str, seconds: float, peak: int | None = None, spread: float = 0.0):
    return chunk.Result(
        strategy=name,
        seconds=seconds,
        chunks=10,
        tokens_median=500.0,
        tokens_p90=512.0,
        peak_vram_mib=peak,
        spread=spread,
    )


class TestCorpus:
    def test_a_missing_path_says_so_rather_than_reading_nothing(self, tmp_path):
        assert chunk.read_corpus(tmp_path / "gone").detail is not None

    def test_a_directory_with_no_text_says_so(self, tmp_path):
        (tmp_path / "picture.png").write_bytes(b"\x89PNG")
        assert chunk.read_corpus(tmp_path).detail is not None

    def test_it_reads_a_single_file(self, tmp_path):
        target = tmp_path / "one.txt"
        target.write_text("hello", encoding="utf-8")
        assert len(chunk.read_corpus(target).documents) == 1

    def test_the_order_is_stable_so_two_runs_compare(self, tmp_path):
        for name in ("b.txt", "a.txt", "c.txt"):
            (tmp_path / name).write_text("text", encoding="utf-8")
        first = [d.path.name for d in chunk.read_corpus(tmp_path).documents]
        second = [d.path.name for d in chunk.read_corpus(tmp_path).documents]
        assert first == second == ["a.txt", "b.txt", "c.txt"]

    def test_a_limit_stops_it_reading_the_whole_thing(self, tmp_path):
        for name in ("a.txt", "b.txt", "c.txt"):
            (tmp_path / name).write_text("x" * 1000, encoding="utf-8")
        assert len(chunk.read_corpus(tmp_path, limit_bytes=1500).documents) == 2

    def test_a_small_corpus_is_marked_unreliable(self):
        assert not corpus_of("tiny").reliable

    def test_a_corpus_past_the_threshold_is_not(self):
        assert corpus_of("x" * (chunk.MIN_CORPUS_BYTES + 1)).reliable

    def test_empty_files_are_skipped(self, tmp_path):
        (tmp_path / "empty.txt").write_text("   \n", encoding="utf-8")
        (tmp_path / "real.txt").write_text("content", encoding="utf-8")
        assert len(chunk.read_corpus(tmp_path).documents) == 1


class TestCalibration:
    def test_it_converts_a_token_budget_into_characters(self):
        # Four characters to a token here, so the ratio has to come back as four.
        body = corpus_of("abcd" * 10_000)
        assert chunk.calibrate(body, lambda text: len(text) // 4) == 4.0

    def test_a_tokenizer_that_answers_nothing_falls_back_rather_than_dividing_by_zero(self):
        assert chunk.calibrate(corpus_of("text"), lambda text: 0) == 4.0

    def test_an_empty_corpus_falls_back_too(self):
        assert chunk.calibrate(chunk.Corpus(), lambda text: len(text)) == 4.0


class TestRunning:
    def test_it_chunks_every_document(self):
        body = corpus_of("one two", "three four")
        strategy = chunk.by_name("token", 100)
        out = chunk.run(strategy, lambda text: text.split(), body, lambda text: 1)
        assert out.chunks == 4

    def test_a_chunker_that_raises_is_reported_not_swallowed(self):
        def broken(text):
            raise RuntimeError("no tokenizer")

        strategy = chunk.by_name("token", 100)
        out = chunk.run(strategy, broken, corpus_of("text"), lambda text: 1)
        assert not out.ok
        assert "no tokenizer" in out.detail

    def test_sizes_are_sampled_rather_than_counted_one_by_one(self):
        # Counting every chunk means a round trip per chunk; the median does not need it.
        body = corpus_of(" ".join(str(i) for i in range(2000)))
        strategy = chunk.by_name("token", 100)
        out = chunk.run(strategy, lambda text: text.split(), body, lambda text: 7, sample_size=50)
        assert out.chunks == 2000
        assert out.sampled == 50
        assert out.tokens_median == 7

    def test_the_sample_is_the_same_between_runs(self):
        # Two runs of the same strategy have to be comparable, so the sample cannot
        # move between them.
        body = corpus_of(" ".join(str(i) for i in range(500)))
        strategy = chunk.by_name("token", 100)
        first = chunk.run(strategy, lambda t: t.split(), body, len, sample_size=20)
        second = chunk.run(strategy, lambda t: t.split(), body, len, sample_size=20)
        assert first.tokens_median == second.tokens_median


class TestPickingARun:
    def test_the_middle_run_is_kept(self):
        picked = chunk.pick([result("token", 3.0), result("token", 1.0), result("token", 2.0)])
        assert picked.seconds == 2.0

    def test_the_spread_between_repetitions_is_carried(self):
        picked = chunk.pick([result("token", 1.0), result("token", 2.0), result("token", 3.0)])
        assert picked.spread == 1.0

    def test_the_worst_peak_is_kept_not_the_middle_one(self):
        # VRAM is a ceiling question: the run that came closest to the edge is the one
        # that says whether this fits.
        picked = chunk.pick([result("s", 1.0, peak=400), result("s", 2.0, peak=900)])
        assert picked.peak_vram_mib == 900

    def test_no_runs_is_a_failure_not_a_zero(self):
        assert not chunk.pick([]).ok


class TestStrategies:
    def test_the_cheap_ones_declare_that_they_run_no_model(self):
        for name in ("token", "recursive", "sentence"):
            assert not chunk.by_name(name, 100).needs_embeddings

    def test_semantic_declares_its_embedding_pass(self):
        strategy = chunk.by_name("semantic", 100)
        assert strategy.needs_embeddings
        assert "embedding pass" in strategy.cost_note

    def test_a_strategy_that_runs_nothing_says_so(self):
        assert chunk.by_name("token", 100).cost_note == "no model runs"

    def test_an_unknown_name_is_none_rather_than_a_guess(self):
        assert chunk.by_name("agentic", 100) is None


class TestThroughput:
    def test_it_is_reported_against_the_corpus_size(self):
        out = result("token", 2.0)
        assert out.throughput(2 * 1024**2) == 1.0

    def test_a_run_that_took_no_measurable_time_is_not_infinitely_fast(self):
        assert result("token", 0.0).throughput(1024**2) == 0.0

    def test_context_per_query_multiplies_by_k(self):
        assert result("token", 1.0).context_per_query(5) == 2500.0
