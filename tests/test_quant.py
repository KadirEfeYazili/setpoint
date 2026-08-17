"""Quantization comparison tests.

The divergence pass itself needs two models and a corpus. What is pinned here is the
reading of its report and the two judgements around it: which local files are the same
model, and how large the reference logits will be before anything is written.
"""

from __future__ import annotations

from pathlib import Path

from setpoint import quant
from setpoint.model.types import Attention, ModelInfo, Weights

# The real report, as llama-perplexity wrote it for Q4_K_M against Q8_0.
REPORT = """
====== Perplexity statistics ======
Mean PPL(Q)                   :   7.141132 +/-   0.681569
Mean PPL(base)                :   6.965427 +/-   0.660250
Mean ln(PPL(Q)/PPL(base))     :   0.024912 +/-   0.014146
Mean PPL(Q)/PPL(base)         :   1.025225 +/-   0.014503

====== KL divergence statistics ======
Mean    KLD:   0.070951 +/-   0.006884
Maximum KLD:   2.169640
Median  KLD:   0.015033
Minimum KLD:   0.000009

====== Token probability statistics ======
Mean    dp: -0.418 +/- 0.228 %
RMS dp    :  7.296 +/- 0.537 %
Same top p: 90.098 +/- 0.936 %
"""


def model(
    architecture: str = "gemma3",
    blocks: int = 26,
    embedding: int = 1152,
    vocab: int | None = 262144,
    file_type: str = "Q4_K_M",
) -> ModelInfo:
    return ModelInfo(
        path=Path(f"{file_type}.gguf"),
        file_bytes=0,
        architecture=architecture,
        name="Test",
        block_count=blocks,
        embedding_length=embedding,
        attention=Attention((4,) * blocks, (1,) * blocks, 256, 256),
        weights=Weights((0,) * blocks, (0,) * blocks, 0, 0),
        parameter_count=999_885_952,
        quant_mix=(),
        vocab_size=vocab,
        file_type=file_type,
    )


class TestSameModel:
    def test_two_quantizations_of_one_model_match(self):
        assert quant.same_model(model(file_type="Q4_K_M"), model(file_type="Q8_0"))

    def test_a_different_architecture_does_not_match(self):
        assert not quant.same_model(model(), model(architecture="qwen3"))

    def test_a_different_depth_does_not_match(self):
        assert not quant.same_model(model(), model(blocks=36))

    def test_a_different_width_does_not_match(self):
        assert not quant.same_model(model(), model(embedding=2048))

    def test_a_different_vocabulary_does_not_match(self):
        # Same architecture and shape but a different tokenizer is a different model.
        assert not quant.same_model(model(), model(vocab=151936))


class TestLogitsSize:
    def test_it_projects_the_size_measured_on_this_hardware(self):
        # Measured: 510 MiB of logits for 4 chunks of 512 tokens at vocab 262144.
        projected = quant.logits_bytes(262144, chunks=4, context=512)
        assert projected is not None
        assert 500 <= projected / 1024**2 <= 520

    def test_it_grows_with_the_vocabulary(self):
        small = quant.logits_bytes(32000, 4, 512)
        large = quant.logits_bytes(64000, 4, 512)
        assert large == 2 * small

    def test_it_grows_with_the_corpus(self):
        assert quant.logits_bytes(32000, 8, 512) == 2 * quant.logits_bytes(32000, 4, 512)

    def test_a_model_that_does_not_report_a_vocabulary_cannot_be_projected(self):
        assert quant.logits_bytes(None, 4, 512) is None


class TestReport:
    def read(self, text: str = REPORT) -> quant.Quality:
        return quant.parse_quality(text, variant="q4", reference_variant="q8", corpus="c", chunks=4)

    def test_it_reads_the_perplexity_ratio_and_its_error(self):
        quality = self.read()
        assert quality.ppl_ratio == 1.025225
        assert quality.ppl_ratio_error == 0.014503

    def test_column_padding_is_not_structure(self):
        # The report pads its labels to line up, and the padding has changed between
        # releases. Matching on exact spacing would break on the next one.
        padded = "Mean    PPL(Q)/PPL(base)  :  1.025225 +/- 0.014503"
        assert self.read(padded).ppl_ratio == 1.025225

    def test_it_prefers_the_median_divergence_over_the_mean(self):
        # The mean is dragged by a maximum of 2.17; the median is the typical token.
        assert self.read().median_kld == 0.015033

    def test_it_reads_how_often_the_top_token_agrees(self):
        assert self.read().same_top_pct == 90.098

    def test_it_reads_the_probability_shift_whatever_the_codec_did_to_the_delta(self):
        # The locale codec on this machine cannot represent the delta at all, so the
        # label arrives mangled and the figure still has to be readable.
        for label in ("RMS Δp    :  7.296", "RMS ?p    :  7.296", "RMS \xce\x94p :  7.296"):
            assert self.read(label).rms_delta_p_pct == 7.296

    def test_a_report_with_no_figures_says_so_rather_than_reading_zero(self):
        quality = self.read("the run said nothing useful")
        assert not quality.measured
        assert quality.detail

    def test_a_partial_report_keeps_what_it_found(self):
        quality = self.read("Same top p: 90.098 +/- 0.936 %")
        assert quality.measured
        assert quality.same_top_pct == 90.098
        assert quality.ppl_ratio is None
