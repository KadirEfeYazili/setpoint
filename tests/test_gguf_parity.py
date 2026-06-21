"""Parity between setpoint's GGUF reader and the reference implementation.

setpoint reads GGUF headers itself because the reference reader materialises every
tokenizer entry, which costs seconds per call. That trade is only defensible while the
two agree, so these tests write a file with the reference writer and read it back both
ways.
"""

from __future__ import annotations

import numpy as np
import pytest

from setpoint.model.reader import ARRAY_DECODE_LIMIT, ArraySummary, read_header

gguf = pytest.importorskip("gguf", reason="reference implementation is a development dependency")


@pytest.fixture(scope="module")
def sample(tmp_path_factory) -> str:
    """A small GGUF file that exercises every metadata shape setpoint cares about."""
    path = tmp_path_factory.mktemp("gguf") / "sample.gguf"
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_name("Parity Sample")
    writer.add_block_count(4)
    writer.add_context_length(8192)
    writer.add_embedding_length(64)
    writer.add_head_count(8)
    writer.add_head_count_kv(2)
    writer.add_key_length(8)
    writer.add_value_length(8)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
    writer.add_rope_freq_base(10000.0)
    writer.add_token_list([f"tok{i}" for i in range(ARRAY_DECODE_LIMIT + 5)])
    writer.add_array("llama.attention.head_count_kv_per_block", [2, 2, 4, 4])

    writer.add_tensor("token_embd.weight", np.zeros((64, 32), dtype=np.float32))
    for block in range(4):
        writer.add_tensor(f"blk.{block}.attn_q.weight", np.zeros((64, 64), dtype=np.float16))
        writer.add_tensor(f"blk.{block}.ffn_down_exps.weight", np.zeros((64, 8), dtype=np.float32))
    writer.add_tensor("output.weight", np.zeros((64, 32), dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return str(path)


class TestQuantTable:
    def test_block_sizes_match_the_reference(self):
        from setpoint.model.reader import QUANT_TYPES

        for type_id, (name, block, block_bytes) in QUANT_TYPES.items():
            reference = gguf.GGMLQuantizationType(type_id)
            assert reference.name == name
            assert gguf.GGML_QUANT_SIZES[reference] == (block, block_bytes)

    def test_every_reference_type_is_known(self):
        from setpoint.model.reader import QUANT_TYPES

        assert {t.value for t in gguf.GGMLQuantizationType} <= set(QUANT_TYPES)


class TestHeaderParity:
    def test_scalar_metadata_matches(self, sample):
        ours = read_header(sample)
        theirs = gguf.GGUFReader(sample)
        for key, value in ours.metadata.items():
            if isinstance(value, ArraySummary):
                continue
            expected = theirs.get_field(key).contents()
            if isinstance(value, float):
                assert value == pytest.approx(expected)
            else:
                assert value == expected

    def test_long_string_array_is_summarised_not_decoded(self, sample):
        tokens = read_header(sample).metadata["tokenizer.ggml.tokens"]
        assert isinstance(tokens, ArraySummary)
        assert tokens.count == ARRAY_DECODE_LIMIT + 5

    def test_short_numeric_array_is_decoded(self, sample):
        value = read_header(sample).metadata["llama.attention.head_count_kv_per_block"]
        assert value == [2, 2, 4, 4]

    def test_tensor_inventory_matches(self, sample):
        ours = read_header(sample)
        theirs = gguf.GGUFReader(sample)
        assert len(ours.tensors) == len(theirs.tensors)
        for mine, reference in zip(ours.tensors, theirs.tensors, strict=True):
            assert mine.name == reference.name
            assert mine.byte_count == reference.n_bytes
            assert mine.element_count == reference.n_elements
            assert mine.quant == gguf.GGMLQuantizationType(reference.tensor_type).name

    def test_data_offset_and_alignment_match(self, sample):
        ours = read_header(sample)
        theirs = gguf.GGUFReader(sample)
        assert ours.alignment == theirs.alignment
        assert ours.data_offset == theirs.data_offset
