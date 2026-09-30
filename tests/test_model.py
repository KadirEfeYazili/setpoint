"""Model analysis tests.

These build headers directly instead of writing files, so they cover the grouping and
fallback rules rather than the byte-level parsing that `test_gguf_parity.py` pins.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from setpoint.model import ModelError, describe, resolve, stores
from setpoint.model.reader import ArraySummary, GgufHeader, TensorEntry


def tensor(name: str, byte_count: int, quant: str = "Q4_K", elements: int = 0) -> TensorEntry:
    return TensorEntry(
        name=name,
        dimensions=(byte_count,),
        quant=quant,
        element_count=elements or byte_count,
        byte_count=byte_count,
        offset=0,
    )


def header(metadata: dict[str, object], tensors: tuple[TensorEntry, ...]) -> GgufHeader:
    return GgufHeader(
        path=Path("model.gguf"),
        version=3,
        alignment=32,
        metadata=metadata,
        tensors=tensors,
        data_offset=0,
        file_bytes=sum(t.byte_count for t in tensors),
    )


def dense_header(**overrides: object) -> GgufHeader:
    metadata: dict[str, object] = {
        "general.architecture": "llama",
        "general.name": "Test",
        "general.file_type": 15,
        "llama.block_count": 3,
        "llama.embedding_length": 64,
        "llama.context_length": 4096,
        "llama.attention.head_count": 8,
        "llama.attention.head_count_kv": 2,
    }
    metadata.update(overrides)
    tensors = (
        tensor("token_embd.weight", 500),
        *(tensor(f"blk.{i}.attn_q.weight", 100) for i in range(3)),
        tensor("output.weight", 300),
        tensor("output_norm.weight", 10, quant="F32"),
    )
    return header(metadata, tensors)


class TestRequiredMetadata:
    def test_missing_architecture_is_an_error(self):
        with pytest.raises(ModelError):
            describe(header({}, ()))

    def test_missing_block_count_is_an_error(self):
        with pytest.raises(ModelError):
            describe(header({"general.architecture": "llama"}, ()))


class TestWeightGrouping:
    def test_blocks_embedding_and_output_are_separated(self):
        weights = describe(dense_header()).weights
        assert weights.block_bytes == (100, 100, 100)
        assert weights.input_bytes == 500
        assert weights.output_bytes == 310
        assert weights.total_bytes == 1110

    def test_expert_tensors_are_tracked_inside_their_block(self):
        head = dense_header(**{"llama.expert_count": 8, "llama.expert_used_count": 2})
        head = header(
            head.metadata,
            (*head.tensors, tensor("blk.1.ffn_down_exps.weight", 900)),
        )
        info = describe(head)
        assert info.is_moe
        assert info.experts.used == 2
        assert info.weights.expert_bytes == (0, 900, 0)
        assert info.weights.block_bytes[1] == 1000

    def test_tensors_beyond_the_block_count_are_counted_as_resident(self):
        head = dense_header()
        info = describe(header(head.metadata, (*head.tensors, tensor("blk.9.attn_q.weight", 42))))
        assert info.weights.output_bytes == 352
        assert any("beyond the declared block count" in n for n in info.notes)


class TestTiedEmbedding:
    def test_a_separate_output_projection_means_untied(self):
        assert not describe(dense_header()).weights.tied_embedding

    def test_no_output_projection_means_tied(self):
        head = dense_header()
        kept = tuple(t for t in head.tensors if not t.name.startswith("output.weight"))
        info = describe(header(head.metadata, kept))
        assert info.weights.tied_embedding
        assert info.weights.resident_bytes == info.weights.input_bytes

    def test_an_untied_model_holds_nothing_resident(self):
        assert describe(dense_header()).weights.resident_bytes == 0

    def test_the_output_norm_alone_does_not_count_as_a_projection(self):
        # output_norm is a tiny vector, not the head; its presence must not hide tying.
        head = dense_header()
        kept = tuple(t for t in head.tensors if not t.name.startswith("output.weight"))
        assert any(t.name == "output_norm.weight" for t in kept)
        assert describe(header(head.metadata, kept)).weights.tied_embedding


class TestAttention:
    def test_head_counts_are_expanded_to_one_per_block(self):
        attention = describe(dense_header()).attention
        assert attention.head_count_kv == (2, 2, 2)
        assert attention.gqa_ratio == 4.0

    def test_a_listed_head_count_is_kept_per_block(self):
        head = dense_header(**{"llama.attention.head_count_kv": [2, 4, 8]})
        attention = describe(head).attention
        assert attention.head_count_kv == (2, 4, 8)
        assert attention.uniform_head_count_kv is None

    def test_head_dimension_falls_back_to_embedding_over_heads(self):
        attention = describe(dense_header()).attention
        assert attention.key_length == 8
        assert attention.value_length == 8

    def test_explicit_key_length_wins(self):
        head = dense_header(**{"llama.attention.key_length": 128})
        assert describe(head).attention.key_length == 128

    def test_latent_attention_is_flagged(self):
        head = dense_header(**{"llama.attention.key_length_mla": 512})
        assert any("latent vector" in n for n in describe(head).notes)


class TestDescription:
    def test_vocab_size_comes_from_the_token_array_shape(self):
        head = dense_header(**{"tokenizer.ggml.tokens": ArraySummary(8, 32000)})
        assert describe(head).vocab_size == 32000

    def test_file_type_is_named(self):
        assert describe(dense_header()).file_type == "Q4_K_M"

    def test_unknown_file_type_stays_none(self):
        assert describe(dense_header(**{"general.file_type": 999})).file_type is None

    def test_quant_mix_is_ordered_by_size(self):
        mix = describe(dense_header()).quant_mix
        assert [share.quant for share in mix] == ["Q4_K", "F32"]
        assert mix[0].tensor_count == 5

    def test_parameter_label_rounds_to_the_usual_suffix(self):
        info = describe(dense_header())
        assert info.parameter_label == "1.1K"


class TestNameResolution:
    """Names are resolved out of the stores on this machine, not from any one product."""

    def manifest_store(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
        monkeypatch.delenv("SETPOINT_MODELS_DIR", raising=False)
        blob = tmp_path / "blobs" / "sha256-abc"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"gguf")
        manifest = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "demo" / "7b"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            '{"layers": ['
            '{"mediaType": "application/vnd.ollama.image.params", "digest": "sha256:zzz"},'
            '{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:abc"}]}',
            encoding="utf-8",
        )
        return blob

    def test_a_manifest_resolves_to_the_file_it_points_at(self, tmp_path, monkeypatch):
        blob = self.manifest_store(tmp_path, monkeypatch)
        assert resolve("demo:7b").path == blob

    def test_an_unknown_name_is_an_error_rather_than_a_guess(self, tmp_path, monkeypatch):
        self.manifest_store(tmp_path, monkeypatch)
        with pytest.raises(ModelError, match="no model file found"):
            resolve("nothing-here")

    def test_a_path_like_reference_is_not_a_name(self, tmp_path, monkeypatch):
        self.manifest_store(tmp_path, monkeypatch)
        with pytest.raises(ModelError):
            resolve("a/b/c/d:1")

    def test_a_file_path_wins_over_any_store(self, tmp_path, monkeypatch):
        self.manifest_store(tmp_path, monkeypatch)
        direct = tmp_path / "direct.gguf"
        direct.write_bytes(b"gguf")
        found = resolve(str(direct))
        assert found.path == direct
        assert found.source == "path"


class TestStores:
    def test_a_configured_directory_is_searched_first(self, tmp_path, monkeypatch):
        # No store is privileged, and the one the user names comes before the rest.
        monkeypatch.setenv("SETPOINT_MODELS_DIR", str(tmp_path))
        assert stores()[0].root == tmp_path
        assert stores()[0].label == "configured"

    def test_several_directories_can_be_configured(self, tmp_path, monkeypatch):
        import os

        one, two = tmp_path / "one", tmp_path / "two"
        monkeypatch.setenv("SETPOINT_MODELS_DIR", os.pathsep.join([str(one), str(two)]))
        roots = [store.root for store in stores()]
        assert roots[:2] == [one, two]

    def test_a_plain_directory_resolves_by_file_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SETPOINT_MODELS_DIR", str(tmp_path))
        (tmp_path / "my-model.gguf").write_bytes(b"gguf")
        assert resolve("my-model").path == tmp_path / "my-model.gguf"
        assert resolve("my-model.gguf").path == tmp_path / "my-model.gguf"

    def test_a_missing_directory_is_listed_rather_than_dropped(self, tmp_path, monkeypatch):
        # `doctor` says where it looked, and an empty list would hide that.
        monkeypatch.setenv("SETPOINT_MODELS_DIR", str(tmp_path / "absent"))
        assert any(not store.root.exists() for store in stores())

    def test_every_store_says_what_layout_it_has(self):
        from setpoint.model.index import FLAT, MANIFEST

        assert all(store.layout in (FLAT, MANIFEST) for store in stores())
