"""GGUF header reader.

Reads the key/value block and the tensor directory, nothing else. Long metadata
arrays are summarised instead of decoded: a 150k-entry tokenizer costs seconds to
materialise and setpoint never needs the strings. `tests/test_gguf_parity.py` holds
this reader to the reference implementation.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

from .types import ModelError

GGUF_MAGIC = 0x46554747
SUPPORTED_VERSIONS = (2, 3)
DEFAULT_ALIGNMENT = 32

# Arrays longer than this are summarised. Real per-layer parameters (head counts,
# rope sections) are far shorter; anything longer belongs to the tokenizer.
ARRAY_DECODE_LIMIT = 4096

# GGML type id -> (name, elements per block, bytes per block).
QUANT_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18),
}

# GGUF metadata value type id -> (struct format, size in bytes).
_SCALARS: dict[int, tuple[str, int]] = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<?", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}
_TYPE_STRING = 8
_TYPE_ARRAY = 9


class GgufError(ModelError):
    """The file is not a GGUF file setpoint can read."""


@dataclass(frozen=True)
class ArraySummary:
    """An array whose payload was skipped. Only its shape survives."""

    element_type: int
    count: int


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dimensions: tuple[int, ...]
    quant: str
    element_count: int
    byte_count: int
    offset: int


@dataclass(frozen=True)
class GgufHeader:
    path: Path
    version: int
    alignment: int
    metadata: dict[str, object]
    tensors: tuple[TensorEntry, ...]
    data_offset: int
    file_bytes: int


class _Walker:
    """Sequential cursor over the mapped file."""

    def __init__(self, buf: mmap.mmap | bytes) -> None:
        self._buf = buf
        self.pos = 0

    def take(self, fmt: str, size: int) -> tuple[object, ...]:
        try:
            values = struct.unpack_from(fmt, self._buf, self.pos)
        except struct.error as exc:
            raise GgufError(f"header ends unexpectedly at byte {self.pos}") from exc
        self.pos += size
        return values

    def uint32(self) -> int:
        return int(self.take("<I", 4)[0])

    def uint64(self) -> int:
        return int(self.take("<Q", 8)[0])

    def string(self) -> str:
        length = self.uint64()
        raw = self._buf[self.pos : self.pos + length]
        if len(raw) != length:
            raise GgufError(f"string of {length} bytes runs past the end of the file")
        self.pos += length
        return bytes(raw).decode("utf-8", "replace")

    def skip_strings(self, count: int) -> None:
        for _ in range(count):
            self.pos += 8 + self.uint64()

    def value(self, type_id: int) -> object:
        if type_id == _TYPE_STRING:
            return self.string()
        if type_id == _TYPE_ARRAY:
            return self._array()
        fmt, size = _scalar(type_id)
        return self.take(fmt, size)[0]

    def _array(self) -> object:
        element_type = self.uint32()
        count = self.uint64()
        if element_type == _TYPE_STRING:
            self.skip_strings(count)
            return ArraySummary(element_type, count)
        if element_type == _TYPE_ARRAY:
            raise GgufError("nested metadata arrays are not part of the GGUF spec")
        fmt, size = _scalar(element_type)
        if count > ARRAY_DECODE_LIMIT:
            self.pos += count * size
            return ArraySummary(element_type, count)
        return list(self.take(f"<{count}{fmt[1:]}", count * size))


def _scalar(type_id: int) -> tuple[str, int]:
    try:
        return _SCALARS[type_id]
    except KeyError:
        raise GgufError(f"unknown metadata value type {type_id}") from None


def tensor_byte_count(type_id: int, element_count: int) -> tuple[str, int]:
    """Quantization name and stored size for a tensor of `element_count` elements."""
    entry = QUANT_TYPES.get(type_id)
    if entry is None:
        raise GgufError(f"unknown tensor type {type_id}")
    name, block, block_bytes = entry
    return name, element_count * block_bytes // block


def read_header(path: str | Path) -> GgufHeader:
    """Parse the GGUF header. Raises `GgufError` on anything unreadable."""
    file_path = Path(path)
    size = file_path.stat().st_size
    if size < 24:
        raise GgufError(f"{file_path} is too small to be a GGUF file")

    with open(file_path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        walker = _Walker(buf)
        if walker.uint32() != GGUF_MAGIC:
            raise GgufError(f"{file_path} does not start with the GGUF magic number")
        version = walker.uint32()
        if version not in SUPPORTED_VERSIONS:
            raise GgufError(f"GGUF version {version} is not supported")

        tensor_count = walker.uint64()
        kv_count = walker.uint64()

        metadata: dict[str, object] = {}
        for _ in range(kv_count):
            key = walker.string()
            metadata[key] = walker.value(walker.uint32())

        tensors: list[TensorEntry] = []
        for _ in range(tensor_count):
            name = walker.string()
            dims = tuple(int(walker.uint64()) for _ in range(walker.uint32()))
            type_id = walker.uint32()
            offset = walker.uint64()
            elements = 1
            for dim in dims:
                elements *= dim
            quant, byte_count = tensor_byte_count(type_id, elements)
            tensors.append(TensorEntry(name, dims, quant, elements, byte_count, offset))

        alignment = _alignment(metadata)
        padding = walker.pos % alignment
        data_offset = walker.pos + (alignment - padding if padding else 0)

    return GgufHeader(
        path=file_path,
        version=version,
        alignment=alignment,
        metadata=metadata,
        tensors=tuple(tensors),
        data_offset=data_offset,
        file_bytes=size,
    )


def _alignment(metadata: dict[str, object]) -> int:
    value = metadata.get("general.alignment")
    if not isinstance(value, int) or value <= 0 or value & (value - 1):
        return DEFAULT_ALIGNMENT
    return value
