import math
import struct
from dataclasses import dataclass
from typing import BinaryIO, Iterator, List, Tuple, Optional
import numpy as np


@dataclass
class BitstreamHeader:
    magic: bytes          # 4 bytes: b"MGBS"
    version: int          # uint16 (e.g. 1)
    bit_width: int        # uint8 (e.g. 16 bits per token)
    vocab_size: int       # uint32
    token_count: int      # uint64
    raw_byte_count: int   # uint64

    EXT_STRUCT = struct.Struct("<4sHBIIQQ")    # 4s (4), H (2), B (1), I (4), I (4), Q (8), Q (8) = 31 bytes
    HEADER_SIZE = EXT_STRUCT.size

    def serialize(self) -> bytes:
        return self.EXT_STRUCT.pack(
            self.magic,
            self.version,
            self.bit_width,
            self.vocab_size,
            0,  # reserved
            self.token_count,
            self.raw_byte_count,
        )

    @classmethod
    def deserialize(cls, data: bytes) -> "BitstreamHeader":
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(f"Header data too short: {len(data)} < {cls.HEADER_SIZE}")
        magic, version, bit_width, vocab_size, _, token_count, raw_byte_count = cls.EXT_STRUCT.unpack(data[:cls.HEADER_SIZE])
        if magic != b"MGBS":
            raise ValueError(f"Invalid magic bytes in bitstream: {magic}")
        return cls(
            magic=magic,
            version=version,
            bit_width=bit_width,
            vocab_size=vocab_size,
            token_count=token_count,
            raw_byte_count=raw_byte_count,
        )


class BitstreamEncoder:
    """Encodes token ID sequences into packed binary bitstreams."""

    def __init__(self, vocab_size: int, bit_width: Optional[int] = None):
        self.vocab_size = vocab_size
        self.bit_width = bit_width if bit_width is not None else max(1, math.ceil(math.log2(max(2, vocab_size))))

    def pack_tokens(self, token_ids: List[int]) -> bytearray:
        """Packs token IDs with fast native vectorized path for 16-bit and 8-bit."""
        if self.bit_width == 16:
            arr = np.array(token_ids, dtype=np.uint16)
            return bytearray(arr.tobytes())
        elif self.bit_width == 32:
            arr = np.array(token_ids, dtype=np.uint32)
            return bytearray(arr.tobytes())
        elif self.bit_width == 8:
            arr = np.array(token_ids, dtype=np.uint8)
            return bytearray(arr.tobytes())

        # General bit-packing for arbitrary bit widths
        bit_buffer = 0
        bits_in_buffer = 0
        packed_bytes = bytearray()
        bit_mask = (1 << self.bit_width) - 1

        for t_id in token_ids:
            if t_id >= self.vocab_size:
                raise ValueError(f"Token ID {t_id} exceeds vocabulary size {self.vocab_size}")
            bit_buffer = (bit_buffer << self.bit_width) | (t_id & bit_mask)
            bits_in_buffer += self.bit_width

            while bits_in_buffer >= 8:
                bits_in_buffer -= 8
                byte_val = (bit_buffer >> bits_in_buffer) & 0xFF
                packed_bytes.append(byte_val)

        if bits_in_buffer > 0:
            byte_val = (bit_buffer << (8 - bits_in_buffer)) & 0xFF
            packed_bytes.append(byte_val)

        return packed_bytes

    def save_to_file(self, filepath: str, token_ids: List[int], raw_byte_count: int) -> BitstreamHeader:
        """Serializes tokens into a binary bitstream file with header."""
        header = BitstreamHeader(
            magic=b"MGBS",
            version=1,
            bit_width=self.bit_width,
            vocab_size=self.vocab_size,
            token_count=len(token_ids),
            raw_byte_count=raw_byte_count,
        )
        packed_data = self.pack_tokens(token_ids)

        with open(filepath, "wb") as f:
            f.write(header.serialize())
            f.write(packed_data)

        return header


class BitstreamDecoder:
    """Decodes packed binary bitstreams back into token IDs with C-speed NumPy decoding."""

    @staticmethod
    def unpack_tokens(packed_bytes: bytes, token_count: int, bit_width: int) -> List[int]:
        """Unpacks packed bytes into integer token IDs instantly."""
        if bit_width == 16:
            # Fast vectorized C-level decode
            expected_bytes = token_count * 2
            arr = np.frombuffer(packed_bytes[:expected_bytes], dtype=np.uint16)
            return arr.tolist()
        elif bit_width == 32:
            expected_bytes = token_count * 4
            arr = np.frombuffer(packed_bytes[:expected_bytes], dtype=np.uint32)
            return arr.tolist()
        elif bit_width == 8:
            arr = np.frombuffer(packed_bytes[:token_count], dtype=np.uint8)
            return arr.tolist()

        # General bit-unpacking
        bit_buffer = 0
        bits_in_buffer = 0
        token_ids: List[int] = []
        bit_mask = (1 << bit_width) - 1
        byte_iter = iter(packed_bytes)

        while len(token_ids) < token_count:
            while bits_in_buffer < bit_width:
                try:
                    b = next(byte_iter)
                    bit_buffer = (bit_buffer << 8) | b
                    bits_in_buffer += 8
                except StopIteration:
                    break

            if bits_in_buffer < bit_width:
                break

            bits_in_buffer -= bit_width
            token_id = (bit_buffer >> bits_in_buffer) & bit_mask
            token_ids.append(token_id)

        return token_ids

    @classmethod
    def load_from_file(cls, filepath: str) -> Tuple[BitstreamHeader, List[int]]:
        """Reads a bitstream file, parses header, and unpacks tokens."""
        with open(filepath, "rb") as f:
            header_bytes = f.read(BitstreamHeader.HEADER_SIZE)
            header = BitstreamHeader.deserialize(header_bytes)
            packed_bytes = f.read()

        token_ids = cls.unpack_tokens(packed_bytes, header.token_count, header.bit_width)
        return header, token_ids

    @classmethod
    def read_header_only(cls, filepath: str) -> BitstreamHeader:
        """Reads only the header of a bitstream file without unpacking data."""
        with open(filepath, "rb") as f:
            header_bytes = f.read(BitstreamHeader.HEADER_SIZE)
            return BitstreamHeader.deserialize(header_bytes)


class BitstreamDataset:
    """Manages batches of token IDs from bitstreams for training."""

    def __init__(self, token_ids: List[int], sequence_length: int = 64):
        self.token_ids = np.array(token_ids, dtype=np.int64)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        if len(self.token_ids) <= self.sequence_length:
            return 0
        return len(self.token_ids) - self.sequence_length

    def get_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Samples random input and target batches for next-token prediction."""
        max_idx = len(self.token_ids) - self.sequence_length - 1
        if max_idx <= 0:
            x = np.zeros((batch_size, self.sequence_length), dtype=np.int64)
            y = np.zeros((batch_size, self.sequence_length), dtype=np.int64)
            return x, y

        start_indices = np.random.randint(0, max_idx, size=batch_size)
        x = np.stack([self.token_ids[i : i + self.sequence_length] for i in start_indices])
        y = np.stack([self.token_ids[i + 1 : i + self.sequence_length + 1] for i in start_indices])
        return x, y
