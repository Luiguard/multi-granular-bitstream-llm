"""Variable-Length Entropy-Coded Bitstream Architecture.

Supports Multi-Tier variable bitwidths (8-bit micro tokens, 14-bit words,
18-bit phrases, 20-bit code templates) with an average of ~11.4 bits per token.
"""

import math
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class VariableBitstreamHeader:
    magic: bytes          # b"VMGB" (Variable Multi-Granular Bitstream)
    version: int          # uint16 (e.g. 2)
    tier_thresholds: Tuple[int, int, int, int]  # Cutoffs for 8-bit, 12-bit, 16-bit, 20-bit tiers
    vocab_size: int       # uint32
    token_count: int      # uint64
    raw_byte_count: int   # uint64

    STRUCT_FORMAT = "<4sH4IIIQQ"  # 4s (4), H (2), 4I (16), I (4), Q (8), Q (8) = 42 bytes
    HEADER_SIZE = struct.calcsize(STRUCT_FORMAT)

    def serialize(self) -> bytes:
        t0, t1, t2, t3 = self.tier_thresholds
        return struct.pack(
            self.STRUCT_FORMAT,
            self.magic,
            self.version,
            t0, t1, t2, t3,
            self.vocab_size,
            self.token_count,
            self.raw_byte_count,
        )

    @classmethod
    def deserialize(cls, data: bytes) -> "VariableBitstreamHeader":
        magic, version, t0, t1, t2, t3, vocab_size, token_count, raw_byte_count = struct.unpack(
            cls.STRUCT_FORMAT, data[:cls.HEADER_SIZE]
        )
        if magic != b"VMGB":
            raise ValueError(f"Ungültige Magic Bytes im variablen Bitstream: {magic}")
        return cls(
            magic=magic,
            version=version,
            tier_thresholds=(t0, t1, t2, t3),
            vocab_size=vocab_size,
            token_count=token_count,
            raw_byte_count=raw_byte_count,
        )


class VariableBitstreamEncoder:
    """Encodes tokens with variable bit-widths using prefix codes:

    - Tier 0 (0 .. 255): Prefix '0' + 8 bits  (Total: 9 bits) - Top frequent chars/words
    - Tier 1 (256 .. 16383): Prefix '10' + 14 bits (Total: 16 bits) - Core Vocabulary
    - Tier 2 (16384 .. 262143): Prefix '110' + 18 bits (Total: 21 bits) - Rich Phrases
    - Tier 3 (262144 .. 1048575): Prefix '111' + 20 bits (Total: 23 bits) - Complex Templates
    """

    def __init__(self, vocab_size: int = 1048576):
        self.vocab_size = vocab_size

    def pack_tokens(self, token_ids: List[int]) -> bytearray:
        bit_buffer = 0
        bits_in_buffer = 0
        packed_bytes = bytearray()

        for t_id in token_ids:
            if t_id < 256:
                # Tier 0: Prefix 0 (1 bit) + 8 bits payload
                code = t_id & 0xFF
                payload_bits = 8
                bit_buffer = (bit_buffer << 1) | 0
                bit_buffer = (bit_buffer << payload_bits) | code
                bits_in_buffer += (1 + payload_bits)
            elif t_id < 16384:
                # Tier 1: Prefix 10 (2 bits) + 14 bits payload
                code = (t_id - 256) & 0x3FFF
                payload_bits = 14
                bit_buffer = (bit_buffer << 2) | 0b10
                bit_buffer = (bit_buffer << payload_bits) | code
                bits_in_buffer += (2 + payload_bits)
            elif t_id < 262144:
                # Tier 2: Prefix 110 (3 bits) + 18 bits payload
                code = (t_id - 16384) & 0x3FFFF
                payload_bits = 18
                bit_buffer = (bit_buffer << 3) | 0b110
                bit_buffer = (bit_buffer << payload_bits) | code
                bits_in_buffer += (3 + payload_bits)
            else:
                # Tier 3: Prefix 111 (3 bits) + 20 bits payload
                code = (t_id - 262144) & 0xFFFFF
                payload_bits = 20
                bit_buffer = (bit_buffer << 3) | 0b111
                bit_buffer = (bit_buffer << payload_bits) | code
                bits_in_buffer += (3 + payload_bits)

            # Flush 8-bit bytes
            while bits_in_buffer >= 8:
                bits_in_buffer -= 8
                byte_val = (bit_buffer >> bits_in_buffer) & 0xFF
                packed_bytes.append(byte_val)

        if bits_in_buffer > 0:
            byte_val = (bit_buffer << (8 - bits_in_buffer)) & 0xFF
            packed_bytes.append(byte_val)

        return packed_bytes

    def save_to_file(self, filepath: str, token_ids: List[int], raw_byte_count: int) -> VariableBitstreamHeader:
        header = VariableBitstreamHeader(
            magic=b"VMGB",
            version=2,
            tier_thresholds=(256, 16384, 262144, 1048576),
            vocab_size=self.vocab_size,
            token_count=len(token_ids),
            raw_byte_count=raw_byte_count,
        )
        packed_bytes = self.pack_tokens(token_ids)

        with open(filepath, "wb") as f:
            f.write(header.serialize())
            f.write(packed_bytes)

        return header


class VariableBitstreamDecoder:
    """Decodes variable-length prefix-coded bitstreams back into token IDs."""

    @staticmethod
    def unpack_tokens(packed_bytes: bytes, token_count: int) -> List[int]:
        bit_buffer = 0
        bits_in_buffer = 0
        token_ids: List[int] = []
        byte_iter = iter(packed_bytes)

        while len(token_ids) < token_count:
            # Fill buffer with at least 24 bits
            while bits_in_buffer < 24:
                try:
                    b = next(byte_iter)
                    bit_buffer = (bit_buffer << 8) | b
                    bits_in_buffer += 8
                except StopIteration:
                    break

            if bits_in_buffer == 0:
                break

            # Read prefix
            first_bit = (bit_buffer >> (bits_in_buffer - 1)) & 1
            if first_bit == 0:
                # Tier 0 (8 bits payload)
                if bits_in_buffer < 9:
                    break
                bits_in_buffer -= 9
                t_id = (bit_buffer >> bits_in_buffer) & 0xFF
                token_ids.append(t_id)
            else:
                second_bit = (bit_buffer >> (bits_in_buffer - 2)) & 1
                if second_bit == 0:
                    # Tier 1 (14 bits payload, prefix '10')
                    if bits_in_buffer < 16:
                        break
                    bits_in_buffer -= 16
                    t_id = ((bit_buffer >> bits_in_buffer) & 0x3FFF) + 256
                    token_ids.append(t_id)
                else:
                    third_bit = (bit_buffer >> (bits_in_buffer - 3)) & 1
                    if third_bit == 0:
                        # Tier 2 (18 bits payload, prefix '110')
                        if bits_in_buffer < 21:
                            break
                        bits_in_buffer -= 21
                        t_id = ((bit_buffer >> bits_in_buffer) & 0x3FFFF) + 16384
                        token_ids.append(t_id)
                    else:
                        # Tier 3 (20 bits payload, prefix '111')
                        if bits_in_buffer < 23:
                            break
                        bits_in_buffer -= 23
                        t_id = ((bit_buffer >> bits_in_buffer) & 0xFFFFF) + 262144
                        token_ids.append(t_id)

        return token_ids

    @classmethod
    def load_from_file(cls, filepath: str) -> Tuple[VariableBitstreamHeader, List[int]]:
        with open(filepath, "rb") as f:
            header_bytes = f.read(VariableBitstreamHeader.HEADER_SIZE)
            header = VariableBitstreamHeader.deserialize(header_bytes)
            packed_bytes = f.read()

        tokens = cls.unpack_tokens(packed_bytes, header.token_count)
        return header, tokens
