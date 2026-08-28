import json
import math
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class TokenTier(IntEnum):
    BYTE = 0       # Raw byte fallback (0x00 - 0xFF) -> guaranteed zero OOV
    WORD = 1       # Single words / unigrams
    PHRASE = 2     # Multi-word phrases / collocations
    TEMPLATE = 3   # Parameterized sentence templates with slot placeholders


class MultiGranularVocabulary:
    """Hierarchical Multi-Granular Vocabulary for text representation and bitstreams."""

    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.id_to_tier: Dict[int, TokenTier] = {}
        self.id_to_byte_len: Dict[int, int] = {}
        self.id_to_frequency: Dict[int, int] = {}
        self.id_to_pmi: Dict[int, float] = {}

        self._init_byte_fallback()

    def _init_byte_fallback(self) -> None:
        """Initializes Tier 0: 256 exact byte tokens (0x00 to 0xFF)."""
        for byte_val in range(256):
            # Encode byte token as a unique string representation
            byte_char = bytes([byte_val]).decode("latin1")
            token_id = byte_val
            self.token_to_id[byte_char] = token_id
            self.id_to_token[token_id] = byte_char
            self.id_to_tier[token_id] = TokenTier.BYTE
            self.id_to_byte_len[token_id] = 1
            self.id_to_frequency[token_id] = 1
            self.id_to_pmi[token_id] = 0.0

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    @property
    def required_bits(self) -> int:
        """Number of bits required to represent any token in this vocabulary."""
        if self.size <= 1:
            return 1
        return math.ceil(math.log2(self.size))

    def add_token(
        self,
        token_text: str,
        tier: TokenTier,
        frequency: int = 1,
        pmi: float = 0.0,
    ) -> int:
        """Adds a token to the vocabulary or updates its frequency/tier."""
        if token_text in self.token_to_id:
            token_id = self.token_to_id[token_text]
            self.id_to_frequency[token_id] += frequency
            if pmi > self.id_to_pmi.get(token_id, 0.0):
                self.id_to_pmi[token_id] = pmi
            return token_id

        token_id = len(self.id_to_token)
        self.token_to_id[token_text] = token_id
        self.id_to_token[token_id] = token_text
        self.id_to_tier[token_id] = tier
        self.id_to_byte_len[token_id] = len(token_text.encode("utf-8"))
        self.id_to_frequency[token_id] = max(1, frequency)
        self.id_to_pmi[token_id] = pmi
        return token_id

    def get_id(self, token_text: str) -> Optional[int]:
        return self.token_to_id.get(token_text)

    def get_token(self, token_id: int) -> Optional[str]:
        return self.id_to_token.get(token_id)

    def get_tier(self, token_id: int) -> Optional[TokenTier]:
        return self.id_to_tier.get(token_id)

    def get_byte_len(self, token_id: int) -> int:
        return self.id_to_byte_len.get(token_id, 1)

    def get_unigram_probabilities(self) -> Dict[int, float]:
        """Computes empirical probability distribution over all tokens."""
        total_freq = sum(self.id_to_frequency.values())
        return {
            token_id: freq / total_freq
            for token_id, freq in self.id_to_frequency.items()
        }

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "required_bits": self.required_bits,
            "tokens": [
                {
                    "id": token_id,
                    "text": self.id_to_token[token_id],
                    "tier": int(self.id_to_tier[token_id]),
                    "byte_len": self.id_to_byte_len[token_id],
                    "frequency": self.id_to_frequency[token_id],
                    "pmi": self.id_to_pmi[token_id],
                }
                for token_id in range(self.size)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MultiGranularVocabulary":
        vocab = cls()
        vocab.token_to_id.clear()
        vocab.id_to_token.clear()
        vocab.id_to_tier.clear()
        vocab.id_to_byte_len.clear()
        vocab.id_to_frequency.clear()
        vocab.id_to_pmi.clear()

        for item in data["tokens"]:
            token_id = item["id"]
            text = item["text"]
            vocab.token_to_id[text] = token_id
            vocab.id_to_token[token_id] = text
            vocab.id_to_tier[token_id] = TokenTier(item["tier"])
            vocab.id_to_byte_len[token_id] = item["byte_len"]
            vocab.id_to_frequency[token_id] = item["frequency"]
            vocab.id_to_pmi[token_id] = item.get("pmi", 0.0)

        return vocab

    def save_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, filepath: str) -> "MultiGranularVocabulary":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
