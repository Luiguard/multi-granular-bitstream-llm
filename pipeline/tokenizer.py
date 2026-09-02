import math
from typing import Dict, List, Optional, Tuple

try:
    from .vocabulary import MultiGranularVocabulary, TokenTier
except (ImportError, ValueError):
    from vocabulary import MultiGranularVocabulary, TokenTier


class ViterbiTokenizer:
    """Byte-level Dynamic Programming (Viterbi) Multi-Granular Tokenizer.

    Operates directly on UTF-8 byte sequences with zero OOV possibility and 100%
    lossless round-trip reconstruction guarantee.
    """

    def __init__(self, vocab: MultiGranularVocabulary, length_bonus_factor: float = 0.5):
        self.vocab = vocab
        self.length_bonus_factor = length_bonus_factor

        # Map UTF-8 byte sequence -> (token_id, cost, byte_len)
        self._byte_trie: Dict[bytes, Tuple[int, float, int]] = {}
        self._max_byte_len = 1
        self._build_cost_table()

    def _build_cost_table(self) -> None:
        """Precomputes cost table indexed by exact UTF-8 byte sequences."""
        total_freq = sum(self.vocab.id_to_frequency.values())
        smooth_total = total_freq + self.vocab.size

        # 1. Register Tier 0 raw bytes (0x00 - 0xFF)
        for byte_val in range(256):
            freq = self.vocab.id_to_frequency.get(byte_val, 1)
            prob = freq / smooth_total
            # Base byte cost with fallback penalty so multi-char tokens are strongly preferred
            cost = -math.log2(max(1e-15, prob)) + 6.0
            byte_seq = bytes([byte_val])
            self._byte_trie[byte_seq] = (byte_val, cost, 1)

        # 2. Register Tier 1, 2, 3 tokens
        for token_id in range(256, self.vocab.size):
            token_text = self.vocab.id_to_token.get(token_id, "")
            tier = self.vocab.id_to_tier.get(token_id, TokenTier.WORD)
            token_bytes = token_text.encode("utf-8")
            byte_len = len(token_bytes)
            if byte_len == 0:
                continue

            freq = self.vocab.id_to_frequency.get(token_id, 1)
            prob = freq / smooth_total
            base_cost = -math.log2(max(1e-15, prob))

            if tier == TokenTier.TEMPLATE:
                cost = base_cost - (1.5 * byte_len)
            elif tier == TokenTier.PHRASE:
                cost = base_cost - (1.0 * byte_len)
            elif tier == TokenTier.WORD:
                cost = base_cost - (0.7 * byte_len)
            else:
                cost = base_cost - (0.5 * byte_len)

            self._byte_trie[token_bytes] = (token_id, cost, byte_len)
            if byte_len > self._max_byte_len:
                self._max_byte_len = byte_len

    def encode(self, text: str) -> List[int]:
        """Encodes text into optimal token IDs via Viterbi dynamic programming."""
        if not text:
            return []

        raw_bytes = text.encode("utf-8")
        n = len(raw_bytes)

        # dp[i] = (min_cost, prev_byte_index, token_id)
        dp: List[Tuple[float, int, int]] = [(float("inf"), -1, -1)] * (n + 1)
        dp[0] = (0.0, 0, 0)

        for i in range(n):
            current_cost = dp[i][0]
            if current_cost == float("inf"):
                continue

            # Check all matching sub-byte slices in the vocabulary
            max_lookahead = min(n, i + self._max_byte_len)
            for j in range(i + 1, max_lookahead + 1):
                sub_bytes = raw_bytes[i:j]
                if sub_bytes in self._byte_trie:
                    t_id, t_cost, _ = self._byte_trie[sub_bytes]
                    cand_cost = current_cost + t_cost
                    if cand_cost < dp[j][0]:
                        dp[j] = (cand_cost, i, t_id)

        # Backtrack optimal path
        token_ids: List[int] = []
        curr = n
        while curr > 0:
            cost, prev, t_id = dp[curr]
            if prev == -1:
                curr -= 1
                continue
            token_ids.append(t_id)
            curr = prev

        token_ids.reverse()
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs back into string with exact byte reconstruction."""
        byte_stream = bytearray()
        for t_id in token_ids:
            tier = self.vocab.id_to_tier.get(t_id, TokenTier.WORD)
            if tier == TokenTier.BYTE:
                byte_stream.append(t_id)
            else:
                token_text = self.vocab.id_to_token.get(t_id, "")
                byte_stream.extend(token_text.encode("utf-8"))

        return byte_stream.decode("utf-8", errors="replace")

    def get_token_breakdown(self, text: str) -> List[Tuple[int, str, TokenTier, int]]:
        """Returns (token_id, token_repr, tier, byte_len) for analysis."""
        token_ids = self.encode(text)
        result = []
        for t_id in token_ids:
            tier = self.vocab.id_to_tier.get(t_id, TokenTier.WORD)
            if tier == TokenTier.BYTE:
                token_repr = f"0x{t_id:02X}"
            else:
                token_repr = self.vocab.id_to_token.get(t_id, "")
            byte_len = self.vocab.id_to_byte_len.get(t_id, 1)
            result.append((t_id, token_repr, tier, byte_len))
        return result
