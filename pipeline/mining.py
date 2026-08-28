import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from .vocabulary import MultiGranularVocabulary, TokenTier


@dataclass
class MiningStats:
    total_raw_characters: int
    total_words: int
    unique_unigrams: int
    mined_phrases_count: int
    mined_templates_count: int
    final_vocab_size: int
    required_bits: int


class PhraseMiner:
    """Extracts high-information phrases, collocations, and syntactic sentence patterns from text."""

    def __init__(
        self,
        min_ngram_freq: int = 2,
        min_pmi: float = 0.4,
        max_ngram_len: int = 5,
        max_vocab_budget: int = 65536,
        word_budget_ratio: float = 0.40,
        phrase_budget_ratio: float = 0.50,
        template_budget_ratio: float = 0.10,
    ):
        self.min_ngram_freq = min_ngram_freq
        self.min_pmi = min_pmi
        self.max_ngram_len = max_ngram_len
        self.max_vocab_budget = max_vocab_budget
        self.word_budget_ratio = word_budget_ratio
        self.phrase_budget_ratio = phrase_budget_ratio
        self.template_budget_ratio = template_budget_ratio

    def _tokenize_to_words(self, text: str) -> List[str]:
        """Splits text into words and punctuation tokens while preserving whitespace delimiters."""
        tokens = re.findall(r"\w+|[^\w\s]|\s+", text, re.UNICODE)
        return tokens

    def mine_from_corpus(self, text_corpus: List[str]) -> Tuple[MultiGranularVocabulary, MiningStats]:
        """Performs statistical n-gram mining, PMI calculation, and balanced multi-tier induction."""
        vocab = MultiGranularVocabulary()

        # Step 1: Tokenize corpus into words
        all_token_sequences: List[List[str]] = []
        total_raw_chars = 0
        total_words = 0

        unigram_counts: Counter = Counter()
        ngram_counts: Dict[int, Counter] = {n: Counter() for n in range(2, self.max_ngram_len + 1)}

        for doc in text_corpus:
            total_raw_chars += len(doc)
            words = self._tokenize_to_words(doc)
            all_token_sequences.append(words)

            for w in words:
                unigram_counts[w] += 1
                total_words += 1

            # Count n-grams
            for n in range(2, self.max_ngram_len + 1):
                for i in range(len(words) - n + 1):
                    ngram = tuple(words[i : i + n])
                    ngram_counts[n][ngram] += 1

        total_unigrams = max(1, sum(unigram_counts.values()))
        usable_budget = max(100, self.max_vocab_budget - 256)

        max_word_slots = int(usable_budget * self.word_budget_ratio)
        max_phrase_slots = int(usable_budget * self.phrase_budget_ratio)
        max_template_slots = int(usable_budget * self.template_budget_ratio)

        # Step 2: Add top unigram words (Tier 1)
        added_words = 0
        for word, count in unigram_counts.most_common():
            if added_words >= max_word_slots or vocab.size >= self.max_vocab_budget:
                break
            vocab.add_token(word, tier=TokenTier.WORD, frequency=count, pmi=0.0)
            added_words += 1

        # Step 3: Compute PMI and Information Gain for N-Grams (Tier 2)
        candidate_phrases: List[Tuple[str, float, int, float]] = []

        for n in range(2, self.max_ngram_len + 1):
            total_ngrams_n = max(1, sum(ngram_counts[n].values()))
            for ngram, freq in ngram_counts[n].items():
                if freq < self.min_ngram_freq:
                    continue

                p_joint = freq / total_ngrams_n
                prod_p_marginal = 1.0
                for w in ngram:
                    p_w = max(1e-12, unigram_counts[w] / total_unigrams)
                    prod_p_marginal *= p_w

                pmi = math.log2(p_joint / prod_p_marginal)
                if pmi >= self.min_pmi:
                    phrase_str = "".join(ngram)
                    byte_len = len(phrase_str.encode("utf-8"))
                    info_gain = freq * max(1, byte_len - 1) * max(0.1, pmi)
                    candidate_phrases.append((phrase_str, info_gain, freq, pmi))

        # Sort candidate phrases by Information Gain
        candidate_phrases.sort(key=lambda x: x[1], reverse=True)

        added_phrases = 0
        for phrase_str, info_gain, freq, pmi in candidate_phrases:
            if added_phrases >= max_phrase_slots or vocab.size >= self.max_vocab_budget:
                break
            vocab.add_token(phrase_str, tier=TokenTier.PHRASE, frequency=freq, pmi=pmi)
            added_phrases += 1

        # Step 4: Extract Frequent Syntactic Sentence Templates (Tier 3)
        template_candidates: Counter = Counter()
        for doc in text_corpus:
            sentences = re.split(r"[.!?\n]+", doc)
            for s in sentences:
                s_strip = s.strip()
                if not s_strip:
                    continue
                words = self._tokenize_to_words(s_strip)
                if len(words) >= 3:
                    prefix_pattern = "".join(words[:2]) + " {SLOT}"
                    template_candidates[prefix_pattern] += 1
                    if len(words) >= 4:
                        prefix_3 = "".join(words[:3]) + " {SLOT}"
                        template_candidates[prefix_3] += 1

        added_templates = 0
        for template_str, freq in template_candidates.most_common():
            if freq >= self.min_ngram_freq:
                if added_templates >= max_template_slots or vocab.size >= self.max_vocab_budget:
                    break
                vocab.add_token(template_str, tier=TokenTier.TEMPLATE, frequency=freq, pmi=1.0)
                added_templates += 1

        # Fill remaining budget with remaining top words or candidate phrases
        if vocab.size < self.max_vocab_budget:
            for word, count in unigram_counts.most_common():
                if vocab.size >= self.max_vocab_budget:
                    break
                vocab.add_token(word, tier=TokenTier.WORD, frequency=count, pmi=0.0)

        stats = MiningStats(
            total_raw_characters=total_raw_chars,
            total_words=total_words,
            unique_unigrams=len(unigram_counts),
            mined_phrases_count=added_phrases,
            mined_templates_count=added_templates,
            final_vocab_size=vocab.size,
            required_bits=vocab.required_bits,
        )

        return vocab, stats
