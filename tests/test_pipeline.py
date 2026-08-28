import os
import unittest
import numpy as np
from pipeline.vocabulary import MultiGranularVocabulary, TokenTier
from pipeline.mining import PhraseMiner
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.bitstream import BitstreamEncoder, BitstreamDecoder, BitstreamDataset
from pipeline.model_components import FactorizedEmbedding, ByteWeightedCrossEntropyLoss, MiniTransformerBlock


class TestMultiGranularPipeline(unittest.TestCase):

    def setUp(self):
        # Real German text corpus with recurring phrases, idioms, and sentence patterns
        self.raw_corpus = [
            "Ich bin ein Mensch und ich denke nach.",
            "Ich bin ein Mensch, der die Welt verstehen will.",
            "Auf der Grundlage dieser Daten können wir das Problem lösen.",
            "Auf der Grundlage neuer Erkenntnisse wird die Hypothese bestätigt.",
            "In der Regel führt dieser Ansatz zu besseren Ergebnissen.",
            "In der Regel ist die Komplexität quadratisch bezüglich der Sequenzlänge.",
            "Das Modell lernt sowohl Wörter als auch Phrasen und Satzmuster.",
            "Künstliche Intelligenz und maschinelles Lernen verändern die Softwarearchitektur.",
            "Ein Bitstream speichert Daten ohne unnötigen Overhead.",
            "Ich bin ein Forscher auf dem Gebiet der künstlichen Intelligenz.",
        ]

    def test_01_vocabulary_byte_fallback(self):
        vocab = MultiGranularVocabulary()
        # Ensure Tier 0 has exactly 256 bytes
        self.assertEqual(vocab.size, 256)
        self.assertEqual(vocab.required_bits, 8)
        # Check byte tokens
        for b in range(256):
            b_char = bytes([b]).decode("latin1")
            self.assertIn(b_char, vocab.token_to_id)
            self.assertEqual(vocab.id_to_tier[b], TokenTier.BYTE)

    def test_02_phrase_miner_and_pmi(self):
        miner = PhraseMiner(min_ngram_freq=2, min_pmi=0.1, max_ngram_len=4, max_vocab_budget=1024)
        vocab, stats = miner.mine_from_corpus(self.raw_corpus)

        self.assertGreater(stats.total_words, 50)
        self.assertGreater(stats.unique_unigrams, 20)
        self.assertGreater(stats.mined_phrases_count, 0)
        self.assertGreater(vocab.size, 256)

        # Verify that frequent collocations like "Ich bin ein" or "Auf der" or "In der" were mined into Tier 2/3
        phrases = [vocab.get_token(i) for i in range(256, vocab.size)]
        self.assertTrue(any("Ich bin" in p or "Auf der" in p or "In der" in p for p in phrases))

    def test_03_viterbi_tokenizer_roundtrip_lossless(self):
        miner = PhraseMiner(min_ngram_freq=2, min_pmi=0.1, max_ngram_len=4, max_vocab_budget=1024)
        vocab, stats = miner.mine_from_corpus(self.raw_corpus)
        tokenizer = ViterbiTokenizer(vocab)

        test_sentences = [
            "Ich bin ein Mensch",
            "Auf der Grundlage dieser Daten",
            "In der Regel",
            "Ein völlig unbekannter OOV Text mit Sonderzeichen: @#$% & äöüß!",
            "1234567890",
            "   Viele   Leerzeichen   und   Tabs \t \n",
        ]

        for s in test_sentences:
            token_ids = tokenizer.encode(s)
            reconstructed = tokenizer.decode(token_ids)
            self.assertEqual(
                reconstructed, s,
                f"Lossless reconstruction failed for '{s}' != '{reconstructed}'"
            )

    def test_04_bitstream_packing_exactness(self):
        vocab_size = 512
        encoder = BitstreamEncoder(vocab_size=vocab_size, bit_width=9)  # 9 bits per token
        test_ids = [0, 255, 511, 42, 128, 300, 1, 99, 450, 12]

        packed = encoder.pack_tokens(test_ids)
        # 10 tokens * 9 bits = 90 bits -> ceil(90 / 8) = 12 bytes
        self.assertEqual(len(packed), 12)

        unpacked = BitstreamDecoder.unpack_tokens(bytes(packed), token_count=len(test_ids), bit_width=9)
        self.assertEqual(test_ids, unpacked)

    def test_05_bitstream_file_roundtrip(self):
        tmp_file = "/tmp/test_bitstream.mgbs"
        encoder = BitstreamEncoder(vocab_size=1024, bit_width=10)
        tokens = [10, 20, 30, 40, 50, 100, 200, 500, 1000]
        raw_byte_count = 150

        header = encoder.save_to_file(tmp_file, tokens, raw_byte_count)
        self.assertEqual(header.token_count, len(tokens))

        loaded_header, loaded_tokens = BitstreamDecoder.load_from_file(tmp_file)
        self.assertEqual(loaded_header.magic, b"MGBS")
        self.assertEqual(loaded_header.version, 1)
        self.assertEqual(loaded_header.bit_width, 10)
        self.assertEqual(loaded_header.token_count, len(tokens))
        self.assertEqual(loaded_header.raw_byte_count, raw_byte_count)
        self.assertEqual(loaded_tokens, tokens)

        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    def test_06_factorized_embedding_and_loss(self):
        vocab_size = 500
        rank = 16
        embedding_dim = 64
        batch_size = 2
        seq_len = 8

        embedding = FactorizedEmbedding(vocab_size, rank, embedding_dim)
        # Check parameter reduction
        self.assertLess(embedding.parameter_count, embedding.standard_parameter_count)

        # Real token test batch
        token_batch = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        embedded = embedding.forward(token_batch)
        self.assertEqual(embedded.shape, (batch_size, seq_len, embedding_dim))

        # Transformer step
        transformer = MiniTransformerBlock(embedding_dim, hidden_dim=128)
        transformed = transformer.forward(embedded)
        self.assertEqual(transformed.shape, (batch_size, seq_len, embedding_dim))

        # Output logits
        W_out = np.random.normal(0, 0.02, (embedding_dim, vocab_size)).astype(np.float32)
        logits = np.matmul(transformed, W_out)

        # Loss computation
        id_to_bytes = {i: np.random.randint(1, 10) for i in range(vocab_size)}
        criterion = ByteWeightedCrossEntropyLoss(id_to_bytes, vocab_size)
        targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

        loss, grad_logits = criterion.compute(logits, targets)
        self.assertGreater(loss, 0.0)
        self.assertEqual(grad_logits.shape, (batch_size, seq_len, vocab_size))

        # Backprop test
        grad_embed = np.matmul(grad_logits, W_out.T)
        embedding.backward(grad_embed)
        self.assertFalse(np.all(embedding.grad_E_proj == 0.0))


if __name__ == "__main__":
    unittest.main()
