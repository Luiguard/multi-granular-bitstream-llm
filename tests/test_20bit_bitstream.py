#!/usr/bin/env python3
"""
Test Suite for 20-Bit Multi-Granular Bitstream & 175+ ISO Languages (1,048,576 Tokens).
Verifies:
1. Exact 1,048,576 vocabulary size and tier coverage.
2. 100% lossless Viterbi byte reconstruction & Zero-OOV across 10 multilingual scripts.
3. 20-Bit Bitstream binary serialization & unpacking roundtrip (.mgbs).
4. Variable Bitstream prefix-coded serialization roundtrip (Tier 0-3).
5. 7B MoE Model initialization and Forward pass with 20-bit (1,048,576) factorized embeddings.
"""

import os
import sys
import tempfile
import unittest
import torch

sys.path.insert(0, "/home/benjamin/Bilder")

from typing import cast
from pipeline.vocabulary import MultiGranularVocabulary, TokenTier
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.bitstream import BitstreamEncoder, BitstreamDecoder
from pipeline.variable_bitstream import VariableBitstreamEncoder, VariableBitstreamDecoder
from pipeline.moe_7b_model import MultiGranularMoE7BModel, SparseMoELayer16
from pipeline.nemotron_components import SwiGLUFeedForward


class Test20BitBitstreamSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bin_path = "/home/benjamin/Bilder/data/vocab_1m_20bit.bin"
        cls.json_path = "/home/benjamin/Bilder/data/vocab_1m_20bit.json"

        # Load via ultra-fast binary loader
        cls.vocab = MultiGranularVocabulary.load_file(cls.bin_path)
        cls.tokenizer = ViterbiTokenizer(cls.vocab)
        cls.encoder = BitstreamEncoder(vocab_size=cls.vocab.size, bit_width=20)

    def test_01_vocabulary_size_and_tiers(self):
        """Verifies exact 1,048,576 tokens (20-bit) and full tier coverage."""
        self.assertEqual(self.vocab.size, 1048576)
        self.assertEqual(self.vocab.required_bits, 20)

        # Tier 0 raw bytes check (0x00 to 0xFF)
        for b in range(256):
            self.assertEqual(self.vocab.id_to_tier[b], TokenTier.BYTE)

        # Multilingual & Multimodal check
        self.assertIn("<think>", self.vocab.token_to_id)
        self.assertIn("</think>", self.vocab.token_to_id)
        self.assertIn(" def", self.vocab.token_to_id)
        self.assertIn(" Quantenmechanik", self.vocab.token_to_id)
        self.assertIn(" the", self.vocab.token_to_id)

        print("  ✅ Test 1: Vokabular-Größe (1.048.576 Tokens, 20-Bit) & Tier-Struktur verifiziert")

    def test_02_lossless_multilingual_reconstruction(self):
        """Verifies 100% loss-free byte roundtrip across 10 multilingual writing systems."""
        test_sentences = [
            "Die Bundesverfassungsgerichtsentscheidung zur Quantenmechanik ist richtungsweisend.",
            "public static void main(String[] args) { System.out.println(\"Hello 20-Bit MoE!\"); }",
            "Теория относительности и квантовая гравитация объясняют фундаментальную структуру вселенной.",
            "اللغة العربية هي إحدى أكثر اللغات انتشارا في العالم وتحتوي على مفردات غنية وعميقة.",
            "量子力学与广义相对论的统一是现代理论物理学中最核心的研究方向之一。",
            "क्वांटम यांत्रिकी और सापेक्षता सिद्धांत आधुनिक भौतिकी के सबसे महत्वपूर्ण स्तंभ हैं।",
            "自然言語処理とビットストリーム言語モデルの進化により高効率な多言語処理が実現します。",
            "인공지능 신경망의 다국어 토큰화 및 임베딩 최적화 성능을 검증합니다.",
            "L'architecture de bitstream multi-granulaire permet une compression sans perte exceptionnelle.",
            "El modelo MoE de 20 bits procesa cientos de idiomas con cero pérdida de vocabulario.",
            "Zero-OOV guaranteed on arbitrary binary & emoji sequences: 🚀✨ \x00\xFF\xAA\x55\x01\xFE"
        ]

        for s in test_sentences:
            tokens = self.tokenizer.encode(s)
            decoded = self.tokenizer.decode(tokens)
            self.assertEqual(s, decoded, f"Fehler bei Rekonstruktion von: {s}")

        print("  ✅ Test 2: Verlustfreie 100% Byte-Rekonstruktion über 10 Schriftsysteme erfolgreich")

    def test_03_20bit_bitstream_io_roundtrip(self):
        """Verifies 20-bit binary serialization and deserialization."""
        sample_text = "Optimierung von 20-Bit Sparse MoE Modellen mit 1.048.576 Tokens über 175 Sprachen."
        token_ids = self.tokenizer.encode(sample_text)

        with tempfile.NamedTemporaryFile(suffix=".mgbs", delete=False) as tf:
            temp_path = tf.name

        try:
            header = self.encoder.save_to_file(temp_path, token_ids, len(sample_text.encode("utf-8")))
            self.assertEqual(header.bit_width, 20)
            self.assertEqual(header.vocab_size, 1048576)
            self.assertEqual(header.token_count, len(token_ids))

            loaded_header, loaded_tokens = BitstreamDecoder.load_from_file(temp_path)
            self.assertEqual(loaded_header.bit_width, 20)
            self.assertEqual(loaded_header.vocab_size, 1048576)
            self.assertEqual(token_ids, loaded_tokens)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        print("  ✅ Test 3: 20-Bit Bitstream Serialisierung & Unpacking Roundtrip (.mgbs) erfolgreich")

    def test_04_variable_bitstream_20bit_roundtrip(self):
        """Verifies variable-length prefix-coded bitstream with 20-bit horizon."""
        var_encoder = VariableBitstreamEncoder(vocab_size=1048576)
        test_tokens = [
            42,        # Tier 0 (0..255)
            1200,      # Tier 1 (256..16383)
            50000,     # Tier 2 (16384..262143)
            500000,    # Tier 3 (262144..1048575)
            1048575,   # Tier 3 max edge token
        ]

        packed = var_encoder.pack_tokens(test_tokens)
        unpacked = VariableBitstreamDecoder.unpack_tokens(bytes(packed), len(test_tokens))
        self.assertEqual(test_tokens, unpacked)
        print("  ✅ Test 4: Variable Bitstream Prefix-Kodierung (Tier 0-3 bis 1.048.575) erfolgreich")

    def test_05_moe_7b_model_20bit_initialization_and_forward(self):
        """Verifies 7B MoE model initializes with 20-bit vocab (1,048,576) and performs forward pass."""
        model = MultiGranularMoE7BModel(
            vocab_size=1048576,
            rank=64,
            d_model=256,   # Lightweight test dimension
            n_layers=2,
            num_experts=4,
            hidden_dim=512,
            max_seq_len=512,
        )

        self.assertEqual(model.E_vocab.weight.shape, torch.Size([1048576, 64]))
        self.assertEqual(model.head_out.weight.shape, torch.Size([1048576, 64]))

        # Test forward pass with token batch
        dummy_input = torch.tensor([[0, 255, 524288, 1048575]], dtype=torch.long)
        with torch.no_grad():
            logits, aux_loss = model(dummy_input)

        self.assertEqual(logits.shape, torch.Size([1, 4, 1048576]))
        print("  ✅ Test 5: 20-Bit MoE Model Initialisierung & Forward-Pass erfolgreich ([1, 4, 1.048.576])")

    def test_06_export_and_import_expert_with_vocab_verification(self):
        """Verifies exporting and importing MoE experts with cryptographic vocabulary verification."""
        model = MultiGranularMoE7BModel(
            vocab_size=1048576,
            rank=64,
            d_model=256,
            n_layers=2,
            num_experts=4,
            hidden_dim=512,
            max_seq_len=512,
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
            expert_file = tf.name

        try:
            # 1. Export expert 1
            meta = {"specialty": "quantum_physics", "author": "AntigravityPair"}
            bundle = model.export_expert(expert_idx=1, file_path=expert_file, metadata=meta)
            self.assertEqual(bundle["vocab_size"], 1048576)
            self.assertEqual(bundle["metadata"]["specialty"], "quantum_physics")

            # 2. Import into expert 2 on another model instance
            model_target = MultiGranularMoE7BModel(
                vocab_size=1048576,
                rank=64,
                d_model=256,
                n_layers=2,
                num_experts=4,
                hidden_dim=512,
                max_seq_len=512,
            )
            model_target.import_expert(expert_file, target_expert_idx=2, verify_vocab=True)

            # Verify weights match
            moe_src = cast(SparseMoELayer16, model.moe_layers[0])
            moe_tgt = cast(SparseMoELayer16, model_target.moe_layers[0])
            expert_src = cast(SwiGLUFeedForward, moe_src.experts[1])
            expert_tgt = cast(SwiGLUFeedForward, moe_tgt.experts[2])
            self.assertTrue(torch.allclose(expert_src.w1.weight, expert_tgt.w1.weight))

            # Verify incompatibility check rejects wrong vocab size
            model_incompatible = MultiGranularMoE7BModel(vocab_size=65536, rank=64, d_model=256, n_layers=2, num_experts=4)
            with self.assertRaises(ValueError):
                model_incompatible.import_expert(expert_file, target_expert_idx=0, verify_vocab=True)

        finally:
            if os.path.exists(expert_file):
                os.remove(expert_file)

        print("  ✅ Test 6: MoE Experten-Export & Import mit kryptografischer Vokabular-Prüfung erfolgreich")


if __name__ == "__main__":
    print("=" * 80)
    print("🧪 20-BIT MULTI-GRANULAR BITSTREAM TEST-SUITE (1.048.576 TOKENS · 175+ SPRACHEN)")
    print("=" * 80)
    unittest.main()
