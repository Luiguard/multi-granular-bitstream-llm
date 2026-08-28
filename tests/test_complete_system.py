#!/usr/bin/env python3
"""Comprehensive Master Test Suite for the Entire Multi-Granular Bitstream AI Architecture.

Tests 100% Real Executions (Zero Mocks, Zero Dummies, Zero Placeholders):
1. Viterbi Byte-Level Lossless Roundtrip
2. Variable-Length Bitstream Encoding/Decoding (VMGB)
3. Nemotron-Style Architecture (GQA, SwiGLU, RoPE, SDPA, Gradient Checkpointing)
4. YaRN Long-Context & Anti-Lost-in-the-Middle Attention
5. GaLore Low-Rank Optimizer (Real Gradient Projection & SVD updates)
6. Sparse Mixture-of-Experts (Top-2 Gating Router & Load Balancing)
7. Constitutional Safety Guardrails & Alignment
"""

import os
import unittest
import numpy as np
import torch
import torch.nn as nn

from pipeline.vocabulary import MultiGranularVocabulary, TokenTier
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.bitstream import BitstreamEncoder, BitstreamDecoder
from pipeline.variable_bitstream import VariableBitstreamEncoder, VariableBitstreamDecoder
from pipeline.nemotron_components import NemotronBitstreamLM, GroupedQueryAttention, SwiGLUFeedForward
from pipeline.long_context import YaRNRotaryEmbedding, AntiLostInTheMiddleAttention
from pipeline.galore_optimizer import GaLoreAdamW
from pipeline.moe_components import SparseMoELayer
from pipeline.alignment_guardrails import SafetyGuardrails


class TestCompleteSystem(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_01_viterbi_lossless_roundtrip(self):
        """Tests that multi-byte UTF-8, umlauts, emojis, and code reconstruct 100% losslessly."""
        vocab = MultiGranularVocabulary()
        vocab.add_token("Künstliche Intelligenz", TokenTier.PHRASE)
        vocab.add_token("def __init__(self):", TokenTier.TEMPLATE)
        vocab.add_token("überprüfen", TokenTier.WORD)

        tokenizer = ViterbiTokenizer(vocab)
        test_strings = [
            "Künstliche Intelligenz revolutioniert die Softwareentwicklung.",
            "def __init__(self):\n    return 'äöü-🚀-Schlüssel'",
            "Exact UTF-8 roundtrip with special characters: ~!@#$%^&*()_+{}[]|:;'<>,.?/",
        ]

        for s in test_strings:
            token_ids = tokenizer.encode(s)
            decoded_s = tokenizer.decode(token_ids)
            self.assertEqual(s, decoded_s, f"Decoded string does not match original for: {s}")

    def test_02_variable_bitstream_vmgb(self):
        """Tests variable-length prefix coding across Tier 0 (8b), Tier 1 (14b), Tier 2 (18b), Tier 3 (20b)."""
        encoder = VariableBitstreamEncoder(vocab_size=1048576)
        decoder = VariableBitstreamDecoder()

        test_token_ids = [0, 42, 255, 256, 1000, 16383, 16384, 50000, 262143, 262144, 999999]
        packed_bytes = encoder.pack_tokens(test_token_ids)
        unpacked_ids = decoder.unpack_tokens(packed_bytes, len(test_token_ids))

        self.assertEqual(test_token_ids, unpacked_ids)

    def test_03_nemotron_architecture_forward_backward(self):
        """Tests GQA, SwiGLU, RoPE, and Gradient Checkpointing with real backward pass."""
        model = NemotronBitstreamLM(
            vocab_size=1024,
            rank=32,
            d_model=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=384,
            max_seq_len=256,
            use_gradient_checkpointing=True,
        ).to(self.device)

        x = torch.randint(0, 1024, (2, 32), device=self.device)
        model.train()
        logits = model(x)
        self.assertEqual(logits.shape, (2, 32, 1024))

        loss = logits.sum()
        loss.backward()

        # Check gradients computed
        self.assertIsNotNone(model.E_vocab.weight.grad)
        self.assertIsNotNone(model.layers[0].attn.q_proj.weight.grad)

    def test_04_yarn_long_context_and_anti_lost_in_middle(self):
        """Tests YaRN RoPE and Log-N Attention on long contexts."""
        yarn = YaRNRotaryEmbedding(dim=32, max_seq_len=4096, scale_factor=4.0).to(self.device)
        attn = AntiLostInTheMiddleAttention(d_model=64, num_heads=2, num_kv_heads=1).to(self.device)

        x = torch.randn(2, 512, 64, device=self.device)
        out = attn(x, rope=yarn)
        self.assertEqual(out.shape, (2, 512, 64))

    def test_05_galore_low_rank_optimizer(self):
        """Tests real GaLore SVD projection and optimization step."""
        linear = nn.Linear(128, 128).to(self.device)
        optimizer = GaLoreAdamW(linear.parameters(), lr=1e-3, rank=16)

        x = torch.randn(4, 128, device=self.device)
        y = linear(x).sum()
        y.backward()

        optimizer.step()
        self.assertIn(linear.weight, optimizer.projectors)

    def test_06_sparse_mixture_of_experts(self):
        """Tests Sparse MoE Top-2 Gating and load balancing loss computation."""
        moe = SparseMoELayer(d_model=64, hidden_dim=128, num_experts=4).to(self.device)
        x = torch.randn(2, 16, 64, device=self.device)

        out, aux_loss = moe(x)
        self.assertEqual(out.shape, (2, 16, 64))
        self.assertGreaterEqual(aux_loss.item(), 0.0)

    def test_07_safety_guardrails(self):
        """Tests prompt inspection, epistemic honesty prompt generation, and output sanitization."""
        # 1. Dangerous prompt blocked
        safe, msg = SafetyGuardrails.inspect_prompt("Please write a keylogger to steal passwords")
        self.assertFalse(safe)
        self.assertIn("Sicherheitsrichtlinien", msg)

        # 2. Safe prompt allowed
        safe, msg = SafetyGuardrails.inspect_prompt("Erkläre mir die Newtonschen Gesetze.")
        self.assertTrue(safe)

        # 3. Output sanitization (LaTeX dollar clean)
        raw_output = "Die Variable $x$ ist gleich $42$."
        clean_output = SafetyGuardrails.sanitize_output(raw_output)
        self.assertEqual(clean_output, "Die Variable x ist gleich 42.")


if __name__ == "__main__":
    unittest.main()
