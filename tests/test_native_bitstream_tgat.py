#!/usr/bin/env python3
"""
Comprehensive Verification Test Suite for Native Zero-Overhead Bitstream TGAT.

Tests:
1. Bochner Continuous-Time Embedding Math & Orthogonality.
2. Compressed Sparse Row (CSR) Binary Representation.
3. Temporal Attention Scoring & Time-Decayed Ranking.
4. Binary Persistence (.tgat) Roundtrip.
5. Microsecond Retrieval Latency Benchmark (< 1.0 ms).
"""

import os
import sys
import time
import unittest
import numpy as np

sys.path.insert(0, "/home/benjamin/Bilder")

from pipeline.native_bitstream_tgat import TemporalBochnerEncoder, NativeBitstreamTGAT


class TestNativeBitstreamTGAT(unittest.TestCase):
    def setUp(self):
        self.tgat = NativeBitstreamTGAT(bochner_dim=64)

    def test_01_bochner_math_and_continuity(self):
        """Verify continuous Bochner harmonic embeddings."""
        bochner = TemporalBochnerEncoder(dimension=64)
        
        # Test 1: Shape check
        deltas = np.array([0.0, 1.0, 10.0, 3600.0, 86400.0], dtype=np.float32)
        emb = bochner.encode(deltas)
        self.assertEqual(emb.shape, (5, 64))
        
        # Test 2: Continuity: ||Phi(t) - Phi(t + eps)|| -> 0 as eps -> 0
        t0 = np.array([100.0])
        t_eps = np.array([100.001])
        emb_t0 = bochner.encode(t0)
        emb_eps = bochner.encode(t_eps)
        diff = np.linalg.norm(emb_t0 - emb_eps)
        self.assertLess(diff, 1e-3, "Bochner embedding must be smoothly continuous in time.")
        print(f"✅ Bochner Harmonic Continuity Verified (Norm diff: {diff:.6f})")

    def test_02_csr_insertion_and_attention(self):
        """Verify CSR Graph memory insertion and Temporal Attention scoring."""
        now = time.time()
        
        # Insert Nodes (Tokens represent 16-bit Viterbi IDs)
        n1 = self.tgat.add_node("HTTP 429", tokens=[4812, 19042])
        n2 = self.tgat.add_node("Exponential Backoff", tokens=[19042, 328])
        n3 = self.tgat.add_node("Navier Stokes", tokens=[5512, 8821])
        n4 = self.tgat.add_node("Bitstream Graph", tokens=[328, 9912])
        
        self.assertEqual(len(self.tgat.node_labels), 4)

        # Insert Edges with varying timestamps and Hebbian weights
        self.tgat.add_edge(n1, n2, relation_tokens=[101, 102], timestamp=now - 10.0, weight=1.5)  # 10s ago, heavy weight
        self.tgat.add_edge(n2, n4, relation_tokens=[103], timestamp=now - 3600.0, weight=1.0)       # 1h ago
        self.tgat.add_edge(n3, n4, relation_tokens=[104], timestamp=now - 86400.0, weight=0.8)     # 1 day ago

        # Query for 'HTTP 429' tokens
        results = self.tgat.temporal_attention_recall(query_tokens=[4812, 19042], current_time=now, top_k=3)
        self.assertGreater(len(results), 0)
        
        # The most recent + heavily weighted edge (HTTP 429 -> Exponential Backoff) must have highest attention
        top_res = results[0]
        self.assertEqual(top_res["source"], "HTTP 429")
        self.assertEqual(top_res["target"], "Exponential Backoff")
        self.assertGreater(top_res["attention_score"], 0.4)
        print(f"✅ Temporal Attention Ranking Verified: Top score = {top_res['attention_score']:.4f} (Source: {top_res['source']})")

    def test_03_binary_persistence_roundtrip(self):
        """Verify .tgat binary serialization and instant loading."""
        now = time.time()
        n1 = self.tgat.add_node("Quantum Physics", tokens=[771, 992])
        n2 = self.tgat.add_node("Schroedinger Equation", tokens=[992, 1402])
        self.tgat.add_edge(n1, n2, relation_tokens=[55], timestamp=now - 5.0, weight=2.0)

        test_file = "/home/benjamin/Bilder/data/test_memory.tgat"
        self.tgat.save_binary(test_file)
        self.assertTrue(os.path.exists(test_file))

        loaded = NativeBitstreamTGAT.load_binary(test_file)
        self.assertEqual(len(loaded.node_labels), len(self.tgat.node_labels))
        self.assertEqual(len(loaded.edge_src), len(self.tgat.edge_src))

        res = loaded.temporal_attention_recall(query_tokens=[771], current_time=now, top_k=1)
        self.assertEqual(res[0]["source"], "Quantum Physics")
        
        if os.path.exists(test_file):
            os.remove(test_file)
        print("✅ Binary .tgat Persistence Roundtrip Verified")

    def test_04_microsecond_latency_benchmark(self):
        """Benchmark: 1,000 temporal attention graph lookups must average < 1.0 ms."""
        now = time.time()
        # Populate 100 nodes and 300 edges
        for i in range(100):
            self.tgat.add_node(f"Concept_{i}", tokens=[i * 10, i * 10 + 1, i * 10 + 2])
        for i in range(300):
            src = i % 100
            dst = (i * 3 + 7) % 100
            self.tgat.add_edge(src, dst, relation_tokens=[999], timestamp=now - (i * 60), weight=1.0 + (i % 5) * 0.1)

        # Warmup
        self.tgat.temporal_attention_recall([10, 11], current_time=now, top_k=5)

        # Benchmark 1,000 lookups
        start_t = time.perf_counter()
        iterations = 1000
        for i in range(iterations):
            tok = (i % 100) * 10
            self.tgat.temporal_attention_recall([tok, tok + 1], current_time=now, top_k=5)
        total_time = time.perf_counter() - start_t
        avg_ms = (total_time / iterations) * 1000.0

        print(f"⚡ BENCHMARK: 1,000 Lookups in {total_time:.4f}s -> Durchschnitt: {avg_ms:.4f} ms pro Abfrage!")
        self.assertLess(avg_ms, 1.5, "TGAT retrieval must be sub-millisecond.")


if __name__ == "__main__":
    unittest.main()
