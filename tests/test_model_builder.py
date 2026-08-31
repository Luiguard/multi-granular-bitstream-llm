#!/usr/bin/env python3
"""
Unit Test Suite for Bitstream AI Model Builder Engine & Custom Topologies.
"""

import ast
import os
import sys
import unittest

sys.path.insert(0, "/home/benjamin/Bilder")

from pipeline.model_builder_engine import ModelArchitectureSpecs, MODEL_PRESETS


class TestModelBuilder(unittest.TestCase):

    def test_edge_100m_single_expert(self):
        """Tests that a 100M single-expert model computes valid parameter counts and footprint."""
        specs = ModelArchitectureSpecs(
            name="Test-Edge-100M",
            d_model=512,
            n_layers=8,
            n_heads=8,
            num_experts=1,
            top_k=1,
            ffn_multiplier=2.0,
            vocab_size=262144,
            rank_embedding=32,
        )
        params = specs.compute_parameters()
        footprint = specs.compute_hardware_footprint()

        self.assertGreater(params["total_params"], 20_000_000)
        self.assertLess(params["total_params"], 200_000_000)
        self.assertEqual(params["total_params"], params["active_params"], "Single expert is 100% active (Dense)")
        self.assertEqual(params["sparsity_ratio"], 0.0)
        self.assertTrue(footprint["fits_rtx3060_train"])
        self.assertTrue(footprint["fits_rtx3060_infer_full"])

    def test_laptop_7b_moe(self):
        """Tests standard 7.45B MoE specs (12 experts, Top-2)."""
        specs = ModelArchitectureSpecs(
            name="Test-Laptop-7.45B",
            d_model=2048,
            n_layers=24,
            n_heads=16,
            num_experts=12,
            top_k=2,
            ffn_multiplier=2.0,
            vocab_size=262144,
            rank_embedding=64,
        )
        params = specs.compute_parameters()
        footprint = specs.compute_hardware_footprint()

        self.assertGreaterEqual(params["total_params_billion"], 7.0)
        self.assertLessEqual(params["total_params_billion"], 8.0)
        self.assertLessEqual(params["active_params_billion"], 2.0)  # ~1.6B active
        self.assertGreaterEqual(params["sparsity_ratio"], 75.0)
        self.assertTrue(footprint["fits_rtx3060_train"])

    def test_trillion_1_7t_moe(self):
        """Tests 1.73 Trillion Parameter Multi-MoE Model (12x 144B Spliced)."""
        specs = ModelArchitectureSpecs(
            name="Test-Galaxy-1.73T-MoE",
            d_model=12288,
            n_layers=80,
            n_heads=96,
            num_experts=12,
            top_k=2,
            ffn_multiplier=3.5,
            vocab_size=262144,
            rank_embedding=256,
        )
        params = specs.compute_parameters()
        self.assertGreaterEqual(params["total_params_billion"], 1500.0)  # 1.57 Trillion (1.57T)!
        self.assertGreaterEqual(params["sparsity_ratio"], 75.0)

    def test_dense_144b_single_expert(self):
        """Tests a massive 144B Single-Expert Dense model (100% compute active)."""
        specs = ModelArchitectureSpecs(
            name="Test-Dense-Titan-144B",
            d_model=12288,
            n_layers=80,
            n_heads=96,
            num_experts=1,
            top_k=1,
            ffn_multiplier=3.5,
            vocab_size=262144,
            rank_embedding=256,
        )
        params = specs.compute_parameters()
        footprint = specs.compute_hardware_footprint()

        self.assertGreaterEqual(params["total_params_billion"], 120.0)
        self.assertEqual(params["total_params"], params["active_params"], "Dense 144B has 100% active parameters")
        self.assertEqual(params["sparsity_ratio"], 0.0)
        self.assertFalse(footprint["fits_rtx3060_train"], "Dense 144B cannot fit 6GB VRAM for 16-bit backprop")
        self.assertIn("Mega-Frontier", footprint["tier_badge"])

    def test_titan_144b_128_experts_moe(self):
        """Tests 144B MoE with 128 experts (Top-8 active)."""
        specs = ModelArchitectureSpecs(
            name="Test-Titan-144B-MoE",
            d_model=8192,
            n_layers=48,
            n_heads=64,
            num_experts=128,
            top_k=8,
            ffn_multiplier=3.5,
            vocab_size=262144,
            rank_embedding=256,
        )
        params = specs.compute_parameters()
        self.assertGreaterEqual(params["total_params_billion"], 100.0)
        self.assertGreaterEqual(params["sparsity_ratio"], 90.0)

    def test_pytorch_code_generation_syntax(self):
        """Tests that generated PyTorch code is valid Python syntax."""
        specs = ModelArchitectureSpecs(name="GeneratedMoEModel", d_model=1024, n_layers=12, num_experts=8)
        code = specs.generate_pytorch_code()
        # Verify it parses cleanly with ast
        tree = ast.parse(code)
        self.assertIsNotNone(tree)
        self.assertIn("class GeneratedMoEModel(nn.Module):", code)

    def test_model_presets_catalog(self):
        """Tests that all model presets in the catalog compute valid specs."""
        self.assertIn("edge_100m", MODEL_PRESETS)
        self.assertIn("laptop_7b", MODEL_PRESETS)
        self.assertIn("titan_144b_dense", MODEL_PRESETS)
        self.assertIn("titan_144b_moe", MODEL_PRESETS)

        for k, p in MODEL_PRESETS.items():
            specs = ModelArchitectureSpecs(
                name=p["name"],
                d_model=p["d_model"],
                n_layers=p["n_layers"],
                n_heads=p["n_heads"],
                num_experts=p["num_experts"],
                top_k=p["top_k"],
                ffn_multiplier=p["ffn_multiplier"],
                vocab_size=p["vocab_size"],
                rank_embedding=p["rank_embedding"],
            )
            params = specs.compute_parameters()
            self.assertGreater(params["total_params"], 0)


if __name__ == "__main__":
    unittest.main()
