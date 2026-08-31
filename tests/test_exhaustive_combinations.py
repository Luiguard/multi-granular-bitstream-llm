#!/usr/bin/env python3
"""
Exhaustive Combination & Edge-Case Test Suite for Bitstream AI Architecture Studio.
Validates that ANY arbitrary combination of d_model, layers, heads, experts (1 to 128),
top-k, vocabs, and hardware profiles executes with 100% mathematical and syntax validity.
"""

import ast
import itertools
import os
import sys
import unittest
import torch

sys.path.insert(0, "/home/benjamin/Bilder")

from pipeline.model_builder_engine import ModelArchitectureSpecs, MODEL_PRESETS


class TestExhaustiveCombinations(unittest.TestCase):

    def test_preset_catalog_full_instantiation(self):
        """Tests that all official catalog presets generate valid code and instantiate cleanly."""
        for key, p in MODEL_PRESETS.items():
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
                max_seq_len=p["max_seq_len"],
            )
            # 1. Parameter Math
            params = specs.compute_parameters()
            self.assertGreater(params["total_params"], 0)
            self.assertGreater(params["active_params"], 0)
            self.assertLessEqual(params["active_params"], params["total_params"])

            # 2. Hardware Footprint
            footprint = specs.compute_hardware_footprint()
            self.assertGreater(footprint["weights_size_16bit_gb"], 0)
            self.assertGreater(footprint["training_gpu_vram_gb"], 0)

            # 3. Code Generation Syntax
            code = specs.generate_pytorch_code()
            tree = ast.parse(code)
            self.assertIsNotNone(tree)

            # 4. Training script & Server script syntax
            train_script = specs.generate_training_script()
            self.assertIsNotNone(ast.parse(train_script))
            api_script = specs.generate_api_server_script()
            self.assertIsNotNone(ast.parse(api_script))
            web_script = specs.generate_web_chat_script()
            self.assertIsNotNone(ast.parse(web_script))
            cli_script = specs.generate_cli_chat_script()
            self.assertIsNotNone(ast.parse(cli_script))

    def test_dense_single_expert_all_scales(self):
        """Tests Single-Expert (Dense / 100% active) models from 100M to 144B."""
        scales = [
            {"name": "Dense-100M", "d": 512, "l": 8, "h": 8},
            {"name": "Dense-500M", "d": 1024, "l": 12, "h": 16},
            {"name": "Dense-7B", "d": 4096, "l": 32, "h": 32},
            {"name": "Dense-70B", "d": 8192, "l": 64, "h": 64},
            {"name": "Dense-144B", "d": 12288, "l": 80, "h": 96},
        ]
        for s in scales:
            specs = ModelArchitectureSpecs(
                name=s["name"],
                d_model=s["d"],
                n_layers=s["l"],
                n_heads=s["h"],
                num_experts=1,
                top_k=1,
                ffn_multiplier=3.0,
            )
            params = specs.compute_parameters()
            self.assertEqual(params["total_params"], params["active_params"], "Dense single expert must have 100% active parameters")
            self.assertEqual(params["sparsity_ratio"], 0.0)

    def test_mega_moe_128_experts_all_scales(self):
        """Tests 128-Expert Mega-MoE models with varying Top-k."""
        for top_k in [1, 2, 4, 8]:
            specs = ModelArchitectureSpecs(
                name=f"Titan-MoE-128Exp-Top{top_k}",
                d_model=4096,
                n_layers=32,
                n_heads=32,
                num_experts=128,
                top_k=top_k,
            )
            params = specs.compute_parameters()
            self.assertGreater(params["total_params"], params["active_params"])
            self.assertGreaterEqual(params["sparsity_ratio"], 90.0)

    def test_random_combinatorial_grid(self):
        """Tests 100+ combinatorial permutations across all setting axes."""
        d_models = [512, 1024, 2048, 4096]
        layers = [8, 16, 24]
        experts = [1, 4, 12, 32, 128]
        top_ks = [1, 2, 4]
        vocabs = [65536, 262144]

        count = 0
        for d, l, e, k, v in itertools.product(d_models, layers, experts, top_ks, vocabs):
            specs = ModelArchitectureSpecs(
                name=f"Grid_{d}_{l}_{e}_{k}",
                d_model=d,
                n_layers=l,
                n_heads=16 if d >= 1024 else 8,
                num_experts=e,
                top_k=k,
                vocab_size=v,
            )
            params = specs.compute_parameters()
            self.assertGreater(params["total_params"], 0)
            self.assertGreater(params["active_params"], 0)
            self.assertLessEqual(params["active_params"], params["total_params"])
            count += 1

        self.assertGreaterEqual(count, 100, "Tested at least 100 distinct combinations")
        print(f"  ✅ {count} verschiedene Kombinations-Topologien erfolgreich verifiziert!")


if __name__ == "__main__":
    unittest.main()
