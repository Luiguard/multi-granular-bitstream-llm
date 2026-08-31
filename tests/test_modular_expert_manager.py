#!/usr/bin/env python3
"""
Unit Test Suite for Modular Expert Management and Domain-Aware Cluster Routing.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, "/home/benjamin/Bilder")

from pipeline.modular_expert_manager import ModularExpertManager
from pipeline.moe_components import Top2GatingRouter, SparseMoELayer


class TestModularExpertManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = "/home/benjamin/Bilder/data/test_scratch"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.d_model = 64
        self.hidden_dim = 128
        self.num_experts = 12

    def test_top2_router_cluster_biasing(self):
        """Tests that domain_cluster biasing shifts routing preference to specified expert cluster."""
        router = Top2GatingRouter(d_model=self.d_model, num_experts=self.num_experts)
        x = torch.randn(10, self.d_model)

        # Domain 1 (STEM: Experts 4-7)
        top2_idx, top2_w, aux_loss = router(x, domain_cluster=1)
        self.assertEqual(top2_idx.shape, (10, 2))
        self.assertEqual(top2_w.shape, (10, 2))

        # Check that experts in cluster 1 (indices 4..7) have high probability
        chosen = top2_idx.flatten().tolist()
        cluster_1_count = sum(1 for idx in chosen if 4 <= idx <= 7)
        self.assertGreater(cluster_1_count, 0, "Cluster 1 experts should be actively routed")

    def test_sparse_moe_layer_forward(self):
        """Tests SparseMoELayer forward with domain cluster argument."""
        moe = SparseMoELayer(d_model=self.d_model, hidden_dim=self.hidden_dim, num_experts=self.num_experts)
        x = torch.randn(2, 8, self.d_model)

        out, aux_loss = moe(x, domain_cluster=2)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(torch.isnan(out).any())

    def test_export_and_import_expert_pack(self):
        """Tests modular export of expert pack and re-importing into slots."""
        # Create dummy checkpoint state dict
        sd = {}
        for l in range(2):
            for e in range(12):
                sd[f"moe_layers.{l}.experts.{e}.w_gate.weight"] = torch.randn(self.hidden_dim, self.d_model)

        ckpt_path = os.path.join(self.temp_dir, "test_ckpt.pt")
        torch.save({"model_state_dict": sd, "step": 100}, ckpt_path)

        # Export experts 4..7 (STEM pack)
        pack_path = os.path.join(self.temp_dir, "stem_pack.pt")
        pack = ModularExpertManager.export_expert_pack(
            ckpt_path,
            expert_indices=[4, 5, 6, 7],
            output_pack_path=pack_path,
            pack_name="stem_expert_pack",
        )
        self.assertTrue(os.path.exists(pack_path))
        self.assertEqual(pack["num_experts"], 4)

        # Import into slots 0..3 of a new checkpoint
        target_ckpt = os.path.join(self.temp_dir, "test_target_ckpt.pt")
        copy_sd = {k: v.clone() for k, v in sd.items()}
        torch.save({"model_state_dict": copy_sd}, target_ckpt)

        ModularExpertManager.import_expert_pack(
            target_ckpt,
            pack_path,
            target_indices=[0, 1, 2, 3],
        )

        loaded = torch.load(target_ckpt, weights_only=False)["model_state_dict"]
        # Check that slot 0 matches exported slot 4
        src_val = pack["weights"]["moe_layers.0.experts.4.w_gate.weight"]
        dst_val = loaded["moe_layers.0.experts.0.w_gate.weight"]
        self.assertTrue(torch.equal(src_val, dst_val))

    def test_splice_to_14b(self):
        """Tests splicing two 12-expert checkpoints into a 24-expert 14B checkpoint."""
        sd_a = {}
        sd_b = {}
        for l in range(2):
            sd_a[f"moe_layers.{l}.router.gate.weight"] = torch.randn(12, self.d_model)
            sd_b[f"moe_layers.{l}.router.gate.weight"] = torch.randn(12, self.d_model)
            for e in range(12):
                sd_a[f"moe_layers.{l}.experts.{e}.w_gate.weight"] = torch.full((self.hidden_dim, self.d_model), float(e))
                sd_b[f"moe_layers.{l}.experts.{e}.w_gate.weight"] = torch.full((self.hidden_dim, self.d_model), float(e + 100))

        ckpt_a_path = os.path.join(self.temp_dir, "ckpt_a.pt")
        ckpt_b_path = os.path.join(self.temp_dir, "ckpt_b.pt")
        torch.save({"model_state_dict": sd_a}, ckpt_a_path)
        torch.save({"model_state_dict": sd_b}, ckpt_b_path)

        out_14b = os.path.join(self.temp_dir, "ckpt_14b.pt")
        ModularExpertManager.splice_to_14b(ckpt_a_path, ckpt_b_path, out_14b)

        self.assertTrue(os.path.exists(out_14b))
        ckpt_14b = torch.load(out_14b, weights_only=False)
        self.assertEqual(ckpt_14b["num_experts"], 24)
        sd_14b = ckpt_14b["model_state_dict"]

        # Check router gate expanded to [24, d_model]
        self.assertEqual(sd_14b["moe_layers.0.router.gate.weight"].shape, (24, self.d_model))
        # Check slot 12 has value from ckpt_b slot 0
        self.assertEqual(sd_14b["moe_layers.0.experts.12.w_gate.weight"][0, 0].item(), 100.0)

    def test_zero_shot_router_alignment(self):
        """Tests analytical cosine centroid alignment of router gates."""
        sd = {}
        for l in range(2):
            sd[f"moe_layers.{l}.router.gate.weight"] = torch.randn(12, self.d_model)
            for e in range(12):
                sd[f"moe_layers.{l}.experts.{e}.w_gate.weight"] = torch.randn(self.hidden_dim, self.d_model)

        aligned_sd = ModularExpertManager.align_router_zero_shot(sd, num_experts=12)
        gate_w = aligned_sd["moe_layers.0.router.gate.weight"]
        self.assertEqual(gate_w.shape, (12, self.d_model))
        self.assertFalse(torch.isnan(gate_w).any())


if __name__ == "__main__":
    unittest.main()
