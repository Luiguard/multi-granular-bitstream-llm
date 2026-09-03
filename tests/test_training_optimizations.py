#!/usr/bin/env python3
"""Test training loop optimizations:
1. Selective Weight Decay in GaLoreAdamW
2. Confidence-Gated Dynamic-K MoE Sparsity
3. Soft Curriculum Blending in CurriculumTrainingGraph
4. Cross-Entropy Chunk-Size 128 equivalence
"""

import os
import torch
import torch.nn as nn
from pipeline.galore_optimizer import GaLoreAdamW
from pipeline.moe_7b_model import MultiGranularMoE7BModel, SparseMoELayer16
from pipeline.training_graph import TrainingKnowledgeGraph

def test_selective_weight_decay():
    print("Testing Selective Weight Decay in GaLoreAdamW...")
    w2d = nn.Parameter(torch.ones(10, 10))
    b1d = nn.Parameter(torch.ones(10))
    
    # Artificial gradients
    w2d.grad = torch.zeros_like(w2d)
    b1d.grad = torch.zeros_like(b1d)
    
    param_groups = [
        {"params": [w2d], "weight_decay": 0.1, "rank": 4},
        {"params": [b1d], "weight_decay": 0.0, "rank": 4},
    ]
    opt = GaLoreAdamW(param_groups, lr=1e-2)
    
    # Step param
    opt.step_param(w2d)
    opt.step_param(b1d)
    
    # 2D weight should experience weight decay (shrunk from 1.0)
    assert w2d.data[0, 0].item() < 1.0, "Weight decay was not applied to 2D weight!"
    # 1D bias should NOT experience weight decay (remains 1.0)
    assert abs(b1d.data[0].item() - 1.0) < 1e-6, "Weight decay was incorrectly applied to 1D bias!"
    print("  ✅ Selective weight decay verified: 2D decayed, 1D preserved!")


def test_confidence_gated_sparsity():
    print("Testing Confidence-Gated Dynamic-K MoE Sparsity...")
    moe = SparseMoELayer16(d_model=64, hidden_dim=128, num_experts=4)
    moe.eval()
    
    x = torch.randn(2, 4, 64)
    out, aux = moe(x)
    assert out.shape == (2, 4, 64)
    assert not torch.isnan(out).any()
    print("  ✅ Dynamic-K MoE sparsity executed smoothly with valid outputs!")


def test_curriculum_blending():
    print("Testing Soft Curriculum Blending...")
    graph = TrainingKnowledgeGraph()
    
    # Sample 40 batches and verify exploratory nodes are blended
    sampled_nodes = set()
    for _ in range(40):
        _, _, node_id = graph.sample_batch(batch_size=1, seq_len=128)
        sampled_nodes.add(node_id)
        
    print(f"  Sampled nodes in test: {sampled_nodes}")
    assert "node_0_foundation" in sampled_nodes
    has_exploratory = any(n in sampled_nodes for n in ("node_1_cyber_web", "node_2_stem_math"))
    assert has_exploratory, "Soft curriculum blending did not sample any exploratory nodes!"
    print("  ✅ Soft curriculum blending successfully exposed exploratory nodes!")


def test_chunk_size_128():
    print("Testing Cross-Entropy Chunk-Size 128...")
    model = MultiGranularMoE7BModel(vocab_size=256, rank=16, d_model=64, n_layers=1, num_experts=2, hidden_dim=128)
    x = torch.randint(0, 256, (2, 256))
    y = torch.randint(0, 256, (2, 256))
    
    loss_16, _, _ = model.compute_loss(x, y, chunk_size=16)
    loss_128, _, _ = model.compute_loss(x, y, chunk_size=128)
    
    assert abs(loss_16.item() - loss_128.item()) < 1e-4, "Chunk size change altered the mathematical loss!"
    print(f"  Loss chunk 16:  {loss_16.item():.6f}")
    print(f"  Loss chunk 128: {loss_128.item():.6f}")
    print("  ✅ Chunk size 128 produces identical mathematical loss with 8x fewer loop iterations!")


if __name__ == "__main__":
    test_selective_weight_decay()
    test_confidence_gated_sparsity()
    test_curriculum_blending()
    test_chunk_size_128()
    print("ALL TRAINING OPTIMIZATION TESTS PASSED!")
