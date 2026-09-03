#!/usr/bin/env python3
"""Test FactorizedChunkedCrossEntropy with target loss masking (ignore_index=-100)."""

import torch
import torch.nn.functional as F
from pipeline.moe_7b_model import FactorizedChunkedCrossEntropy

def test_chunked_ce_masking():
    print("Testing FactorizedChunkedCrossEntropy ignore_index=-100 masking...")
    torch.manual_seed(42)
    
    T = 64
    rank = 16
    vocab_size = 256
    
    proj_h = torch.randn(T, rank, requires_grad=True)
    head_weight = torch.randn(vocab_size, rank, requires_grad=True)
    targets = torch.randint(0, vocab_size, (T,))
    
    # Mask half the tokens with -100 (simulating prompt masking in SFT)
    targets[::2] = -100
    
    # 1. Compute with custom autograd
    loss_custom = FactorizedChunkedCrossEntropy.apply(proj_h, head_weight, targets, 16, -100)
    loss_custom.backward()
    
    assert proj_h.grad is not None
    assert head_weight.grad is not None
    grad_h_custom = proj_h.grad.clone()
    grad_w_custom = head_weight.grad.clone()
    
    # 2. Compute ground truth via PyTorch standard F.cross_entropy
    proj_h_ref = proj_h.detach().clone().requires_grad_(True)
    head_weight_ref = head_weight.detach().clone().requires_grad_(True)
    
    logits = F.linear(proj_h_ref, head_weight_ref)
    loss_ref = F.cross_entropy(logits, targets, ignore_index=-100)
    loss_ref.backward()
    
    assert proj_h_ref.grad is not None
    assert head_weight_ref.grad is not None
    
    print(f"  Custom Loss: {loss_custom.item():.6f}")
    print(f"  Ref Loss:    {loss_ref.item():.6f}")
    assert abs(loss_custom.item() - loss_ref.item()) < 1e-4, "Forward loss mismatch!"
    
    # Check gradient similarity
    h_diff = (grad_h_custom - proj_h_ref.grad).abs().max().item()
    w_diff = (grad_w_custom - head_weight_ref.grad).abs().max().item()
    print(f"  Max grad_h difference: {h_diff:.6f}")
    print(f"  Max grad_w difference: {w_diff:.6f}")
    assert h_diff < 1e-4, "Backward grad_h mismatch!"
    assert w_diff < 1e-4, "Backward grad_w mismatch!"
    
    # Check that masked token gradients are zero
    # Tokens with target == -100 should have zero contribution to grad_h
    masked_indices = (targets == -100)
    assert (grad_h_custom[masked_indices].abs() == 0.0).all(), "Masked tokens must have 0 gradient!"
    
    print("  ✅ Forward and backward target masking verified with exact mathematical parity!")
    print("ALL MASKING TESTS PASSED!")

if __name__ == "__main__":
    test_chunked_ce_masking()
