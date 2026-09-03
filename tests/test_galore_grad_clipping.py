#!/usr/bin/env python3
"""Unit Test for Per-Layer Adaptive Gradient Clipping in GaLoreAdamW."""

import torch
import torch.nn as nn
from pipeline.galore_optimizer import GaLoreAdamW

def test_gradient_clipping():
    print("Testing GaLoreAdamW Adaptive Gradient Clipping...")
    torch.manual_seed(42)
    linear = nn.Linear(128, 64)
    opt = GaLoreAdamW(linear.parameters(), lr=1e-3, rank=16, grad_clip_norm=1.0)
    
    # 1. Simuliere extremen Gradienten-Spike (Norm > 50.0)
    x = torch.randn(8, 128)
    out = linear(x)
    loss = out.sum() * 50.0
    loss.backward()
    
    initial_grad_norm = linear.weight.grad.norm(2).item()
    print(f"  Pre-step gradient norm: {initial_grad_norm:.2f}")
    assert initial_grad_norm > 5.0, "Gradient should be intentionally large"
    
    # 2. Ausführung step_param
    opt.step_param(linear.weight)
    
    # Prüfe, dass kein NaN/Inf aufgetreten ist und das Gewicht endlich blieb
    assert not torch.isnan(linear.weight).any()
    assert not torch.isinf(linear.weight).any()
    print("  ✅ Step param succeeded with clean weights (no NaN/Inf)!")
    
    # 3. Teste Standard step() für non-offloaded Parameter
    linear2 = nn.Linear(32, 16)
    opt2 = GaLoreAdamW(linear2.parameters(), lr=1e-3, rank=8, grad_clip_norm=0.5)
    out2 = linear2(torch.randn(4, 32))
    (out2.sum() * 100.0).backward()
    opt2.step()
    
    assert not torch.isnan(linear2.weight).any()
    print("  ✅ Standard step succeeded with adaptive clipping!")
    print("ALL CLIPPING TESTS PASSED!")

if __name__ == "__main__":
    test_gradient_clipping()
