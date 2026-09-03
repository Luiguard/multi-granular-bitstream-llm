#!/usr/bin/env python3
"""Test KV-Cache inference parity with standard forward pass."""

import torch
from pipeline.moe_7b_model import MultiGranularMoE7BModel

def test_kv_cache_generation():
    print("Testing MultiGranularMoE7BModel Static KV-Cache Generation...")
    torch.manual_seed(42)

    # Small test model
    model = MultiGranularMoE7BModel(
        vocab_size=256,
        rank=16,
        d_model=64,
        n_layers=2,
        num_experts=2,
        hidden_dim=128,
        use_shared_expert=False,
        routing_k=1,
    )
    model.eval()

    prompt_tokens = torch.tensor([[10, 25, 42, 99]], dtype=torch.long)

    # 1. Prefill step with prompt tokens
    logits_prefill, kv_caches = model.generate_step(prompt_tokens)
    next_token_1 = logits_prefill.argmax(dim=-1).unsqueeze(0)
    print(f"  Token 1 (from prefill): {next_token_1.item()}")

    # 2. Generate token 2 with single token input using KV Cache!
    logits_step2, kv_caches = model.generate_step(next_token_1, kv_caches=kv_caches)
    next_token_2 = logits_step2.argmax(dim=-1).unsqueeze(0)
    print(f"  Token 2 (from KV-Cache step): {next_token_2.item()}")

    # 3. Ground Truth: Compute standard forward pass with full sequence [10, 25, 42, 99, next_token_1]
    full_seq = torch.cat([prompt_tokens, next_token_1], dim=1)
    full_logits, _ = model(full_seq)
    expected_token_2 = full_logits[0, -1, :].argmax(dim=-1).item()
    print(f"  Expected Token 2 (from full forward): {expected_token_2}")

    assert next_token_2.item() == expected_token_2, "KV-Cache token prediction mismatch!"
    print("  ✅ KV-Cache generation matched full forward pass exactly!")
    print("ALL KV-CACHE TESTS PASSED!")

if __name__ == "__main__":
    test_kv_cache_generation()
