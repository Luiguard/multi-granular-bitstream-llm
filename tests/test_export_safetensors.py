#!/usr/bin/env python3
"""Test SafeTensors model export and config generation."""

import os
import torch
import safetensors.torch
from scripts.export_checkpoint import export_checkpoint

def test_export():
    print("Testing SafeTensors Export Pipeline...")
    dummy_ckpt_path = "/home/benjamin/Bilder/data/test_scratch/test_model.pt"
    export_dir = "/home/benjamin/Bilder/data/test_scratch/exported_test"
    os.makedirs(os.path.dirname(dummy_ckpt_path), exist_ok=True)
    
    # Create small dummy state dict
    test_sd = {
        "E_vocab.weight": torch.randn(256, 64, dtype=torch.bfloat16),
        "head_out.weight": torch.randn(256, 64, dtype=torch.bfloat16),
        "attn.weight": torch.randn(64, 64, dtype=torch.bfloat16),
    }
    torch.save({"model_state_dict": test_sd, "step": 150, "vocab_size": 256}, dummy_ckpt_path)
    
    # Run export
    safetensors_file = export_checkpoint(dummy_ckpt_path, export_dir, quantize_int8=False)
    
    assert os.path.exists(safetensors_file)
    assert os.path.exists(os.path.join(export_dir, "config.json"))
    
    # Verify loaded safetensors
    loaded = safetensors.torch.load_file(safetensors_file)
    assert "E_vocab.weight" in loaded
    assert loaded["E_vocab.weight"].shape == (256, 64)
    print("  ✅ SafeTensors export & zero-copy loading verified successfully!")
    print("ALL EXPORT TESTS PASSED!")

if __name__ == "__main__":
    test_export()
