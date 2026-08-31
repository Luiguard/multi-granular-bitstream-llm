#!/usr/bin/env python3
"""
Comprehensive Verification Test Suite for the 4 Advanced LLM Subsystems:
1. RLVR Verifier (Code & Math Sandbox Execution).
2. Multimodal 16-Bit Vision Bitstream Projector.
3. YaRN Long-Context RoPE (Context Extension up to 32k).
4. CUDA Throughput Accelerator (AMP & SDPA FlashAttention).
"""

import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, "/home/benjamin/Bilder")

from pipeline.rlvr_verifier import RLVRVerifier
from pipeline.vision_bitstream_projector import VisionBitstreamProjector
from pipeline.yarn_rope import YaRNScaledRotaryEmbedding
from pipeline.throughput_accelerator import CUDAAccelerator


class TestFourLLMPillars(unittest.TestCase):
    def test_01_rlvr_code_and_math_execution(self):
        """Test RLVR execution rewards on correct code vs runtime error."""
        verifier = RLVRVerifier(timeout_seconds=2.0)
        
        # Test A: Correct code -> Reward +1.0
        good_response = """
<think>
Berechne die Summe der ersten 10 Quadratzahlen:
</think>
```python
total = sum(x**2 for x in range(1, 11))
print(f"Result: {total}")
```
"""
        res_good = verifier.verify_response(good_response)
        self.assertEqual(res_good["mean_reward"], 1.0)
        self.assertEqual(res_good["details"][0]["status"], "SUCCESS")
        self.assertIn("Result: 385", res_good["details"][0]["stdout"])

        # Test B: Broken code -> Negative Reward
        bad_response = """
```python
x = 1 / 0
```
"""
        res_bad = verifier.verify_response(bad_response)
        self.assertLess(res_bad["mean_reward"], 0.0)
        self.assertEqual(res_bad["details"][0]["status"], "RUNTIME_ERROR")
        print("✅ RLVR Verifier Test: Validated positive (+1.0) and negative rewards cleanly.")

    def test_02_vision_bitstream_projection(self):
        """Test 2D Patch Convolution and tokenization into 16-bit visual range."""
        projector = VisionBitstreamProjector(image_size=224, patch_size=16, d_model=512)
        sample_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        tokens = projector.project_to_tokens(sample_img)
        self.assertEqual(len(tokens), 196)  # (224/16)^2 = 196 patches
        
        # Check token IDs within the 16-bit range
        for t in tokens:
            self.assertGreaterEqual(t, 60000)
            self.assertLessEqual(t, 65535)

        ctx_str = projector.format_vision_context(sample_img, "Test-Diagramm")
        self.assertIn("196 Patches", ctx_str)
        print("✅ Multimodal Vision Projector Test: 196 Patches projected into 16-Bit Viterbi Token space.")

    def test_03_yarn_long_context_rope(self):
        """Test YaRN RoPE frequency interpolation for extended 32k context."""
        yarn = YaRNScaledRotaryEmbedding(dim=64, scale=4.0, max_position_embeddings=32768)
        
        q = torch.randn(1, 8, 2048, 64)
        k = torch.randn(1, 8, 2048, 64)
        q_rot, k_rot = yarn(q, k, seq_len=2048)
        
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        self.assertGreater(yarn.mscale, 1.0)
        print(f"✅ YaRN Long-Context Scaler Test: 4x Scale validated (Temperature MScale: {yarn.mscale:.4f}).")

    def test_04_cuda_throughput_accelerator(self):
        """Test Flash-SDPA and AMP Throughput Accelerator."""
        accelerator = CUDAAccelerator()
        bench = accelerator.benchmark_throughput(batch_size=2, seq_len=512, iterations=20)
        
        self.assertGreater(bench["throughput_tokens_per_sec"], 1000)
        print(f"✅ CUDA Accelerator Test: {bench['throughput_tokens_per_sec']:,} Tokens/s auf {bench['device']}.")


if __name__ == "__main__":
    unittest.main()
