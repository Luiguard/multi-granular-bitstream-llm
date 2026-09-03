#!/usr/bin/env python3
"""Test for Native Benchmark Evaluator."""

import torch
import torch.nn as nn
from pipeline.benchmark_evaluator import NativeBenchmarkEvaluator, BENCHMARK_TASKS

class MockTokenizer:
    def encode(self, text: str):
        # Deterministic ascii-based token mapping
        return [ord(c) % 256 for c in text]

class TinyModel(nn.Module):
    def __init__(self, vocab_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32)
        self.head = nn.Linear(32, vocab_size)
    def forward(self, x):
        return self.head(self.embed(x))

def test_evaluator():
    print("Testing NativeBenchmarkEvaluator...")
    model = TinyModel()
    tok = MockTokenizer()
    evaluator = NativeBenchmarkEvaluator(output_path="/home/benjamin/Bilder/data/test_scratch/benchmark_results.json")
    
    res = evaluator.evaluate(model, tok, torch.device("cpu"))
    print("  Benchmark evaluation finished with results:", res)
    assert "composite_benchmark_acc" in res
    assert "arc_challenge_acc" in res
    assert "hellaswag_acc" in res
    assert "mmlu_acc" in res
    print("ALL BENCHMARK EVALUATOR TESTS PASSED!")

if __name__ == "__main__":
    test_evaluator()
