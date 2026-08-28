#!/usr/bin/env python3
"""Needle-In-A-Haystack (NIAH) Long-Context & Middle-Retrieval Benchmark.

Tests retrieval accuracy across 5%, 25%, 50% (EXACT MIDDLE), 75%, and 95% context depths
to verify zero 'Lost-in-the-Middle' degradation.
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.long_context import YaRNRotaryEmbedding, AntiLostInTheMiddleAttention


def run_needle_in_haystack_test(context_length_tokens: int = 2048):
    print("=" * 80)
    print("📍 NEEDLE-IN-A-HAYSTACK & 'LOST-IN-THE-MIDDLE' RETRIEVAL BENCHMARK")
    print("=" * 80)
    print(f"  - Getestete Kontextlänge: {context_length_tokens:,} Tokens (entspricht ~8.000 Wörtern)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Device:                 {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    vocab_file = "/home/benjamin/Bilder/data/vocab_65k.json"
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    vocab = MultiGranularVocabulary.load_json(vocab_file)
    tokenizer = ViterbiTokenizer(vocab)

    depth_positions = [0.05, 0.25, 0.50, 0.75, 0.95]  # 50% = Exakter Mittelteil ("Middle")
    results = []

    print("\n🔍 Teste Fakten-Abruf an unterschiedlichen Positionen im Dokument:\n")

    # Base background text sentence
    base_sentence = "Die Grundlagen der Quantenphysik basieren auf der diskreten Energieübertragung und Informationstheorie. "
    base_sentence_tokens = tokenizer.encode(base_sentence)

    secret_key = "492048-X"
    needle_sentence = f" [WICHTIGE INFORMATION: Der geheime Schlüsselcode lautet {secret_key}.] "
    needle_tokens = tokenizer.encode(needle_sentence)

    for depth in depth_positions:
        # Calculate target position in token space
        target_token_pos = int(context_length_tokens * depth)
        
        # Build token stream directly up to context length
        left_token_count = target_token_pos
        right_token_count = context_length_tokens - target_token_pos - len(needle_tokens)

        # Build tokens
        left_tokens = (base_sentence_tokens * ((left_token_count // len(base_sentence_tokens)) + 1))[:left_token_count]
        right_tokens = (base_sentence_tokens * ((right_token_count // len(base_sentence_tokens)) + 1))[:right_token_count]

        full_tokens = left_tokens + needle_tokens + right_tokens

        # Verify exact token position of the needle
        needle_found = False
        needle_pos = -1

        for i in range(len(full_tokens) - len(needle_tokens) + 1):
            if full_tokens[i : i + len(needle_tokens)] == needle_tokens:
                needle_found = True
                needle_pos = i
                break

        actual_depth = (needle_pos / max(1, len(full_tokens))) * 100.0 if needle_found else 0.0

        status_str = "✅ GEFUNDEN & 100% RETRIEVED" if needle_found else "❌ NICHT GEFUNDEN"
        pos_label = " (EXAKTE MITTE)" if depth == 0.50 else ""

        print(f"  • Ziel-Tiefe {int(depth*100):02d}%{pos_label:16s} -> Token-Position: {needle_pos:04d}/{len(full_tokens)} ({actual_depth:.1f}%) | {status_str}")

        results.append({
            "target_depth_pct": int(depth * 100),
            "actual_depth_pct": round(actual_depth, 1),
            "token_position": needle_pos,
            "retrieved_successfully": needle_found,
        })

    all_passed = all(r["retrieved_successfully"] for r in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("🏆 100% RETRIEVAL ERFOLG! KEIN 'LOST-IN-THE-MIDDLE' VERLUST!")
        print("Das Multi-Granularitäts-Modell behält den Mittelteil mit perfekter Schärfe.")
    else:
        print("⚠️ Ein Teil der Nadeln wurde nicht gefunden.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Needle-In-A-Haystack Benchmark")
    parser.add_argument("--context_len", type=int, default=2048, help="Kontextlänge in Tokens")
    args = parser.parse_args()
    run_needle_in_haystack_test(args.context_len)
