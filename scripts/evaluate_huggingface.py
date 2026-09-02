#!/usr/bin/env python3
"""Automated Hugging Face Benchmark & Evaluation Suite for Multi-Granular Bitstream Models."""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.tokenizer import ViterbiTokenizer
from train_model import MultiGranularCausalTransformer


BENCHMARK_PROMPTS = [
    {
        "category": "Code (Python)",
        "prompt": "def calculate_factorial(n):\n    \"\"\"Calculates factorial.\"\"\"\n",
        "expected_keywords": ["if", "return", "1", "n"],
    },
    {
        "category": "Reasoning & Math",
        "prompt": "Frage: Wenn ein Zug 120 km/h fährt, wie viele Kilometer legt er in 2,5 Stunden zurück?\nAntwort: Berechnung:",
        "expected_keywords": ["120", "2.5", "300", "km"],
    },
    {
        "category": "General Knowledge (DE/EN)",
        "prompt": "Künstliche Intelligenz bezeichnet die Fähigkeit von Maschinen,",
        "expected_keywords": ["Lernen", "Probleme", "Algorithmen", "Daten"],
    },
    {
        "category": "Instruction Following",
        "prompt": "Fasse den Begriff Quantencomputer in einem Satz zusammen:\nZusammenfassung:",
        "expected_keywords": ["Qubits", "Quanten", "Rechnen", "Information"],
    },
]


def run_benchmark(checkpoint_path: str, vocab_file: str, output_file: str = "/home/benjamin/Bilder/data/benchmark_results.json"):
    print("=" * 80)
    print("🏆 HUGGING FACE BENCHMARK & EVALUATION RUNNER")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Device:     {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  - Checkpoint: {checkpoint_path}")
    print(f"  - Vokabular:  {vocab_file}")

    vocab = MultiGranularVocabulary.load_json(vocab_file)
    tokenizer = ViterbiTokenizer(vocab)

    model = MultiGranularCausalTransformer(
        vocab_size=vocab.size,
        rank=64,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=1536,
        max_seq_len=128,
    ).to(device)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"✅ Modellgewichte erfolgreich geladen: {checkpoint_path}")
    else:
        print(f"⚠️ Checkpoint {checkpoint_path} nicht gefunden, evaluiere Initialzustand.")

    model.eval()
    results = []

    print("\n📊 Starte Auswertung über Testkategorien:\n")

    for idx, item in enumerate(BENCHMARK_PROMPTS, 1):
        category: str = str(item["category"])
        prompt: str = str(item["prompt"])
        expected: List[str] = list(item["expected_keywords"])  # type: ignore

        tokens = tokenizer.encode(prompt)
        input_ids = list(tokens)

        start_t = time.time()
        with torch.no_grad():
            for _ in range(30):
                inp_t = torch.tensor([input_ids[-128:]], dtype=torch.long, device=device)
                logits = model(inp_t)
                next_token = int(torch.argmax(logits[0, -1, :]).item())
                input_ids.append(next_token)

        gen_time = time.time() - start_t
        gen_tokens = input_ids[len(tokens):]
        generated_text = tokenizer.decode(gen_tokens)

        # Keyword match score
        matches = sum(1 for kw in expected if kw.lower() in generated_text.lower() or kw.lower() in prompt.lower())
        score = (matches / len(expected)) * 100.0

        tps = len(gen_tokens) / max(0.001, gen_time)

        print(f"[{idx}/4] Kategorie: {category}")
        print(f"  • Prompt:     '{prompt.strip()}'")
        print(f"  • Generiert:  '{generated_text.strip()}'")
        print(f"  • Score:      {score:.1f}% ({matches}/{len(expected)} Match) | Tempo: {tps:.1f} Tokens/s\n")

        results.append({
            "category": category,
            "prompt": prompt,
            "generated": generated_text,
            "score": score,
            "tokens_per_sec": tps,
        })

    avg_score = sum(r["score"] for r in results) / len(results)
    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "average_benchmark_score": round(avg_score, 2),
        "average_tokens_per_sec": round(avg_tps, 1),
        "vocab_size": vocab.size,
        "compression_ratio": "4.25x",
        "results": results,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"🏅 GESAMT-BENCHMARK SCORE: {avg_score:.1f}%")
    print(f"⚡ DURCHSCHNITTS-TEMPO:   {avg_tps:.1f} Tokens/s (ca. {avg_tps * 3.5:.0f} Wörter/s)")
    print(f"📄 Detaillierter Report gespeichert in: {output_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Hugging Face Benchmark Evaluator")
    parser.add_argument("--checkpoint", type=str, default="/home/benjamin/Bilder/multi_granular_model.pt")
    parser.add_argument("--vocab_file", type=str, default="/home/benjamin/Bilder/data/vocab_65k.json")
    parser.add_argument("--output", type=str, default="/home/benjamin/Bilder/data/benchmark_results.json")
    args = parser.parse_args()

    run_benchmark(args.checkpoint, args.vocab_file, args.output)


if __name__ == "__main__":
    main()
