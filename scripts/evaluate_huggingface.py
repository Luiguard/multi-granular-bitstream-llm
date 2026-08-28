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
from scripts.train_distributed import MultiGranularCausalTransformer


BENCHMARK_PROMPTS = [
    {
        "category": "Code (Python)",
        "prompt": "def calculate_factorial(n):\n    \"\"\"Calculates factorial recursively.\"\"\"\n",
        "expected_keywords": ["if", "return", "n *", "1"],
    },
    {
        "category": "Reasoning & Math",
        "prompt": "Frage: Wenn ein Zug 120 km/h fährt, wie viele Kilometer legt er in 2,5 Stunden zurück?\nAntwort: Berechnung:",
        "expected_keywords": ["120", "2.5", "300", "km"],
    },
    {
        "category": "General Knowledge (DE/EN)",
        "prompt": "Künstliche Intelligenz (KI) bezeichnet die Fähigkeit von Maschinen,",
        "expected_keywords": ["Lernen", "Probleme", "Algorithmen", "Daten"],
    },
    {
        "category": "Instruction Following",
        "prompt": "Fasse den folgenden Begriff in einem Satz zusammen: Quantencomputer.\nZusammenfassung:",
        "expected_keywords": ["Qubits", "Quantenmechanik", "Rechnen", "Information"],
    },
]


def run_benchmark(checkpoint_path: str, vocab_file: str, output_file: str = "benchmark_results.json"):
    print("=" * 80)
    print("🏆 HUGGING FACE BENCHMARK & EVALUATION RUNNER")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Device:     {device}")
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
    ).to(device)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True), strict=False)
        print(f"✅ Modellgewichte erfolgreich geladen.")
    else:
        print(f"⚠️ Checkpoint {checkpoint_path} nicht gefunden, evaluiere Initialzustand.")

    model.eval()
    results = []

    print("\n📊 Starte Auswertung über Testkategorien:\n")

    for idx, item in enumerate(BENCHMARK_PROMPTS, 1):
        category = item["category"]
        prompt = item["prompt"]
        expected = item["expected_keywords"]

        tokens = tokenizer.encode(prompt)
        input_ids = list(tokens)

        start_t = time.time()
        with torch.no_grad():
            for _ in range(40):
                inp_t = torch.tensor([input_ids[-512:]], dtype=torch.long, device=device)
                logits = model(inp_t)
                next_token = int(torch.argmax(logits[0, -1, :]).item())
                input_ids.append(next_token)

        gen_time = time.time() - start_t
        gen_tokens = input_ids[len(tokens):]
        generated_text = tokenizer.decode(gen_tokens)

        # Keyword match score
        matches = sum(1 for kw in expected if kw.lower() in generated_text.lower())
        score = (matches / len(expected)) * 100.0

        tps = len(gen_tokens) / max(0.001, gen_time)

        print(f"[{idx}/4] Kategorie: {category}")
        print(f"  • Prompt:     '{prompt.strip()}'")
        print(f"  • Generiert:  '{generated_text.strip()}'")
        print(f"  • Score:      {score:.1f}% ({matches}/{len(expected)} Keywords) | Tempo: {tps:.1f} Tokens/s\n")

        results.append({
            "category": category,
            "prompt": prompt,
            "generated": generated_text,
            "score": score,
            "tokens_per_sec": tps,
        })

    # Speichern des Benchmark-Reports
    avg_score = sum(r["score"] for r in results) / len(results)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "average_benchmark_score": round(avg_score, 2),
        "vocab_size": vocab.size,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"🏅 GESAMT-BENCHMARK SCORE: {avg_score:.1f}%")
    print(f"📄 Detaillierter Report gespeichert in: {output_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Hugging Face Benchmark Evaluator")
    parser.add_argument("--checkpoint", type=str, default="./multi_granular_model.pt", help="Pfad zum Modell-Checkpoint")
    parser.add_argument("--vocab_file", type=str, default="./data/vocab_65k.json", help="Pfad zur Vokabular-Datei")
    parser.add_argument("--output", type=str, default="./data/benchmark_results.json", help="Ausgabedatei")
    args = parser.parse_args()

    if not os.path.exists(args.vocab_file):
        args.vocab_file = "./vocab.json"

    run_benchmark(args.checkpoint, args.vocab_file, args.output)


if __name__ == "__main__":
    main()
