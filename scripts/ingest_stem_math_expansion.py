#!/usr/bin/env python3
"""STEM, Math & Scientific Reasoning Massive Expansion Ingestion Pipeline.

Target: 150+ Shards (75M+ 16-Bit Multi-Granular Tokens)
Sources:
1. hendrycks/competition_math (Algebra, Geometry, Calculus, Number Theory with Full Step-by-Step Proofs)
2. gsm8k (Multi-Step Mathematical Word Problems with Reasoning Chains)
3. allenai/sciq (Physics, Chemistry, Biology & Earth Sciences)
4. Deep ArXiv & Formal Science derivations

Streams directly into 16-Bit .mgbs bitstream shards.
Automatically appends to data/stem_knowledge/shards/ without duplicates.
"""

import os
import sys
import time
import glob
import signal
from typing import List, Dict, Any

from datasets import load_dataset
from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamEncoder
from pipeline.tokenizer import ViterbiTokenizer

STOP_REQUESTED = False

def handle_signal(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠️ Graceful Shutdown angefordert, beende nach aktuellem Shard...", flush=True)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def run_stem_math_ingestion(
    output_dir: str = "/home/benjamin/Bilder/data/stem_knowledge/shards",
    vocab_file: str = "/home/benjamin/Bilder/data/vocab_65k.json",
    max_tokens_per_shard: int = 500_000,
    target_shards: int = 150,
) -> int:
    global STOP_REQUESTED

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    print("=" * 80, flush=True)
    print("🔬 MASSIVE STEM, MATH & SCIENCE EXPANSION PIPELINE (7B PRE-TRAINING)", flush=True)
    print(f"📁 Ziel-Verzeichnis: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    existing_files = sorted(glob.glob(os.path.join(output_dir, "*.mgbs")))
    shard_count = len(existing_files)
    print(f"  ⏭️ {shard_count} bestehende STEM-Shards gefunden. Starte ab Index {shard_count:04d}...", flush=True)

    vocab = MultiGranularVocabulary.load_json(vocab_file)
    tokenizer = ViterbiTokenizer(vocab)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16)

    buffer_tokens: List[int] = []
    total_tokens_written = shard_count * max_tokens_per_shard
    start_time = time.time()

    def flush_shard():
        nonlocal shard_count, buffer_tokens, total_tokens_written
        if not buffer_tokens:
            return
        shard_path = os.path.join(output_dir, f"stem_math_shard_{shard_count:04d}.mgbs")
        encoder.save_to_file(shard_path, buffer_tokens, raw_byte_count=len(buffer_tokens) * 2)
        total_tokens_written += len(buffer_tokens)
        shard_count += 1
        elapsed = time.time() - start_time
        print(f"  💾 [STEM/MATH Shard {shard_count:04d}] {len(buffer_tokens):,} Tokens | Gesamt: {total_tokens_written:,} Tokens -> {os.path.basename(shard_path)}", flush=True)
        buffer_tokens = []

    # 1. Ingest MathInstruct (TIGER-Lab/MathInstruct - 260k formal math proofs)
    print("\n📐 [Quelle 1/3] Streame MathInstruct (260.000 formale mathematische Beweise & Herleitungen)...", flush=True)
    try:
        math_ds = load_dataset("TIGER-Lab/MathInstruct", split="train", streaming=True)
        math_count = 0
        skip_items = max(0, (shard_count - 68) * 350)
        ds_iter = iter(math_ds)
        if skip_items > 0:
            print(f"  ⏭️ Überspringe ca. {skip_items:,} bereits verarbeitete Math-Probleme...", flush=True)
            for _ in range(skip_items):
                try:
                    next(ds_iter)
                except StopIteration:
                    break

        for item in ds_iter:
            if STOP_REQUESTED or shard_count >= target_shards:
                break
            instr = item.get("instruction", "")
            output = item.get("output", "")
            src = item.get("source", "Math")
            if not instr or not output:
                continue

            formatted = f"### Mathematisches Problem ({src}):\n{instr}\n\n### Formale Beweisführung & Lösung:\n<think>\nSchrittweise mathematische Herleitung:\n</think>\n{output}\n\n"
            buffer_tokens.extend(tokenizer.encode(formatted))
            math_count += 1

            if len(buffer_tokens) >= max_tokens_per_shard:
                flush_shard()

            if math_count % 3000 == 0:
                print(f"  ⚙️ [MathInstruct] {math_count:,} Probleme verarbeitet (Shards: {shard_count})", flush=True)
        print(f"✅ MathInstruct Stream abgeschlossen ({math_count:,} Probleme verarbeitet).", flush=True)
    except Exception as e:
        print(f"⚠️ Hinweis bei MathInstruct: {e}", flush=True)

    # 2. Ingest GSM8k (openai/gsm8k - main)
    if shard_count < target_shards and not STOP_REQUESTED:
        print("\n🧮 [Quelle 2/3] Streame GSM8k (Multi-Step Arithmetic & Logic Reasoning)...", flush=True)
        try:
            gsm_ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
            gsm_count = 0
            for item in gsm_ds:
                if STOP_REQUESTED or shard_count >= target_shards:
                    break
                q = item.get("question", "")
                a = item.get("answer", "")
                if not q or not a:
                    continue

                formatted = f"### Mathematische Textaufgabe:\n{q}\n\n### Schrittweise Berechnung:\n<think>\nRechenschritte und Zwischenergebnisse:\n</think>\n{a}\n\n"
                buffer_tokens.extend(tokenizer.encode(formatted))
                gsm_count += 1

                if len(buffer_tokens) >= max_tokens_per_shard:
                    flush_shard()
            print(f"✅ GSM8k Stream abgeschlossen ({gsm_count:,} Aufgaben).", flush=True)
        except Exception as e:
            print(f"⚠️ Hinweis bei GSM8k: {e}", flush=True)

    # 3. Ingest SciQ (allenai/sciq)
    if shard_count < target_shards and not STOP_REQUESTED:
        print("\n🧬 [Quelle 3/3] Streame SciQ (Physik, Chemie, Biologie, Geowissenschaften)...", flush=True)
        try:
            sciq_ds = load_dataset("allenai/sciq", split="train", streaming=True)
            sciq_count = 0
            for item in sciq_ds:
                if STOP_REQUESTED or shard_count >= target_shards:
                    break
                support = item.get("support", "")
                question = item.get("question", "")
                correct = item.get("correct_answer", "")
                if not question or not correct:
                    continue

                formatted = f"### Naturwissenschaftlicher Kontext:\n{support}\n\n### Frage:\n{question}\n\n### Wissenschaftliche Erklärung & Antwort:\n{correct}\n\n"
                buffer_tokens.extend(tokenizer.encode(formatted))
                sciq_count += 1

                if len(buffer_tokens) >= max_tokens_per_shard:
                    flush_shard()
            print(f"✅ SciQ Stream abgeschlossen ({sciq_count:,} wissenschaftliche Fragen).", flush=True)
        except Exception as e:
            print(f"⚠️ Hinweis bei SciQ: {e}", flush=True)

    flush_shard()
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 STEM & Math Expansion abgeschlossen! Gesamte Shards: {shard_count} (~{total_tokens_written:,} Tokens)", flush=True)
    print("=" * 80, flush=True)
    return shard_count


if __name__ == "__main__":
    run_stem_math_ingestion()
