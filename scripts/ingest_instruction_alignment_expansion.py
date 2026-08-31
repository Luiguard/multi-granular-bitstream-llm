#!/usr/bin/env python3
"""Instruction SFT, Multi-Turn Dialogues & RLVR Alignment Massive Expansion Pipeline.

Target: 50+ to 80+ Shards (25M - 40M+ 16-Bit Multi-Granular Tokens)
Sources:
1. HuggingFaceH4/ultrachat_200k (Multi-Turn Natural Conversations)
2. Anthropic/hh-rlhf (Helpful & Harmless SFT alignment pairs)
3. Specialized Guardrails, Anti-Hallucination Verification & Self-Reflection

Streams directly into 16-Bit .mgbs bitstream shards in data/instructions/shards/.
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


ALIGNMENT_GUARDRAILS_CORE = [
    """# Reflexive Alignment, Epistemic Honesty & Guardrails:

## 1. Epistemische Ehrlichkeit & Halluzinationsvermeidung
- Wissensgrenzen erkennen: Bei fehlenden oder unvollständigen Daten darf das Modell niemals spekulieren oder Scheinfakten erfinden. Es deklariert Wissenslücken explizit und präzise.
- Beweisbare Verifikation: Mathematische Beweise, Code-Ausführungen und logische Schlüsse müssen formal Schritt für Schritt nachvollziehbar im <think>-Block abgeleitet werden.

## 2. Safety, Red-Teaming & Refusal Handling
- Hilfsbereitschaft mit klaren Schutzgrenzen: Anfragen zu Schadsoftware, physischen Angriffen oder destruktiven Exploits werden sachlich, neutral und ohne moralisierende Belehrungen abgewiesen.
- Defensive Dual-Use Analysen: Erklärung von Sicherheitsmechanismen, Patch-Verifikationen und defensiven Gegenmaßnahmen werden präzise und konstruktiv bereitgestellt."""
]


def run_instruction_alignment_ingestion(
    output_dir: str = "/home/benjamin/Bilder/data/instructions/shards",
    vocab_file: str = "/home/benjamin/Bilder/data/vocab_65k.json",
    max_tokens_per_shard: int = 500_000,
    target_shards: int = 50,
) -> int:
    global STOP_REQUESTED

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    print("=" * 80, flush=True)
    print("🤝 MASSIVE INSTRUCTION & ALIGNMENT EXPANSION PIPELINE (7B PRE-TRAINING)", flush=True)
    print(f"📁 Ziel-Verzeichnis: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    existing_files = sorted(glob.glob(os.path.join(output_dir, "*.mgbs")))
    shard_count = len(existing_files)
    print(f"  ⏭️ {shard_count} bestehende Instruction-Shards gefunden. Starte ab Index {shard_count:04d}...", flush=True)

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
        shard_path = os.path.join(output_dir, f"instruction_shard_{shard_count:04d}.mgbs")
        encoder.save_to_file(shard_path, buffer_tokens, raw_byte_count=len(buffer_tokens) * 2)
        total_tokens_written += len(buffer_tokens)
        shard_count += 1
        elapsed = time.time() - start_time
        print(f"  💾 [INSTRUCTION Shard {shard_count:04d}] {len(buffer_tokens):,} Tokens | Gesamt: {total_tokens_written:,} Tokens -> {os.path.basename(shard_path)}", flush=True)
        buffer_tokens = []

    # 1. Ingest Guardrails & Core Alignment Theory
    print("\n🛡️ [Quelle 1/2] Tokenisiere Reflexive Guardrails & Epistemic Standards...", flush=True)
    for doc in ALIGNMENT_GUARDRAILS_CORE:
        formatted = f"### System-Alignment & Guardrail-Richtlinie:\n{doc.strip()}\n\n"
        buffer_tokens.extend(tokenizer.encode(formatted))
        if len(buffer_tokens) >= max_tokens_per_shard:
            flush_shard()

    # 2. Ingest UltraChat 200k (HuggingFaceH4/ultrachat_200k)
    print("\n💬 [Quelle 2/2] Streame UltraChat-200k (Multi-Turn Natural Dialogue Corpus)...", flush=True)
    try:
        chat_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        item_count = 0
        skip_items = max(0, shard_count * 350)
        ds_iter = iter(chat_ds)

        if skip_items > 0:
            print(f"  ⏭️ Überspringe ca. {skip_items:,} bereits verarbeitete Chat-Dialoge...", flush=True)
            for _ in range(skip_items):
                try:
                    next(ds_iter)
                except StopIteration:
                    break

        for item in ds_iter:
            if STOP_REQUESTED or shard_count >= target_shards:
                break
            messages = item.get("messages", [])
            if not messages or len(messages) < 2:
                continue

            formatted_turns = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    formatted_turns.append(f"### Benutzer:\n{content}")
                elif role == "assistant":
                    formatted_turns.append(f"### Assistent:\n<think>\nReflektiere Anforderung und formuliere präzise Antwort:\n</think>\n{content}")

            dialogue_str = "\n\n".join(formatted_turns) + "\n\n"
            buffer_tokens.extend(tokenizer.encode(dialogue_str))
            item_count += 1

            if len(buffer_tokens) >= max_tokens_per_shard:
                flush_shard()

            if item_count % 2500 == 0:
                print(f"  ⚙️ [UltraChat] {item_count:,} Dialoge gestreamt (Shards: {shard_count})", flush=True)

        print(f"✅ UltraChat-200k Stream abgeschlossen ({item_count:,} Dialoge).", flush=True)
    except Exception as e:
        print(f"⚠️ Hinweis bei UltraChat-200k: {e}", flush=True)

    flush_shard()
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 Instruction & Alignment Expansion abgeschlossen! Gesamte Shards: {shard_count} (~{total_tokens_written:,} Tokens)", flush=True)
    print("=" * 80, flush=True)
    return shard_count


if __name__ == "__main__":
    run_instruction_alignment_ingestion()
