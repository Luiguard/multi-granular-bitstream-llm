#!/usr/bin/env python3
"""Multi-Domain Instruction & Reasoning Ingestion Pipeline (DE & EN + Code).

Streams instruction datasets, translates them into 16-Bit Multi-Granular Bitstreams,
and prepares the data/instructions/shards/ directory for immediate Step 2 SFT.
"""

import argparse
import json
import os
import sys
from typing import Iterator, Tuple
from tqdm import tqdm

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.bitstream import BitstreamEncoder


def stream_instruction_datasets(max_samples: int = 5000) -> Iterator[Tuple[str, str]]:
    """Streams a balanced mix of German, English, Coding, and Reasoning Instructions."""
    from datasets import load_dataset

    # 1. Multi-turn English & Reasoning Dialogues (UltraChat)
    print("📡 [1/2] Verbinde mit UltraChat (Reasoning & Allgemeindialoge)...")
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        count = 0
        for item in ds:
            messages = item.get("messages", [])
            if len(messages) >= 2:
                user_msg = messages[0].get("content", "").strip()
                assistant_msg = messages[1].get("content", "").strip()
                if 15 < len(user_msg) < 1500 and 15 < len(assistant_msg) < 2500:
                    yield user_msg, assistant_msg
                    count += 1
                    if count >= max_samples // 2:
                        break
        print(f"✅ {count:,} UltraChat-Dialoge empfangen.")
    except Exception as e:
        print(f"⚠️ UltraChat-Streaming: {e}")

    # 2. German & Multi-language Instruction Datasets (e.g. OpenOrca / Aya / Alpaca-DE)
    print("📡 [2/2] Verbinde mit mehrsprachigen & Coding-Instruktionen...")
    try:
        ds_de = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        count_de = 0
        for item in ds_de:
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            user_text = f"{inst}\n{inp}".strip() if inp else inst
            if len(user_text) > 10 and len(out) > 10:
                yield user_text, out
                count_de += 1
                if count_de >= max_samples // 2:
                    break
        print(f"✅ {count_de:,} Instruktions-Paare empfangen.")
    except Exception as e:
        print(f"⚠️ Instruktions-Streaming: {e}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Domain Instruction Ingestion")
    parser.add_argument("--vocab_file", type=str, default="/home/benjamin/Bilder/data/vocab_65k.json", help="Pfad zu vocab.json")
    parser.add_argument("--max_samples", type=int, default=6000, help="Anzahl Instruction-Paare")
    parser.add_argument("--shard_size_tokens", type=int, default=500000, help="Tokens pro Instruction-Shard")
    parser.add_argument("--output_dir", type=str, default="/home/benjamin/Bilder/data/instructions", help="Ausgabeverzeichnis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    shards_dir = os.path.join(args.output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    if not os.path.exists(args.vocab_file):
        args.vocab_file = "/home/benjamin/Bilder/vocab.json"

    vocab = MultiGranularVocabulary.load_json(args.vocab_file)
    tokenizer = ViterbiTokenizer(vocab)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16)

    print("=" * 80)
    print("🚀 SCHRITT 2 VORBEREITUNG: INSTRUCTION BITSTREAM GENERATOR")
    print("=" * 80)
    print(f"  - Vokabular: {args.vocab_file} ({vocab.size:,} Tokens)")
    print(f"  - Zielordner: {shards_dir}")

    current_tokens = []
    current_bytes = 0
    shard_idx = 0
    total_dialogues = 0
    total_tokens = 0

    for user_prompt, assistant_response in tqdm(stream_instruction_datasets(max_samples=args.max_samples), total=args.max_samples, desc="Instruction Sharding"):
        formatted_dialogue = f"### Benutzer:\n{user_prompt}\n\n### Assistent:\n{assistant_response}\n\n"
        tokens = tokenizer.encode(formatted_dialogue)
        doc_bytes = len(formatted_dialogue.encode("utf-8"))

        current_tokens.extend(tokens)
        current_bytes += doc_bytes
        total_tokens += len(tokens)
        total_dialogues += 1

        if len(current_tokens) >= args.shard_size_tokens:
            shard_path = os.path.join(shards_dir, f"instruction_shard_{shard_idx:04d}.mgbs")
            encoder.save_to_file(shard_path, current_tokens, raw_byte_count=current_bytes)
            print(f"  💾 Instruction-Shard {shard_idx:04d}: {len(current_tokens):,} Tokens -> {shard_path}")
            shard_idx += 1
            current_tokens = []
            current_bytes = 0

    if current_tokens:
        shard_path = os.path.join(shards_dir, f"instruction_shard_{shard_idx:04d}.mgbs")
        encoder.save_to_file(shard_path, current_tokens, raw_byte_count=current_bytes)
        print(f"  💾 Instruction-Shard {shard_idx:04d}: {len(current_tokens):,} Tokens -> {shard_path}")
        shard_idx += 1

    print("\n" + "=" * 80)
    print("🎉 SCHRITT 2 VORBEREITUNG ERFOLGREICH ABGESCHLOSSEN!")
    print(f"  - Verarbeitete Dialoge: {total_dialogues:,}")
    print(f"  - Erzeugte Shards:      {shard_idx} Dateien in {shards_dir}")
    print(f"  - Gesamt-Tokens:        {total_tokens:,} Multi-Granular Tokens")
    print("=" * 80)


if __name__ == "__main__":
    main()
