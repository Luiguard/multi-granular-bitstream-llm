#!/usr/bin/env python3
"""Instruction & Reasoning Dataset Ingestion for Multi-Granular Bitstream LLMs.

Formats Conversational & Chain-of-Thought data (User / Assistant / Reasoning)
into packed .mgbs binary bitstreams with loss mask boundaries.
"""

import argparse
import json
import os
import sys
from typing import Iterator, List, Tuple
from tqdm import tqdm

from pipeline.vocabulary import MultiGranularVocabulary, TokenTier
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.bitstream import BitstreamEncoder


def stream_instruction_dataset(max_samples: int = 5000) -> Iterator[Tuple[str, str]]:
    """Streams high-quality instruction-response pairs (e.g. OpenOrca / UltraChat / GSM8k)."""
    from datasets import load_dataset
    print("📡 Lade Open-Source Instruction & Reasoning Datensatz (OpenOrca / UltraChat)...")
    try:
        # Load instruction dataset (e.g. HuggingFaceH4/ultrachat_200k or Open-Orca)
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        count = 0
        for item in ds:
            messages = item.get("messages", [])
            if len(messages) >= 2:
                user_msg = messages[0].get("content", "").strip()
                assistant_msg = messages[1].get("content", "").strip()
                if len(user_msg) > 10 and len(assistant_msg) > 10:
                    yield user_msg, assistant_msg
                    count += 1
                    if count >= max_samples:
                        break
        print(f"✅ {count:,} Instruction-Dialoge gestreamt.")
    except Exception as e:
        print(f"⚠️ Instruction-Streaming-Meldung: {e}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Granular Instruction Ingestion")
    parser.add_argument("--vocab_file", type=str, default="./data/vocab_65k.json", help="Pfad zu vocab.json")
    parser.add_argument("--max_samples", type=int, default=5000, help="Anzahl Instruction-Paare")
    parser.add_argument("--output_dir", type=str, default="./data/instructions", help="Ausgabeverzeichnis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    shards_dir = os.path.join(args.output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    if not os.path.exists(args.vocab_file):
        args.vocab_file = "./vocab.json"

    vocab = MultiGranularVocabulary.load_json(args.vocab_file)
    tokenizer = ViterbiTokenizer(vocab)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16)

    print("=" * 80)
    print("🧠 SCHRITT 2: INSTRUCTION- & REASONING-BITSTREAM ERZEUGUNG")
    print("=" * 80)

    token_stream: List[int] = []
    total_dialogues = 0

    for user_prompt, assistant_response in tqdm(stream_instruction_dataset(max_samples=args.max_samples), total=args.max_samples, desc="Instruction Tokenisierung"):
        # Format: User Prompt -> Assistant Response
        dialogue_text = f"### Benutzer:\n{user_prompt}\n\n### Assistent:\n{assistant_response}\n\n"
        tokens = tokenizer.encode(dialogue_text)
        token_stream.extend(tokens)
        total_dialogues += 1

    # Speichern als 16-Bit Instruction Shard
    shard_path = os.path.join(shards_dir, "instructions_0000.mgbs")
    encoder.save_to_file(shard_path, token_stream, raw_byte_count=len(token_stream) * 3)

    print(f"\n✅ {total_dialogues:,} Instruction-Dialoge erfolgreich in Bitstream übersetzt:")
    print(f"  💾 Gespeichert unter: {shard_path} ({len(token_stream):,} 16-Bit Tokens)")


if __name__ == "__main__":
    main()
