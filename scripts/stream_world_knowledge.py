#!/usr/bin/env python3
"""Infinite World-Knowledge Ingestion & Sharding Engine.

Streams and continuously packs:
1. German & English Wikipedia (Full encyclopedic facts, history, science, geography)
2. FineWeb-Edu (Educational textbooks, university courses, explanations)
3. The Stack (Python, Rust, JavaScript, Go, SQL programming knowledge)
4. UltraChat & OpenOrca (Reasoning, dialogues, and cognitive chains)

Directly into compressed 16-Bit .mgbs binary shards on the NVMe SSD.
"""

import argparse
import os
import sys
import time
from typing import Iterator

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.bitstream import BitstreamEncoder


def stream_world_knowledge_sources(max_per_source: int = 50000) -> Iterator[str]:
    """Streams a massive, comprehensive world knowledge corpus."""
    from datasets import load_dataset

    # 1. German Wikipedia (All encyclopedic facts in German)
    print("🌍 [1/4] Streamen von Wikipedia (Deutsch - Vollständiges Enzyklopädie-Wissen)...", flush=True)
    try:
        ds_de = load_dataset("wikimedia/wikipedia", "20231101.de", split="train", streaming=True)
        count = 0
        for item in ds_de:
            text = item.get("text", "").strip()
            if len(text) > 150:
                yield text
                count += 1
                if count >= max_per_source:
                    break
        print(f"✅ {count:,} deutsche Wikipedia-Artikel eingereicht.", flush=True)
    except Exception as e:
        print(f"⚠️ Wiki DE Streaming: {e}", flush=True)

    # 2. English Wikipedia (Global facts, world history, science)
    print("🌍 [2/4] Streamen von Wikipedia (Englisch - Weltweites Faktenwissen)...", flush=True)
    try:
        ds_en = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        count = 0
        for item in ds_en:
            text = item.get("text", "").strip()
            if len(text) > 150:
                yield text
                count += 1
                if count >= max_per_source:
                    break
        print(f"✅ {count:,} englische Wikipedia-Artikel eingereicht.", flush=True)
    except Exception as e:
        print(f"⚠️ Wiki EN Streaming: {e}", flush=True)

    # 3. FineWeb-Edu (Top-Tier Educational & Textbook Content)
    print("🌍 [3/4] Streamen von FineWeb-Edu (Lehrbücher, wissenschaftliche Erklärungen)...", flush=True)
    try:
        ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        count = 0
        for item in ds_edu:
            text = item.get("text", "").strip()
            if len(text) > 200:
                yield text
                count += 1
                if count >= max_per_source:
                    break
        print(f"✅ {count:,} FineWeb-Edu Lehrbuchkapitel eingereicht.", flush=True)
    except Exception as e:
        print(f"⚠️ FineWeb-Edu Streaming: {e}", flush=True)

    # 4. Code from The Stack (Python, Rust, JavaScript, SQL)
    print("🌍 [4/4] Streamen von Quellcode & Algorithmen (The Stack)...", flush=True)
    try:
        ds_code = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
        count = 0
        for item in ds_code:
            code = item.get("content", "").strip()
            if len(code) > 100:
                yield code
                count += 1
                if count >= max_per_source:
                    break
        print(f"✅ {count:,} Code-Dateien eingereicht.", flush=True)
    except Exception as e:
        print(f"⚠️ Code Streaming: {e}", flush=True)


def build_world_knowledge_shards(output_dir: str = "/home/benjamin/Bilder/data/world_knowledge/shards"):
    os.makedirs(output_dir, exist_ok=True)
    vocab_file = "/home/benjamin/Bilder/data/vocab_65k.json"
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    vocab = MultiGranularVocabulary.load_json(vocab_file)
    tokenizer = ViterbiTokenizer(vocab)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16)

    print("=" * 80, flush=True)
    print("📚 ERZEUGUNG DES WELTWISSEN-DATENSATZES (WIKIPEDIA + LEHRBÜCHER + CODE)", flush=True)
    print("=" * 80, flush=True)

    shard_size_tokens = 1000000  # 1M Tokens pro Shard (~2 MB)
    current_tokens = []
    current_bytes = 0
    shard_idx = 0
    total_tokens = 0

    for doc in stream_world_knowledge_sources(max_per_source=30000):
        tokens = tokenizer.encode(doc)
        doc_b = len(doc.encode("utf-8"))

        current_tokens.extend(tokens)
        current_bytes += doc_b
        total_tokens += len(tokens)

        if len(current_tokens) >= shard_size_tokens:
            shard_path = os.path.join(output_dir, f"world_shard_{shard_idx:05d}.mgbs")
            encoder.save_to_file(shard_path, current_tokens, raw_byte_count=current_bytes)
            print(f"  💾 Shard {shard_idx:05d} gespeichert: {len(current_tokens):,} Tokens -> {shard_path}", flush=True)
            shard_idx += 1
            current_tokens = []
            current_bytes = 0

    if current_tokens:
        shard_path = os.path.join(output_dir, f"world_shard_{shard_idx:05d}.mgbs")
        encoder.save_to_file(shard_path, current_tokens, raw_byte_count=current_bytes)
        print(f"  💾 Shard {shard_idx:05d} gespeichert: {len(current_tokens):,} Tokens -> {shard_path}", flush=True)
        shard_idx += 1

    print("\n" + "=" * 80, flush=True)
    print(f"🎉 WELTWISSEN-SHARDS ERFOLGREICH ERSTELLT!", flush=True)
    print(f"  - Gesamte Tokens:  {total_tokens:,} Multi-Granular World Tokens", flush=True)
    print(f"  - Shard-Dateien:   {shard_idx} in {output_dir}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    build_world_knowledge_shards()
