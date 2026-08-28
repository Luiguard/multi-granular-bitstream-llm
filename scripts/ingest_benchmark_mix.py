#!/usr/bin/env python3
"""Benchmark-Grade Multi-Source Ingestion Pipeline.

Combines FineWeb-Edu (Educational Web), The Stack (Code), Wikipedia (Knowledge),
and Synthetic Reasoning into High-Density Multi-Granular Bitstream Shards (.mgbs).
"""

import argparse
import os
import sys
import time
from typing import Iterator, List
from tqdm import tqdm

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.mining import PhraseMiner
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.bitstream import BitstreamEncoder


def stream_fineweb_edu(max_samples: int = 5000) -> Iterator[str]:
    """Streams top-tier educational web text from FineWeb-Edu."""
    from datasets import load_dataset
    print(f"📡 [1/3] Verbinde mit FineWeb-Edu (Lehrbuch- & Erklärqualität)...")
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        count = 0
        for item in ds:
            text = item.get("text", "").strip()
            if len(text) > 150:
                yield text
                count += 1
                if count >= max_samples:
                    break
        print(f"✅ {count:,} FineWeb-Edu Dokumente empfangen.")
    except Exception as e:
        print(f"⚠️ FineWeb-Edu Streaming-Meldung: {e}")


def stream_the_stack_code(languages: List[str] = ["python", "rust", "javascript"], max_samples: int = 5000) -> Iterator[str]:
    """Streams clean programming code from The Stack."""
    from datasets import load_dataset
    print(f"📡 [2/3] Verbinde mit The Stack (Code & Logik für {', '.join(languages)})...")
    try:
        ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
        count = 0
        for item in ds:
            code = item.get("content", "").strip()
            if len(code) > 80:
                yield code
                count += 1
                if count >= max_samples:
                    break
        print(f"✅ {count:,} Code-Dateien empfangen.")
    except Exception as e:
        print(f"⚠️ Code-Streaming-Meldung: {e}")


def stream_wikipedia_facts(languages: List[str] = ["de", "en"], max_samples: int = 5000) -> Iterator[str]:
    """Streams encyclopedic facts from German & English Wikipedia."""
    from datasets import load_dataset
    print(f"📡 [3/3] Verbinde mit Wikipedia (Faktenwissen DE/EN)...")
    for lang in languages:
        try:
            ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
            count = 0
            for item in ds:
                text = item.get("text", "").strip()
                if len(text) > 100:
                    yield text
                    count += 1
                    if count >= max_samples // len(languages):
                        break
            print(f"✅ {count:,} Wikipedia-Artikel ({lang}) empfangen.")
        except Exception as e:
            print(f"⚠️ Wikipedia-Streaming-Meldung ({lang}): {e}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark-Grade Multi-Source Ingestion")
    parser.add_argument("--samples_per_source", type=int, default=3000, help="Anzahl Samples pro Datenquelle")
    parser.add_argument("--vocab_size", type=int, default=65536, help="Zielgröße des Vokabulars (16 Bit)")
    parser.add_argument("--shard_size_tokens", type=int, default=1000000, help="Tokens pro Shard (1M Tokens = ~2 MB)")
    parser.add_argument("--output_dir", type=str, default="./data/benchmark_mix", help="Ausgabeverzeichnis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    shards_dir = os.path.join(args.output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    vocab_file = os.path.join(args.output_dir, f"vocab_{args.vocab_size}.json")

    print("=" * 80)
    print("🌟 MULTI-SOURCE BENCHMARK INGESTION (FINEWEB + CODE + WIKI)")
    print("=" * 80)

    # 1. PHASE: Vokabular-Mining über Multi-Source Sample
    print(f"\n[PHASE 1] Erstelle balanciertes 16-Bit Vokabular über Text, Code & Fakten...")
    mining_sample = []

    for text in stream_fineweb_edu(max_samples=args.samples_per_source // 3):
        mining_sample.append(text[:2000])

    for code in stream_the_stack_code(max_samples=args.samples_per_source // 3):
        mining_sample.append(code[:2000])

    for wiki in stream_wikipedia_facts(max_samples=args.samples_per_source // 3):
        mining_sample.append(wiki[:2000])

    print(f"\n📊 Berechne PMI und Induktion auf {len(mining_sample):,} heterogenen Dokumenten...")
    miner = PhraseMiner(
        min_ngram_freq=3,
        min_pmi=0.4,
        max_ngram_len=5,
        max_vocab_budget=args.vocab_size,
        word_budget_ratio=0.35,      # 35% Wörter
        phrase_budget_ratio=0.55,    # 55% Phrasen & Code-Blöcke
        template_budget_ratio=0.10,  # 10% Syntaktische Templates
    )

    vocab, stats = miner.mine_from_corpus(mining_sample)
    vocab.save_json(vocab_file)
    print(f"✅ Vokabular gespeichert: {vocab_file} ({vocab.size:,} Tokens, {vocab.required_bits} Bits)")

    # 2. PHASE: Vollständiges Sharding in .mgbs Shards
    print(f"\n[PHASE 2] Tokenisiere alle Quellen in .mgbs Bitstream Shards...")
    tokenizer = ViterbiTokenizer(vocab)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16)

    def full_generator():
        yield from stream_fineweb_edu(max_samples=args.samples_per_source)
        yield from stream_the_stack_code(max_samples=args.samples_per_source)
        yield from stream_wikipedia_facts(max_samples=args.samples_per_source)

    current_tokens = []
    current_bytes = 0
    shard_idx = 0
    total_tokens = 0
    total_bytes = 0
    start_time = time.time()

    for doc in tqdm(full_generator(), total=args.samples_per_source * 3, desc="Benchmark Sharding"):
        tokens = tokenizer.encode(doc)
        doc_b = len(doc.encode("utf-8"))

        current_tokens.extend(tokens)
        current_bytes += doc_b
        total_tokens += len(tokens)
        total_bytes += doc_b

        if len(current_tokens) >= args.shard_size_tokens:
            shard_path = os.path.join(shards_dir, f"shard_{shard_idx:04d}.mgbs")
            encoder.save_to_file(shard_path, current_tokens, raw_byte_count=current_bytes)
            shard_idx += 1
            current_tokens = []
            current_bytes = 0

    if current_tokens:
        shard_path = os.path.join(shards_dir, f"shard_{shard_idx:04d}.mgbs")
        encoder.save_to_file(shard_path, current_tokens, raw_byte_count=current_bytes)
        shard_idx += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 BENCHMARK-DATENSATZ ERFOLGREICH ERSTELLT!")
    print(f"  - Gesamte Token:  {total_tokens:,} Multi-Granular Tokens")
    print(f"  - Erzeugte Shards: {shard_idx} Dateien in {shards_dir}")
    print(f"  - Verarbeitet in:  {elapsed:.1f} Sekunden")
    print("=" * 80)


if __name__ == "__main__":
    main()
