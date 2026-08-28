#!/usr/bin/env python3
"""Cluster-Scale Streaming Ingestion and Multi-Granular Bitstream Sharder."""

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


def stream_corpus(languages: List[str], max_articles_per_lang: int) -> Iterator[str]:
    """Streams cleaned Wikipedia and LLM datasets directly."""
    from datasets import load_dataset

    for lang in languages:
        subset = f"20231101.{lang}"
        print(f"📡 Starte Datenstrom für Wikipedia ({lang.upper()})...")
        try:
            ds = load_dataset("wikimedia/wikipedia", subset, split="train", streaming=True)
            count = 0
            for item in ds:
                text = item.get("text", "").strip()
                if len(text) >= 100:
                    yield text
                    count += 1
                    if count >= max_articles_per_lang:
                        break
            print(f"✅ {count:,} Artikel für Sprache '{lang}' empfangen.")
        except Exception as e:
            print(f"⚠️ Warnung bei {lang}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Cluster Data Ingestion & Bitstream Sharder")
    parser.add_argument("--languages", nargs="+", default=["de", "en"], help="Sprachen für das Korpus")
    parser.add_argument("--mining_articles", type=int, default=10000, help="Artikel für Vokabular-Mining")
    parser.add_argument("--total_articles", type=int, default=100000, help="Artikel gesamt für Shards")
    parser.add_argument("--vocab_size", type=int, default=65536, help="Vokabulargröße (65536 für 16 Bit)")
    parser.add_argument("--shard_size_tokens", type=int, default=2000000, help="Tokens pro Shard (2M Tokens = ~4 MB)")
    parser.add_argument("--output_dir", type=str, default="./data", help="Ausgabeverzeichnis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    shards_dir = os.path.join(args.output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    vocab_file = os.path.join(args.output_dir, f"vocab_{args.vocab_size}.json")

    # Step 1: Vocabulary Mining
    print(f"[1] VOKABULAR-MINING: Sammle {args.mining_articles:,} Artikel für Vokabular-Induktion...")
    mining_sample = []
    stream = stream_corpus(args.languages, args.mining_articles // len(args.languages))
    for doc in tqdm(stream, total=args.mining_articles, desc="Mining-Sample"):
        paragraphs = [p.strip() for p in doc.split("\n\n") if len(p.strip()) > 30]
        mining_sample.extend(paragraphs)
        if len(mining_sample) >= 50000:
            break

    miner = PhraseMiner(
        min_ngram_freq=3,
        min_pmi=0.4,
        max_ngram_len=5,
        max_vocab_budget=args.vocab_size,
    )
    vocab, stats = miner.mine_from_corpus(mining_sample)
    vocab.save_json(vocab_file)
    print(f"✅ 16-Bit Vokabular gespeichert in: {vocab_file} ({vocab.size:,} Tokens)")

    # Step 2: Streaming Tokenization & Sharding
    print(f"\n[2] TOKENISIERUNG & SHARDING: Verarbeite {args.total_articles:,} Artikel in .mgbs Shards...")
    tokenizer = ViterbiTokenizer(vocab)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=vocab.required_bits)

    full_stream = stream_corpus(args.languages, args.total_articles // len(args.languages))
    current_tokens = []
    current_bytes = 0
    shard_idx = 0
    total_tokens = 0
    total_bytes = 0
    start_time = time.time()

    for doc in tqdm(full_stream, total=args.total_articles, desc="Ingestion"):
        tokens = tokenizer.encode(doc)
        doc_b = len(doc.encode("utf-8"))

        current_tokens.extend(tokens)
        current_bytes += doc_b
        total_tokens += len(tokens)
        total_bytes += doc_b

        if len(current_tokens) >= args.shard_size_tokens:
            shard_path = os.path.join(shards_dir, f"shard_{shard_idx:05d}.mgbs")
            encoder.save_to_file(shard_path, current_tokens, raw_byte_count=current_bytes)
            shard_idx += 1
            current_tokens = []
            current_bytes = 0

    if current_tokens:
        shard_path = os.path.join(shards_dir, f"shard_{shard_idx:05d}.mgbs")
        encoder.save_to_file(shard_path, current_tokens, raw_byte_count=current_bytes)
        shard_idx += 1

    elapsed = time.time() - start_time
    print(f"\n🎉 Ingestion abgeschlossen: {total_tokens:,} Tokens in {shard_idx} Shards ({total_bytes / (1024*1024):.1f} MB Text in {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
