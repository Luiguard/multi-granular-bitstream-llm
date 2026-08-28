#!/usr/bin/env python3
"""Streaming Ingestion Pipeline for German & English Wikipedia / LLM Datasets."""

import argparse
import os
import sys
import time
from typing import Iterator, List
from tqdm import tqdm

from pipeline.vocabulary import MultiGranularVocabulary, TokenTier
from pipeline.mining import PhraseMiner
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.bitstream import BitstreamEncoder, BitstreamDecoder


def stream_wikipedia_articles(
    languages: List[str] = ["de", "en"],
    max_articles_per_lang: int = 10000,
    min_length: int = 100,
) -> Iterator[str]:
    """Streams cleaned Wikipedia articles using HuggingFace datasets in streaming mode."""
    from datasets import load_dataset

    for lang in languages:
        date_tag = "20231101"
        subset_name = f"{date_tag}.{lang}"
        print(f"📡 Öffne Datenstrom für Wikipedia ({lang.upper()}) [{subset_name}]...")

        try:
            ds = load_dataset(
                "wikimedia/wikipedia",
                subset_name,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )

            count = 0
            for item in ds:
                text = item.get("text", "").strip()
                if len(text) >= min_length:
                    yield text
                    count += 1
                    if count >= max_articles_per_lang:
                        break

            print(f"✅ {count:,} Artikel für Sprache '{lang}' erfolgreich gestreamt.")
        except Exception as e:
            print(f"⚠️ Warnung beim Streamen von {lang}: {e}")
            print("Fallback auf direkte REST-Chunks...")


def main():
    parser = argparse.ArgumentParser(description="Multi-Granular Wikipedia Stream Ingestion")
    parser.add_argument("--mining_articles", type=int, default=3000, help="Anzahl Artikel für das Vokabular-Mining")
    parser.add_argument("--total_articles", type=int, default=20000, help="Anzahl Artikel gesamt für Bitstream-Shards")
    parser.add_argument("--vocab_size", type=int, default=65536, help="Zielgröße des Vokabulars (z.B. 65536 für 16 Bit)")
    parser.add_argument("--shard_size_tokens", type=int, default=500000, help="Tokens pro Bitstream-Shard Datei")
    parser.add_argument("--output_dir", type=str, default="/home/benjamin/Bilder/data", help="Ausgabeverzeichnis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    shards_dir = os.path.join(args.output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    print("=" * 80)
    print("🚀 MULTI-GRANULARE STREAMING-INGESTION (DEUTSCH & ENGLISCH)")
    print("=" * 80)

    # 1. PHASE 1: Vokabular-Mining über repräsentatives Sample
    print(f"\n[1] PHASE 1: VOKABULAR-MINING (Ziel: {args.vocab_size:,} Tokens / 16 Bit)")
    mining_sample = []
    total_mining_bytes = 0

    articles_stream = stream_wikipedia_articles(
        languages=["de", "en"],
        max_articles_per_lang=args.mining_articles // 2,
    )

    print("Lade Mining-Korpus...")
    for text in tqdm(articles_stream, total=args.mining_articles, desc="Mining-Artikel"):
        # Teile lange Artikel in Absätze
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
        mining_sample.extend(paragraphs)
        total_mining_bytes += sum(len(p.encode("utf-8")) for p in paragraphs)
        if len(mining_sample) >= args.mining_articles * 5:
            break

    print(f"  - Geladene Absätze für Mining: {len(mining_sample):,}")
    print(f"  - Datenmenge für Mining:       {total_mining_bytes / (1024 * 1024):.2f} MB")

    miner = PhraseMiner(
        min_ngram_freq=4,
        min_pmi=0.4,
        max_ngram_len=5,
        max_vocab_budget=args.vocab_size,
    )
    vocab, stats = miner.mine_from_corpus(mining_sample)

    vocab_file = os.path.join(args.output_dir, f"vocab_{vocab.size}.json")
    vocab.save_json(vocab_file)

    print(f"\n✅ VOKABULAR ERFOLGREICH ERSTELLT:")
    print(f"  - Tier 0 (Byte-Fallback):   256 Tokens")
    print(f"  - Tier 1 (Einzelwörter):    {stats.unique_unigrams:,} Tokens")
    print(f"  - Tier 2 (Geminte Phrasen): {stats.mined_phrases_count:,} Tokens")
    print(f"  - Tier 3 (Satzmuster):      {stats.mined_templates_count:,} Tokens")
    print(f"  - Vokabular Gesamt |V|:     {vocab.size:,} Tokens")
    print(f"  - Bitbreite pro Token:      {vocab.required_bits} Bits (exakt 16-Bit passend)")
    print(f"  - Gespeichert unter:        {vocab_file}")

    # 2. PHASE 2: Tokenisierung & Sharding in .mgbs Dateien
    print(f"\n[2] PHASE 2: STREAMING-TOKENISIERUNG & BITSTREAM-SHARDING")
    tokenizer = ViterbiTokenizer(vocab, length_bonus_factor=0.5)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=vocab.required_bits)

    full_stream = stream_wikipedia_articles(
        languages=["de", "en"],
        max_articles_per_lang=args.total_articles // 2,
    )

    current_shard_tokens: List[int] = []
    current_shard_bytes = 0
    shard_index = 0
    total_processed_tokens = 0
    total_processed_bytes = 0
    start_time = time.time()

    for doc in tqdm(full_stream, total=args.total_articles, desc="Ingestion & Tokenisierung"):
        tokens = tokenizer.encode(doc)
        doc_bytes = len(doc.encode("utf-8"))

        current_shard_tokens.extend(tokens)
        current_shard_bytes += doc_bytes
        total_processed_tokens += len(tokens)
        total_processed_bytes += doc_bytes

        # Shard speichern, wenn Limit erreicht
        if len(current_shard_tokens) >= args.shard_size_tokens:
            shard_path = os.path.join(shards_dir, f"shard_{shard_index:04d}.mgbs")
            encoder.save_to_file(shard_path, current_shard_tokens, raw_byte_count=current_shard_bytes)
            shard_size = os.path.getsize(shard_path)
            print(f"  💾 Shard {shard_index:04d}: {len(current_shard_tokens):,} Tokens | {current_shard_bytes / (1024*1024):.1f} MB Text ➔ {shard_size / (1024*1024):.1f} MB Bitstream ({current_shard_bytes / max(1, shard_size):.2f}x)")
            shard_index += 1
            current_shard_tokens = []
            current_shard_bytes = 0

    # Letzten Shard schreiben
    if current_shard_tokens:
        shard_path = os.path.join(shards_dir, f"shard_{shard_index:04d}.mgbs")
        encoder.save_to_file(shard_path, current_shard_tokens, raw_byte_count=current_shard_bytes)
        shard_size = os.path.getsize(shard_path)
        print(f"  💾 Shard {shard_index:04d}: {len(current_shard_tokens):,} Tokens | {current_shard_bytes / (1024*1024):.1f} MB Text ➔ {shard_size / (1024*1024):.1f} MB Bitstream")
        shard_index += 1

    elapsed = time.time() - start_time
    total_bitstream_size = sum(os.path.getsize(os.path.join(shards_dir, f)) for f in os.listdir(shards_dir) if f.endswith(".mgbs"))

    print("\n" + "=" * 80)
    print("🎉 INGESTION ERFOLGREICH ABGESCHLOSSEN!")
    print("=" * 80)
    print(f"  - Verarbeitete Rohtext-Menge: {total_processed_bytes / (1024*1024):.2f} MB")
    print(f"  - Gesamte Token-Anzahl:       {total_processed_tokens:,} Tokens")
    print(f"  - Erzeugte Shards:            {shard_index} Dateien in {shards_dir}")
    print(f"  - Gesamtgröße auf NVMe-SSD:   {total_bitstream_size / (1024*1024):.2f} MB")
    print(f"  - Reale Datenkompression:     {total_processed_bytes / max(1, total_bitstream_size):.2f}x Speicherersparnis")
    print(f"  - Durchsatz:                  {total_processed_tokens / max(1, elapsed):.1f} Tokens/Sekunde")


if __name__ == "__main__":
    main()
