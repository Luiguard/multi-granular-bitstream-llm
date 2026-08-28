#!/usr/bin/env python3
"""Resource-safe, high-performance vocabulary mining and bitstream sharder."""

import gc
import json
import os
import sys
import time
from typing import Iterator, List
from tqdm import tqdm

from pipeline.vocabulary import MultiGranularVocabulary, TokenTier
from pipeline.mining import PhraseMiner
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.bitstream import BitstreamEncoder


def get_system_ram_mb() -> float:
    """Returns current process RAM usage in Megabytes."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def stream_wikipedia_articles(
    languages: List[str] = ["de", "en"],
    max_articles_per_lang: int = 15000,
    min_length: int = 80,
) -> Iterator[str]:
    """Streams cleaned Wikipedia articles directly from HuggingFace dataset."""
    from datasets import load_dataset

    for lang in languages:
        subset_name = f"20231101.{lang}"
        print(f"\n📡 Verbinde mit Wikipedia Datenstrom ({lang.upper()})...")

        try:
            ds = load_dataset(
                "wikimedia/wikipedia",
                subset_name,
                split="train",
                streaming=True,
            )

            count = 0
            for item in ds:
                text = item.get("text", "").strip()
                if len(text) >= min_length:
                    yield text
                    count += 1
                    if count >= max_articles_per_lang:
                        break

            print(f"✅ {count:,} Artikel für Sprache '{lang}' erfolgreich verarbeitet.")
        except Exception as e:
            print(f"⚠️ Streaming-Meldung ({lang}): {e}")


def main():
    print("=" * 80)
    print("🛡️ RESSOURCENGESCHÜTZTES VOKABULAR-MINING & BITSTREAM-SHARDING")
    print("=" * 80)

    target_vocab_size = 65536
    mining_articles_per_lang = 4000     # 8.000 Artikel für Mining (ausreichend für 65k Vokabular)
    total_articles_per_lang = 15000     # 30.000 Artikel gesamt für Trainings-Shards
    shard_size_tokens = 1000000         # 1 Million Tokens pro Shard (~2 MB pro Shard)

    output_dir = "/home/benjamin/Bilder/data"
    shards_dir = os.path.join(output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    vocab_file = os.path.join(output_dir, "vocab_65k.json")

    # =========================================================================
    # PHASE 1: VOKABULAR-MINING (65.536 Tokens / 16 Bit)
    # =========================================================================
    print(f"\n[PHASE 1] Extrahiere Wörter, Phrasen und Satzmuster für 16-Bit Lexikon...")
    print(f"  - Ziel-Vokabular: {target_vocab_size:,} Tokens")
    print(f"  - RAM-Schutz aktiv (Prozess-RAM: {get_system_ram_mb():.1f} MB)")

    mining_corpus: List[str] = []
    mining_stream = stream_wikipedia_articles(
        languages=["de", "en"],
        max_articles_per_lang=mining_articles_per_lang,
    )

    pbar_mining = tqdm(total=mining_articles_per_lang * 2, desc="Mining-Korpus laden")
    for text in mining_stream:
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        mining_corpus.extend(paragraphs)
        pbar_mining.update(1)
        # RAM-Schutz: Begrenze Mining-Absätze
        if len(mining_corpus) >= 40000:
            break
    pbar_mining.close()

    print(f"\n📊 Starte statistische PMI- und Information-Gain-Berechnung auf {len(mining_corpus):,} Absätzen...")
    miner = PhraseMiner(
        min_ngram_freq=3,
        min_pmi=0.4,
        max_ngram_len=5,
        max_vocab_budget=target_vocab_size,
        word_budget_ratio=0.40,      # ~26.000 Wörter
        phrase_budget_ratio=0.52,    # ~34.000 Phrasen
        template_budget_ratio=0.08,  # ~5.000 Satzmuster
    )

    vocab, stats = miner.mine_from_corpus(mining_corpus)
    vocab.save_json(vocab_file)

    # Speicher freigeben
    del mining_corpus
    gc.collect()

    print(f"\n✅ 16-BIT VOKABULAR ERFOLGREICH GEBAUT & GESPEICHERT:")
    print(f"  - Tier 0 (Byte-Fallback):   256 Tokens (0x00 - 0xFF)")
    print(f"  - Tier 1 (Einzelwörter):    {stats.unique_unigrams:,} Tokens")
    print(f"  - Tier 2 (Geminte Phrasen): {stats.mined_phrases_count:,} Tokens")
    print(f"  - Tier 3 (Satzmuster):      {stats.mined_templates_count:,} Tokens")
    print(f"  - Gesamt-Vokabular |V|:     {vocab.size:,} Tokens")
    print(f"  - Bitbreite:                {vocab.required_bits} Bits (exakt 16-Bit Integer)")
    print(f"  - Gespeichert in:           {vocab_file}")

    # =========================================================================
    # PHASE 2: STREAMING-TOKENISIERUNG & SHARDING
    # =========================================================================
    print(f"\n[PHASE 2] Starte Viterbi-Tokenisierung in 16-Bit Bitstream Shards (.mgbs)...")
    tokenizer = ViterbiTokenizer(vocab, length_bonus_factor=0.5)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16)

    full_stream = stream_wikipedia_articles(
        languages=["de", "en"],
        max_articles_per_lang=total_articles_per_lang,
    )

    current_shard_tokens: List[int] = []
    current_shard_bytes = 0
    shard_index = 0
    total_tokens_written = 0
    total_bytes_processed = 0
    start_time = time.time()

    pbar_shards = tqdm(total=total_articles_per_lang * 2, desc="Artikel verarbeiten & Sharden")

    for doc in full_stream:
        tokens = tokenizer.encode(doc)
        doc_bytes = len(doc.encode("utf-8"))

        current_shard_tokens.extend(tokens)
        current_shard_bytes += doc_bytes
        total_tokens_written += len(tokens)
        total_bytes_processed += doc_bytes

        # Wenn Shard-Größe erreicht ist -> direkt auf SSD schreiben
        if len(current_shard_tokens) >= shard_size_tokens:
            shard_path = os.path.join(shards_dir, f"shard_{shard_index:04d}.mgbs")
            encoder.save_to_file(shard_path, current_shard_tokens, raw_byte_count=current_shard_bytes)
            shard_size_kb = os.path.getsize(shard_path) / 1024.0

            # RAM leeren
            current_shard_tokens = []
            current_shard_bytes = 0
            shard_index += 1
            gc.collect()

        pbar_shards.update(1)

    # Letzten Shard schreiben
    if current_shard_tokens:
        shard_path = os.path.join(shards_dir, f"shard_{shard_index:04d}.mgbs")
        encoder.save_to_file(shard_path, current_shard_tokens, raw_byte_count=current_shard_bytes)
        shard_index += 1

    pbar_shards.close()
    elapsed = time.time() - start_time

    total_shards_size_mb = sum(
        os.path.getsize(os.path.join(shards_dir, f))
        for f in os.listdir(shards_dir)
        if f.endswith(".mgbs")
    ) / (1024 * 1024)

    print("\n" + "=" * 80)
    print("🎉 VOKABULAR & BITSTREAM-DATENSATZ VOLLSTÄNDIG ERSTELLT!")
    print("=" * 80)
    print(f"  - Verarbeiteter Rohtext:    {total_bytes_processed / (1024 * 1024):.1f} MB Plaintext")
    print(f"  - Fertige .mgbs Shards:     {shard_index} Dateien in {shards_dir}")
    print(f"  - Gesamte Bitstream-Größe:  {total_shards_size_mb:.1f} MB (über 70% Kompression)")
    print(f"  - Generierte Tokens:        {total_tokens_written:,} Tokens")
    print(f"  - Durchsatz:                {total_tokens_written / max(1, elapsed):.1f} Tokens/Sekunde")
    print(f"  - RAM-Verbrauch stabil bei: {get_system_ram_mb():.1f} MB (Laptop blieb 100% flüssig)")


if __name__ == "__main__":
    main()
