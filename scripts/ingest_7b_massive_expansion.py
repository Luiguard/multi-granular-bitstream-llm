#!/usr/bin/env python3
"""Massive-Scale Data Ingestion Pipeline for 7B Model (140B Tokens).

Streams from FineWeb-Edu to gather the remaining 100 Billion tokens required
for Chinchilla-optimal 7B parameter training.

Target: 100 Billion 16-Bit multi-granular tokens = ~200 GB on disk.
Hardware requirement: Internet connection + ~200 GB free NVMe.
"""

import os
import sys
import time
import signal
import multiprocessing as mp
from typing import List

from datasets import load_dataset
from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamEncoder
from pipeline.tokenizer import ViterbiTokenizer

STOP_REQUESTED = False
def handle_signal(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠️ Graceful shutdown angefordert, beende nach aktuellem Shard...", flush=True)
signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

def stream_and_shard_dataset(
    output_base: str,
    vocab_path: str,
    max_tokens_per_shard: int = 5_000_000,
) -> int:
    global STOP_REQUESTED

    output_dir = os.path.join(output_base, "fineweb_edu_100BT_shards")
    os.makedirs(output_dir, exist_ok=True)

    existing_shards = len([f for f in os.listdir(output_dir) if f.endswith(".mgbs")])
    shard_count = existing_shards
    total_tokens_written = existing_shards * max_tokens_per_shard

    print(f"\n📥 Starte Download: FineWeb-Edu (100BT)", flush=True)
    if existing_shards > 0:
        print(f"  ⏭️ {existing_shards} Shards existieren bereits (~{total_tokens_written:,} Tokens). Setze fort.", flush=True)

    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-100BT", split="train", streaming=True)
    except Exception as e:
        print(f"  ❌ Download fehlgeschlagen: {e}", flush=True)
        return total_tokens_written

    vocab = MultiGranularVocabulary.load_json(vocab_path)
    tokenizer = ViterbiTokenizer(vocab)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16)

    # Skip records we already processed based on the number of shards
    # Assuming ~500 tokens per document on average, to avoid reprocessing too much
    estimated_docs_processed = (shard_count * max_tokens_per_shard) // 500
    
    current_tokens = []
    start_time = time.time()
    
    # We skip documents iterator
    doc_iter = iter(ds)
    if estimated_docs_processed > 0:
        print(f"  ⏭️ Überspringe ca. {estimated_docs_processed:,} bereits verarbeitete Dokumente...", flush=True)
        try:
            for _ in range(estimated_docs_processed):
                next(doc_iter)
        except StopIteration:
            pass

    print("  ⚙️ Beginne Tokenisierung und Sharding...", flush=True)
    for sample in doc_iter:
        if STOP_REQUESTED:
            break

        text = sample.get("text", "")
        if not text:
            continue

        try:
            tokens = tokenizer.encode(text)
            current_tokens.extend(tokens)
        except Exception:
            continue

        if len(current_tokens) >= max_tokens_per_shard:
            chunk = current_tokens[:max_tokens_per_shard]
            current_tokens = current_tokens[max_tokens_per_shard:]
            
            shard_path = os.path.join(output_dir, f"fineweb_edu_{shard_count:06d}.mgbs")
            encoder.save_to_file(shard_path, chunk, raw_byte_count=len(chunk) * 2)
            total_tokens_written += len(chunk)
            shard_count += 1
            
            elapsed = time.time() - start_time
            tps = (shard_count - existing_shards) * max_tokens_per_shard / max(1.0, elapsed)
            print(f"  [Shards: {shard_count}] Tokens: {total_tokens_written:,} | TPS: {int(tps):,}", flush=True)

    return total_tokens_written

if __name__ == "__main__":
    vocab_file = "/home/benjamin/Bilder/data/vocab_65k.json"
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"
        
    output_dir = "/home/benjamin/Bilder/data/world_knowledge"
    
    stream_and_shard_dataset(output_dir, vocab_file)
