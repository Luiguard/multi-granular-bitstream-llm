#!/usr/bin/env python3
"""Quick 16-bit Shard Generator for immediate training stream."""

import os
import sys
from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.bitstream import BitstreamEncoder
from ingest_multilingual_wiki import stream_wikipedia_articles

vocab_file = "/home/benjamin/Bilder/data/vocab_65k.json"
vocab = MultiGranularVocabulary.load_json(vocab_file)
tokenizer = ViterbiTokenizer(vocab)
encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16)

shards_dir = "/home/benjamin/Bilder/data/shards"
os.makedirs(shards_dir, exist_ok=True)

print("Streaming articles and generating clean 16-bit .mgbs shards...")
stream = stream_wikipedia_articles(languages=["de", "en"], max_articles_per_lang=250)
tokens_buf = []
shard_idx = 0
shard_size = 1_000_000  # 1M tokens per shard = 2.5 MB per shard (Optimales I/O-Profil)

for doc in stream:
    t_ids = tokenizer.encode(doc)
    tokens_buf.extend(t_ids)
    if len(tokens_buf) >= shard_size:
        path = os.path.join(shards_dir, f"shard_{shard_idx:04d}.mgbs")
        encoder.save_to_file(path, tokens_buf, raw_byte_count=len(tokens_buf)*3)
        print(f"  Created 16-bit shard {shard_idx}: {len(tokens_buf):,} tokens -> {path}")
        shard_idx += 1
        tokens_buf = []

if tokens_buf:
    path = os.path.join(shards_dir, f"shard_{shard_idx:04d}.mgbs")
    encoder.save_to_file(path, tokens_buf, raw_byte_count=len(tokens_buf)*3)
    print(f"  Created 16-bit shard {shard_idx}: {len(tokens_buf):,} tokens -> {path}")

print(f"✅ Finished generating clean 16-bit shards in {shards_dir}!")
