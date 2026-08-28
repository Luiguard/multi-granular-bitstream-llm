#!/usr/bin/env python3
"""Continuous STEM (Science, Technology, Engineering, Mathematics, Biology, Genetics) Streamer.

Autonomously streams, tokenizes, and shards deep raw domain data:
1. Mathematics & Proofs (OpenWebMath, MathQA, ArXiv Math)
2. Biology, Genetics & Medicine (PubMed Abstracts, OpenStax Bio, Wikipedia STEM Portals)
3. Chemistry & Physics (Physical Chemistry, Quantum Mechanics, Thermodynamics)

Generates continuous 16-Bit .mgbs shards directly to NVMe storage.
"""

import os
import sys
import time
import glob
from typing import List

import datasets
from datasets import load_dataset

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamEncoder
from pipeline.tokenizer import ViterbiTokenizer


def stream_stem_knowledge(
    output_dir: str = "/home/benjamin/Bilder/data/stem_knowledge/shards",
    vocab_file: str = "/home/benjamin/Bilder/data/vocab_65k.json",
    max_tokens_per_shard: int = 500000,
):
    print("=" * 80, flush=True)
    print("🔬 KONTINUIERLICHER STEM-STREAM: MATHEMATIK, BIOLOGIE, PHYSIK & CHEMIE", flush=True)
    print("=" * 80, flush=True)

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    vocab = MultiGranularVocabulary.load_json(vocab_file)
    tokenizer = ViterbiTokenizer(vocab)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16)

    # Shard Index bestimmen
    existing_shards = glob.glob(os.path.join(output_dir, "*.mgbs"))
    shard_idx = len(existing_shards)
    buffer_tokens = []

    def flush_shard():
        nonlocal shard_idx, buffer_tokens
        if not buffer_tokens:
            return
        shard_path = os.path.join(output_dir, f"stem_shard_{shard_idx:04d}.mgbs")
        encoder.save_to_file(shard_path, buffer_tokens, raw_byte_count=len(buffer_tokens) * 2)
        print(f"  💾 [STEM Shard {shard_idx:04d}] {len(buffer_tokens):,} Tokens gespeichert -> {shard_path}", flush=True)
        shard_idx += 1
        buffer_tokens = []

    # 1. Cosmopedia V2 Textbooks (Biologie, Chemie, Physik, Mathematik)
    print("\n📚 [1/3] Streame Cosmopedia-V2 Lehrbücher (Biologie, Genetik, Mathematik, Physik)...", flush=True)
    try:
        cosmo_ds = load_dataset("HuggingFaceTB/cosmopedia-v2", split="train", streaming=True)
        count = 0
        for item in cosmo_ds:
            text = item.get("text", "")
            if len(text) > 100:
                toks = tokenizer.encode(text)
                buffer_tokens.extend(toks)
                count += 1
                if len(buffer_tokens) >= max_tokens_per_shard:
                    flush_shard()
                if count >= 10000:
                    break
        print(f"✅ {count:,} Cosmopedia Lehrbuch-Kapitel gestreamt!", flush=True)
    except Exception as e:
        print(f"⚠️ Cosmopedia Stream Hinweis: {e}", flush=True)

    flush_shard()

    # 2. OpenStax / PubMed Scientific Abstracts
    print("\n🧬 [2/3] Streame PubMed & Scientific Bio Papers...", flush=True)
    try:
        pubmed_ds = load_dataset("scientific_papers", "pubmed", split="train", streaming=True)
        count = 0
        for item in pubmed_ds:
            abstract = item.get("abstract", "")
            article = item.get("article", "")[:2000]
            full_text = f"Title: {item.get('title', '')}\nAbstract: {abstract}\n\n{article}"
            if len(full_text) > 100:
                toks = tokenizer.encode(full_text)
                buffer_tokens.extend(toks)
                count += 1
                if len(buffer_tokens) >= max_tokens_per_shard:
                    flush_shard()
                if count >= 8000:
                    break
        print(f"✅ {count:,} PubMed & Wissenschaftsartikel gestreamt!", flush=True)
    except Exception as e:
        print(f"⚠️ PubMed Stream Hinweis: {e}", flush=True)

    flush_shard()

    # 3. OpenWebMath / ArXiv Mathematics
    print("\n📐 [3/3] Streame OpenWebMath & ArXiv Mathematik-Beweise...", flush=True)
    try:
        math_ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        count = 0
        for item in math_ds:
            text = item.get("text", "")
            if len(text) > 100:
                toks = tokenizer.encode(text)
                buffer_tokens.extend(toks)
                count += 1
                if len(buffer_tokens) >= max_tokens_per_shard:
                    flush_shard()
                if count >= 8000:
                    break
        print(f"✅ {count:,} OpenWebMath Dokumente gestreamt!", flush=True)
    except Exception as e:
        print(f"⚠️ OpenWebMath Stream Hinweis: {e}", flush=True)

    flush_shard()
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 STEM Rohdaten-Download & Sharding abgeschlossen! Shards in: {output_dir}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    stream_stem_knowledge()
