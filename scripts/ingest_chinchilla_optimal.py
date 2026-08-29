#!/usr/bin/env python3
"""Massive-Scale Data Ingestion Pipeline: 7+ Billion Tokens for Chinchilla-Optimal Training.

Streams from the largest freely available pretraining datasets on HuggingFace:
1. Full German Wikipedia (20231101.de) - complete
2. Full English Wikipedia (20231101.en) - complete  
3. FineWeb-Edu (sample-10BT) - massive educational web text
4. SlimPajama (cerebras) - 627B token pretraining mix (we take a large slice)
5. The Stack (Python, JavaScript, TypeScript, Rust, Go, Java, C++) - code
6. Cosmopedia-V2 - synthetic textbooks
7. OpenWebMath - mathematical proofs
8. PubMed Central - biomedical research

Target: ~7 Billion 16-Bit multi-granular tokens = ~14 GB on disk.
Hardware requirement: Internet connection + 877 GB free NVMe.
"""

import os
import sys
import time
import signal
import traceback
from typing import List

from datasets import load_dataset

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamEncoder
from pipeline.tokenizer import ViterbiTokenizer


# Global graceful shutdown
STOP_REQUESTED = False
def handle_signal(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠️ Graceful shutdown angefordert, beende nach aktuellem Shard...", flush=True)
signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


DATASETS = [
    # (name, hf_path, hf_config, split, text_field, max_samples, shard_prefix, description)
    # The New Ultra-Premium Logic & Reasoning Datasets
    ("ArXiv Papers (Math/Physics/CS)", "togethercomputer/RedPajama-Data-1T", "arxiv", "train", "text", 500_000, "arxiv", "Wissenschaftliche Paper (LaTeX)"),
    ("StackExchange (High-Upvote QA)", "togethercomputer/RedPajama-Data-1T", "stackexchange", "train", "text", 2_000_000, "stackexchange", "Gefiltertes QA-Wissen"),
    ("Project Gutenberg (Literatur)", "togethercomputer/RedPajama-Data-1T", "book", "train", "text", 100_000, "gutenberg", "Hochwertige Bücher & Literatur"),
    ("PhilPapers (Philosophie)", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 100_000, "phil", "Tiefe logische Argumentationen"), # using a fallback for now, as specific SEP is rare

    # Original High-Quality Datasets
    ("Wikipedia DE (Vollständig)", "wikimedia/wikipedia", "20231101.de", "train", "text", 2_600_000, "wiki_de", "Komplette deutsche Wikipedia"),
    ("Wikipedia EN (Vollständig)", "wikimedia/wikipedia", "20231101.en", "train", "text", 6_800_000, "wiki_en", "Komplette englische Wikipedia"),
    ("FineWeb-Edu (Bildungswebtext)", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 5_000_000, "fineweb", "Gefilterte, bildungsrelevante Webtexte"),
    ("Cosmopedia-V2 (STEM Lehrbücher)", "HuggingFaceTB/cosmopedia-v2", None, "train", "text", 1_000_000, "cosmo", "Synthetische Lehrbuch-Texte"),
    ("OpenWebMath (Mathematik)", "open-web-math/open-web-math", None, "train", "text", 500_000, "owmath", "Mathematische Beweise und Formeln"),
    ("PubMed (Biomedizin)", "scientific_papers", "pubmed", "train", "article", 300_000, "pubmed", "Biomedizinische Fachartikel"),
    ("The Stack Python", "bigcode/the-stack-smol", "data/python", "train", "content", 500_000, "code_py", "Python-Quellcode"),
    ("The Stack JavaScript", "bigcode/the-stack-smol", "data/javascript", "train", "content", 300_000, "code_js", "JavaScript-Quellcode"),
    ("The Stack TypeScript", "bigcode/the-stack-smol", "data/typescript", "train", "content", 200_000, "code_ts", "TypeScript-Quellcode"),
    ("UltraChat (Dialog SFT)", "HuggingFaceH4/ultrachat_200k", None, "train_sft", "messages", 200_000, "ultrachat", "Dialogdaten für Instruction Following"),
]


def stream_and_shard_dataset(
    name: str,
    hf_path: str,
    hf_config: str,
    split: str,
    text_field: str,
    max_samples: int,
    shard_prefix: str,
    description: str,
    output_base: str,
    tokenizer: ViterbiTokenizer,
    encoder: BitstreamEncoder,
    max_tokens_per_shard: int = 1_000_000,
) -> int:
    """Streams a single dataset and writes shards. Returns total tokens written."""
    global STOP_REQUESTED

    output_dir = os.path.join(output_base, f"{shard_prefix}_shards")
    os.makedirs(output_dir, exist_ok=True)

    # Check how many shards already exist (for resume)
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".mgbs")])
    if existing > 0:
        estimated_existing_tokens = existing * max_tokens_per_shard
        print(f"  ⏭️ {existing} Shards existieren bereits (~{estimated_existing_tokens:,} Tokens). Überspringe.", flush=True)
        return estimated_existing_tokens

    print(f"\n📥 Starte Download: {name}", flush=True)
    print(f"   Quelle: {hf_path} ({hf_config or 'default'}) → max {max_samples:,} Samples", flush=True)

    try:
        if hf_config:
            ds = load_dataset(hf_path, hf_config, split=split, streaming=True)
        else:
            ds = load_dataset(hf_path, split=split, streaming=True)
    except Exception as e:
        print(f"  ❌ Download fehlgeschlagen: {e}", flush=True)
        return 0

    shard_idx = 0
    buffer_tokens = []
    total_tokens = 0
    sample_count = 0
    start_time = time.time()

    for item in ds:
        if STOP_REQUESTED:
            break
        if sample_count >= max_samples:
            break

        # Extract text
        text = ""
        if text_field == "messages":
            # UltraChat format: list of messages
            msgs = item.get("messages", [])
            parts = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "user":
                    parts.append(f"### Benutzer:\n{content}")
                else:
                    parts.append(f"### Assistent:\n{content}")
            text = "\n\n".join(parts) + "\n\n"
        else:
            text = item.get(text_field, "")

        if not text or len(text) < 50:
            continue

        # Truncate very long documents to prevent memory issues
        if len(text) > 50000:
            text = text[:50000]

        try:
            toks = tokenizer.encode(text)
            buffer_tokens.extend(toks)
        except Exception:
            continue

        sample_count += 1

        # Flush shard when buffer is full
        if len(buffer_tokens) >= max_tokens_per_shard:
            import shutil
            total, used, free = shutil.disk_usage(output_base)
            if free < 20 * 1024**3:  # 20 GB free space safety margin
                print(f"\n🚨 KRITISCH: NVMe Speicher fast voll! Nur noch {free / 1024**3:.2f} GB frei. Ingestion wird sicher beendet.", flush=True)
                STOP_REQUESTED = True
                break

            shard_path = os.path.join(output_dir, f"{shard_prefix}_shard_{shard_idx:05d}.mgbs")
            encoder.save_to_file(shard_path, buffer_tokens[:max_tokens_per_shard], raw_byte_count=max_tokens_per_shard * 2)
            tokens_written = len(buffer_tokens[:max_tokens_per_shard])
            total_tokens += tokens_written
            elapsed = time.time() - start_time
            tps = total_tokens / max(1, elapsed)
            print(f"  💾 [{shard_prefix} Shard {shard_idx:05d}] {tokens_written:,} Tokens | Gesamt: {total_tokens:,} | {sample_count:,} Samples | {tps:.0f} tok/s", flush=True)
            buffer_tokens = buffer_tokens[max_tokens_per_shard:]
            shard_idx += 1

    # Flush remaining
    if buffer_tokens and not STOP_REQUESTED:
        shard_path = os.path.join(output_dir, f"{shard_prefix}_shard_{shard_idx:05d}.mgbs")
        encoder.save_to_file(shard_path, buffer_tokens, raw_byte_count=len(buffer_tokens) * 2)
        total_tokens += len(buffer_tokens)
        shard_idx += 1

    elapsed = time.time() - start_time
    print(f"  ✅ {name}: {shard_idx} Shards, {total_tokens:,} Tokens, {sample_count:,} Samples in {elapsed:.0f}s", flush=True)
    return total_tokens


def main():
    global STOP_REQUESTED

    print("=" * 90, flush=True)
    print("🌍 MASSIVE-SCALE DATEN-INGESTION: CHINCHILLA-OPTIMALE 7+ MILLIARDEN TOKENS", flush=True)
    print("=" * 90, flush=True)
    print(f"  Ziel: ~7.000.000.000 Tokens (Chinchilla-Optimal für 350M aktive Parameter)", flush=True)
    print(f"  Speicher verfügbar: ~877 GB NVMe (benötigt: ~14 GB)", flush=True)
    print(f"  Datensätze: {len(DATASETS)} Quellen", flush=True)
    print("=" * 90, flush=True)

    vocab_file = "/home/benjamin/Bilder/data/vocab_65k.json"
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    vocab = MultiGranularVocabulary.load_json(vocab_file)
    tokenizer = ViterbiTokenizer(vocab)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16)

    output_base = "/home/benjamin/Bilder/data/chinchilla_corpus"
    os.makedirs(output_base, exist_ok=True)

    grand_total = 0

    for name, hf_path, hf_config, split, text_field, max_samples, shard_prefix, description in DATASETS:
        if STOP_REQUESTED:
            break

        try:
            tokens = stream_and_shard_dataset(
                name=name,
                hf_path=hf_path,
                hf_config=hf_config,
                split=split,
                text_field=text_field,
                max_samples=max_samples,
                shard_prefix=shard_prefix,
                description=description,
                output_base=output_base,
                tokenizer=tokenizer,
                encoder=encoder,
            )
            grand_total += tokens
        except Exception as e:
            print(f"  ❌ Fehler bei {name}: {e}", flush=True)
            traceback.print_exc()
            continue

        print(f"\n  📊 Laufende Gesamtsumme: {grand_total:,} Tokens ({grand_total/1_000_000_000:.2f} Milliarden)", flush=True)

    print("\n" + "=" * 90, flush=True)
    print(f"🎉 INGESTION ABGESCHLOSSEN!", flush=True)
    print(f"   Gesamte Tokens: {grand_total:,} ({grand_total/1_000_000_000:.2f} Milliarden)", flush=True)
    print(f"   Speicherort: {output_base}", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
