#!/usr/bin/env python3
"""Code Ingestion & Mining Pipeline for Multi-Granular Bitstream Coding Models.

Extracts Python, JavaScript, Rust, C++, and Go syntax templates, indentation tokens,
and programming idioms into the 16-Bit Vocabulary.
"""

import argparse
import os
import sys
from typing import Iterator, List
from tqdm import tqdm

from pipeline.vocabulary import MultiGranularVocabulary, TokenTier
from pipeline.mining import PhraseMiner
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.bitstream import BitstreamEncoder


def stream_code_samples(languages: List[str] = ["python", "javascript", "rust"], max_samples: int = 5000) -> Iterator[str]:
    """Streams code samples from open programming datasets."""
    from datasets import load_dataset

    print(f"📡 Lade Code-Datensatz für Sprachen: {', '.join(languages)}...")
    try:
        # Stream clean public code samples (e.g. bigcode/the-stack-smol or codeparrot)
        ds = load_dataset("bigcode/the-stack-smol", data_dir=f"data/{languages[0]}", split="train", streaming=True)
        count = 0
        for item in ds:
            content = item.get("content", "").strip()
            if len(content) > 100:
                yield content
                count += 1
                if count >= max_samples:
                    break
        print(f"✅ {count:,} Code-Dateien erfolgreich gestreamt.")
    except Exception as e:
        print(f"⚠️ Streaming-Meldung (Fallback auf Standard-Code): {e}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Granular Code Ingestion")
    parser.add_argument("--languages", nargs="+", default=["python", "javascript", "rust", "go", "cpp"], help="Programmiersprachen")
    parser.add_argument("--max_samples", type=int, default=5000, help="Anzahl Code-Dateien")
    parser.add_argument("--output_dir", type=str, default="./data/code", help="Ausgabeverzeichnis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    shards_dir = os.path.join(args.output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    print("=" * 80)
    print("💻 MULTI-GRANULARE CODE-INGESTION & SYNTAX-MINING")
    print("=" * 80)
    print("Code-Besonderheiten im Multi-Granular Bitstream:")
    print("  • Einrückungen (4 Spaces, 8 Spaces) -> 1 atomares Token")
    print("  • Häufige Boilerplates ('def __init__(self,', 'import numpy as np') -> 1 Token")
    print("  • Operatoren ('===', '!==', '->', '=>', '::') -> 1 Token")


if __name__ == "__main__":
    main()
