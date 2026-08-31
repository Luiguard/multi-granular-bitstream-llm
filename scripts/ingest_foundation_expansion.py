#!/usr/bin/env python3
"""Massive Foundation Bitstream & Syntax Expansion Ingestion Pipeline.

Target: 100+ Shards (50M+ 16-Bit Multi-Granular Tokens in data/shards/)
Sources:
1. Wikipedia (German - 20231101.de) - High quality encyclopedic German syntax
2. Wikipedia (English - 20231101.en) - High quality encyclopedic English syntax
3. Curated German Morphology & Syntactic Structures (Conjugations, Declensions, Duden Rules)

Streams directly into 16-Bit .mgbs bitstream shards.
Automatically appends to data/shards/ without overwriting existing shards.
"""

import os
import sys
import time
import glob
import signal
from typing import List, Dict, Any

from datasets import load_dataset
from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamEncoder
from pipeline.tokenizer import ViterbiTokenizer

STOP_REQUESTED = False

def handle_signal(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠️ Graceful Shutdown angefordert, beende nach aktuellem Shard...", flush=True)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

FOUNDATION_SYNTAX_STANDARDS = [
    """# Deutsche Grammatik, Morphologie & Syntaktische Grundregeln:

## 1. Satzbau & Wortstellung (V2-Stellung & Nebensatzstrukturen)
- Hauptsatz: Das finite Verb steht stets an zweiter Stelle (Verb-Zweit-Stellung / V2). Beispiel: "Der Viterbi-Algorithmus optimiert die Token-Sequenz."
- Nebensatz: Das finite Verb steht an letzter Stelle (Verb-Letzt-Stellung / VL). Beispiel: "...weil das Modell den Bitstream verlustfrei rekonstruiert."
- Inversion: Tritt ein Adverbiale an die erste Position, folgt unmittelbar das finite Verb vor dem Subjekt. Beispiel: "Heute trainieren wir das 7B-Modell."

## 2. Kasus- und Rektionslehre (Nominativ, Genitiv, Dativ, Akkusativ)
- Präpositionen mit Genitiv: während, wegen, trotz, anlässlich, aufgrund (+ Genitiv).
- Präpositionen mit Dativ: aus, bei, mit, nach, seit, von, zu, gegenüber (+ Dativ).
- Präpositionen mit Akkusativ: durch, für, gegen, ohne, um (+ Akkusativ).
- Wechselpräpositionen: an, auf, hinter, in, neben, über, unter, vor, zwischen (Dativ bei Ort/Wo? / Akkusativ bei Richtung/Wohin?).

## 3. Kongruenz & Substantivierung
- Kongruenz: Subjekt und Prädikat stimmen in Person und Numerus überein.
- Substantivierung: Verben und Adjektive nach Signalwörtern (das, ein, beim, zum, viel, wenig) werden großgeschrieben (z. B. "beim Lernen", "etwas Neues").""",

    """# English Syntax, Grammar & Morphological Structures:

## 1. Sentence Architecture & Word Order (SVO)
- Canonical Structure: Standard declarative clauses strictly follow Subject-Verb-Object (SVO) order.
- Adverbial Placement: Adverbs of frequency (always, often, rarely) precede the main verb but follow auxiliary/be verbs.
- Subordinate Clauses: Dependent clauses introduced by subordinating conjunctions (although, because, whereas) retain standard SVO internal ordering.

## 2. Tense, Aspect & Modality
- Simple vs. Progressive: Habitual/factual states use simple aspect; ongoing temporary actions require progressive aspect (be + present participle).
- Perfective Aspect: Expresses prior completion relevant to the reference time (have + past participle).
- Modal Auxiliaries: Followed strictly by bare infinitives without 'to' (must, should, can, could, might, will).

## 3. Punctuation & Orthographic Hierarchy
- Coordinate Clauses: Joined by coordinating conjunctions (FANBOYS) preceded by a comma when connecting independent clauses.
- Relative Clauses: Restrictive clauses require no commas ('that'); non-restrictive relative clauses are parenthetical and enclosed by commas ('which')."""
]


def run_foundation_expansion(
    output_dir: str = "/home/benjamin/Bilder/data/shards",
    vocab_file: str = "/home/benjamin/Bilder/data/vocab_65k.json",
    max_tokens_per_shard: int = 500_000,
    target_shards: int = 100,
) -> int:
    global STOP_REQUESTED

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    print("=" * 80, flush=True)
    print("🏛️ MASSIVE FOUNDATION BITSTREAM & SYNTAX EXPANSION (7B PRE-TRAINING)", flush=True)
    print(f"📁 Ziel-Verzeichnis: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    existing_files = sorted(glob.glob(os.path.join(output_dir, "shard_*.mgbs")))
    shard_count = len(existing_files)
    print(f"  ⏭️ {shard_count} bestehende Foundation-Shards gefunden. Starte ab Index {shard_count:04d}...", flush=True)

    vocab = MultiGranularVocabulary.load_json(vocab_file)
    tokenizer = ViterbiTokenizer(vocab)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16)

    buffer_tokens: List[int] = []
    total_tokens_written = shard_count * max_tokens_per_shard
    start_time = time.time()

    def flush_shard():
        nonlocal shard_count, buffer_tokens, total_tokens_written
        if not buffer_tokens:
            return
        shard_path = os.path.join(output_dir, f"shard_{shard_count:04d}.mgbs")
        encoder.save_to_file(shard_path, buffer_tokens, raw_byte_count=len(buffer_tokens) * 2)
        total_tokens_written += len(buffer_tokens)
        shard_count += 1
        elapsed = time.time() - start_time
        print(f"  💾 [FOUNDATION Shard {shard_count:04d}] {len(buffer_tokens):,} Tokens | Gesamt: {total_tokens_written:,} Tokens -> {os.path.basename(shard_path)}", flush=True)
        buffer_tokens = []

    # 1. Ingest Syntax Standards
    print("\n📚 [Quelle 1/3] Tokenisiere formale Grammatik- & Syntaxregeln (DE & EN)...", flush=True)
    for doc in FOUNDATION_SYNTAX_STANDARDS:
        formatted = f"### Linguistische Grundsyntax:\n{doc.strip()}\n\n"
        buffer_tokens.extend(tokenizer.encode(formatted))
        if len(buffer_tokens) >= max_tokens_per_shard:
            flush_shard()

    # 2. Ingest German Wikipedia (20231101.de)
    print("\n🇩🇪 [Quelle 2/3] Streame Deutsche Wikipedia (Syntax, Vokabular & Morphologie)...", flush=True)
    try:
        wiki_de = load_dataset("wikimedia/wikipedia", "20231101.de", split="train", streaming=True)
        count_de = 0
        skip_articles = max(0, (shard_count - 24) * 300)
        ds_iter_de = iter(wiki_de)

        if skip_articles > 0:
            print(f"  ⏭️ Überspringe ca. {skip_articles:,} bereits verarbeitete DE-Artikel...", flush=True)
            for _ in range(skip_articles):
                try:
                    next(ds_iter_de)
                except StopIteration:
                    break

        for item in ds_iter_de:
            if STOP_REQUESTED or shard_count >= (target_shards - 30):
                break
            text = item.get("text", "").strip()
            if len(text) < 150:
                continue

            toks = tokenizer.encode(text + "\n\n")
            buffer_tokens.extend(toks)
            count_de += 1

            if len(buffer_tokens) >= max_tokens_per_shard:
                flush_shard()

            if count_de % 2000 == 0:
                print(f"  ⚙️ [Wiki-DE] {count_de:,} Artikel gestreamt (Shards: {shard_count})", flush=True)

        print(f"✅ Deutsche Wikipedia abgeschlossen ({count_de:,} Artikel verarbeitet).", flush=True)
    except Exception as e:
        print(f"⚠️ Hinweis bei Wiki-DE Stream: {e}", flush=True)

    # 3. Ingest English Wikipedia (20231101.en)
    if shard_count < target_shards and not STOP_REQUESTED:
        print("\n🇬🇧 [Quelle 3/3] Streame Englische Wikipedia (Internationale Syntax & Terminologie)...", flush=True)
        try:
            wiki_en = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
            count_en = 0
            for item in wiki_en:
                if STOP_REQUESTED or shard_count >= target_shards:
                    break
                text = item.get("text", "").strip()
                if len(text) < 150:
                    continue

                toks = tokenizer.encode(text + "\n\n")
                buffer_tokens.extend(toks)
                count_en += 1

                if len(buffer_tokens) >= max_tokens_per_shard:
                    flush_shard()

                if count_en % 2000 == 0:
                    print(f"  ⚙️ [Wiki-EN] {count_en:,} Artikel gestreamt (Shards: {shard_count})", flush=True)

            print(f"✅ Englische Wikipedia abgeschlossen ({count_en:,} Artikel verarbeitet).", flush=True)
        except Exception as e:
            print(f"⚠️ Hinweis bei Wiki-EN Stream: {e}", flush=True)

    flush_shard()
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 Foundation Bitstream Expansion abgeschlossen! Gesamte Shards: {shard_count} (~{total_tokens_written:,} Tokens)", flush=True)
    print("=" * 80, flush=True)
    return shard_count


if __name__ == "__main__":
    run_foundation_expansion()
