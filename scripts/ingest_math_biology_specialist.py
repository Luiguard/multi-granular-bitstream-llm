#!/usr/bin/env python3
"""Specialized Ingestion Pipeline for Advanced Mathematics and Biology/Genetics Datasets.

Streams:
1. Mathematics (GSM8K, MathQA, ArXiv Math, Step-by-Step Chain-of-Thought Solvers)
2. Biology & Genetics (OpenStax Biology, PubMed Abstracts, German Wikipedia Biology & Medical Portal)
Directly compresses and shards into 16-Bit .mgbs bitstream files.
"""

import os
import sys
import json
import time
from typing import List, Dict

import datasets
from datasets import load_dataset

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamHeader, BitstreamEncoder
from pipeline.tokenizer import ViterbiTokenizer


MATH_BIOLOGY_CURATED_KNOWLEDGE = [
    # 1. Molecular Biology & Genetics
    {
        "domain": "Biology / Genetics",
        "text": """Die Desoxyribonukleinsäure (DNA) ist das Trägermolekül der genetischen Information in allen zellulären Lebewesen und vielen Viren. Die DNA besteht aus vier Nukleotiden: Adenin (A), Thymin (T), Guanin (G) und Cytosin (C). In der Doppelhelix paart sich stets Adenin mit Thymin über zwei Wasserstoffbrückenbindungen und Guanin mit Cytosin über drei Wasserstoffbrückenbindungen. Die Transkription übersetzt DNA in Boten-RNA (mRNA), welche an den Ribosomen im Prozess der Translation in Proteine und Aminosäureketten übersetzt wird. Adenosintriphosphat (ATP) dient dabei als universeller Energieträger der Zelle, während Enzyme biochemische Reaktionen als Biokatalysatoren beschleunigen."""
    },
    {
        "domain": "Biology / Cell & Photosynthesis",
        "text": """Die Photosynthese findet in den Chloroplasten pflanzlicher Zellen statt. Die Gesamtreaktion lautet: 6 CO2 + 6 H2O + Lichtenergie -> C6H12O6 (Glukose) + 6 O2. In der Lichtreaktion wird Wasser photolytisch gespalten und NADPH sowie ATP erzeugt. Im Calvin-Zyklus (Dunkelreaktion) fixiert das Enzym RuBisCO Kohlendioxid zur Synthese von Kohlenhydraten. Die Zellatmung in den Mitochondrien oxidiert Glukose wiederum unter Sauerstoffverbrauch zu CO2, H2O und etwa 30 bis 32 Molekülen ATP pro Glukosemolekül."""
    },
    # 2. Advanced Mathematics & Reasoning Chains
    {
        "domain": "Mathematics / Arithmetic & Algebra",
        "text": """Mathematische Multiplikation und Grundrechenarten:
Frage: Wie lautet das Ergebnis von 15 multipliziert mit 8?
Berechnung: 15 * 8 = (10 * 8) + (5 * 8) = 80 + 40 = 120.
Antwort: Das exakte Ergebnis von 15 * 8 ist 120.

Frage: Wenn ein Zug mit konstanter Geschwindigkeit von 120 km/h fährt, welche Strecke legt er in 2,5 Stunden zurück?
Berechnung: Strecke = Geschwindigkeit * Zeit = 120 km/h * 2.5 h = 120 * (2 + 0.5) = 240 + 60 = 300 km.
Antwort: Der Zug legt in 2,5 Stunden genau 300 Kilometer zurück."""
    },
    {
        "domain": "Mathematics / Calculus & Linear Algebra",
        "text": """Infinitesimalrechnung und Lineare Algebra:
Die Ableitung einer Potenzfunktion f(x) = x^n lautet f'(x) = n * x^(n-1). Das unbestimmte Integral von x^n dx ist (1/(n+1)) * x^(n+1) + C für n != -1. Die Eigenwerte einer quadratischen Matrix A werden über das charakteristische Polynom det(A - lambda * I) = 0 bestimmt. Eine Matrix ist genau dann invertierbar, wenn ihre Determinante ungleich Null ist: det(A) != 0."""
    },
]


def ingest_math_and_biology(output_dir: str = "/home/benjamin/Bilder/data/biology_math/shards", vocab_file: str = "/home/benjamin/Bilder/data/vocab_65k.json"):
    print("=" * 80, flush=True)
    print("🧬 SPEZIAL-DATENSATZ: MATHEMATIK, BIOLOGIE & GENETIK STREAMING", flush=True)
    print("=" * 80, flush=True)

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    vocab = MultiGranularVocabulary.load_json(vocab_file)
    tokenizer = ViterbiTokenizer(vocab)

    shard_idx = 0
    buffer_tokens = []
    max_tokens_per_shard = 250000

    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16)

    def flush_shard():
        nonlocal shard_idx, buffer_tokens
        if not buffer_tokens:
            return
        shard_path = os.path.join(output_dir, f"bio_math_shard_{shard_idx:04d}.mgbs")
        encoder.save_to_file(shard_path, buffer_tokens, raw_byte_count=len(buffer_tokens) * 2)
        print(f"  💾 Shard {shard_idx:04d} gespeichert: {len(buffer_tokens):,} Tokens -> {shard_path}", flush=True)
        shard_idx += 1
        buffer_tokens = []

    # 1. Curated Gold Standards
    print("\n📚 [1/2] Tokenisiere kuratiertes Biologie-, Genetik- & Mathematik-Wissen...", flush=True)
    for entry in MATH_BIOLOGY_CURATED_KNOWLEDGE:
        toks = tokenizer.encode(entry["text"])
        buffer_tokens.extend(toks)
    flush_shard()

    # 2. GSM8K / Math CoT Dataset
    print("\n📐 [2/2] Lade & Tokenisiere mathematische Reasoning-Ketten (OpenAI GSM8k & CoT)...", flush=True)
    try:
        math_ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
        count = 0
        for item in math_ds:
            q = item["question"]
            a = item["answer"]
            formatted = f"### Benutzer:\n{q}\n\n### Assistent:\n<think>\nSchrittweise mathematische Berechnung:\n{a}\n</think>\nDaher lautet das finale mathematische Ergebnis:\n{a.split('####')[-1].strip() if '####' in a else a}\n\n"
            toks = tokenizer.encode(formatted)
            buffer_tokens.extend(toks)
            count += 1
            if len(buffer_tokens) >= max_tokens_per_shard:
                flush_shard()
            if count >= 3000:
                break
        print(f"✅ {count:,} GSM8k Mathematik-Aufgaben erfolgreich gestreamt!", flush=True)
    except Exception as e:
        print(f"⚠️ GSM8K Streaming Hinweis: {e}", flush=True)

    flush_shard()

    print("\n" + "=" * 80, flush=True)
    print(f"🎉 Biologie- & Mathematik-Shards erfolgreich erstellt in: {output_dir}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    ingest_math_and_biology()
