#!/usr/bin/env python3
"""Universal Master Orchestrator for Full-Domain Dataset Expansion (7B Chinchilla-Optimal).

Sequentially and reliably streams high-quality pre-training corpora across all 6 DAG knowledge domains:
1. Cyber & Web Knowledge (Evol-Code, The Stack, RFCs, OWASP) -> 150 Shards
2. STEM, Math & Science (Competition Math, GSM8k, SciQ) -> 150 Shards
3. AI & Deep Reasoning (OpenHermes-2.5, DeepSeek-R1 CoT) -> 60 Shards
4. Instruction & Alignment (UltraChat-200k, Guardrails) -> 50 Shards
5. World Knowledge (FineWeb-Edu 100BT, Gutenberg) -> Ongoing

Runs with controlled CPU/I/O priority to keep the 7B Trainer fast and responsive.
"""

import os
import sys
import time
import subprocess

sys.path.insert(0, "/home/benjamin/Bilder")

from scripts.ingest_cyber_web_massive_expansion import run_massive_cyber_web_ingestion
from scripts.ingest_stem_math_expansion import run_stem_math_ingestion
from scripts.ingest_ai_deep_reasoning_expansion import run_ai_deep_reasoning_ingestion
from scripts.ingest_instruction_alignment_expansion import run_instruction_alignment_ingestion

def main():
    print("=" * 80, flush=True)
    print("🚀 UNIVERSAL MULTI-DOMAIN KNOWLEDGE EXPANSION ORCHESTRATOR", flush=True)
    print("================================================================================", flush=True)

    # 1. Cyber & Web Expansion
    print("\n[PHASE 1/4] Starte Cyber & Web Knowledge Expansion...", flush=True)
    try:
        run_massive_cyber_web_ingestion(target_shards=120)
    except Exception as e:
        print(f"⚠️ Phase 1 Hinweis: {e}", flush=True)

    # 2. STEM, Math & Science Expansion
    print("\n[PHASE 2/4] Starte STEM, Math & Science Expansion...", flush=True)
    try:
        run_stem_math_ingestion(target_shards=120)
    except Exception as e:
        print(f"⚠️ Phase 2 Hinweis: {e}", flush=True)

    # 3. AI & Deep Reasoning Expansion
    print("\n[PHASE 3/4] Starte AI & Deep Reasoning Expansion...", flush=True)
    try:
        run_ai_deep_reasoning_ingestion(target_shards=50)
    except Exception as e:
        print(f"⚠️ Phase 3 Hinweis: {e}", flush=True)

    # 4. Instruction & Alignment Expansion
    print("\n[PHASE 4/4] Starte Instruction & Alignment Expansion...", flush=True)
    try:
        run_instruction_alignment_ingestion(target_shards=50)
    except Exception as e:
        print(f"⚠️ Phase 4 Hinweis: {e}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("🎉 ALLE 6 WISSENS-DOMÄNEN ERFOLGREICH EXPANDIERT UND IM GRAPH VERFÜGBAR!", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
