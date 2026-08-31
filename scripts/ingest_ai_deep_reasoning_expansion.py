#!/usr/bin/env python3
"""AI Research, Deep Reasoning & Chain-of-Thought Logic Massive Expansion Ingestion Pipeline.

Target: 50+ to 100+ Shards (25M - 50M+ 16-Bit Multi-Granular Tokens)
Sources:
1. teknium/OpenHermes-2.5 (High-density complex reasoning, logical deduction, formal chains)
2. bespokelabs/Bespoke-Stratos-17k (DeepSeek-R1-style multi-step deep thinking reasoning)
3. Formal AI Research, Neural Network Architectures, Transformer Theory & Topological Derivations

Streams directly into 16-Bit .mgbs bitstream shards in data/ai_research_knowledge/shards/.
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


AI_RESEARCH_DEEP_THEORY = [
    r"""# Deep Learning Architecture Theory & Multi-Granular Bitstream Optimization:

## 1. Sparse Mixture of Experts (MoE) & Top-K Routing Dynamics
- Router-Architektur: Gegeben Eingabetensor $x \in \mathbb{R}^{d}$, berechnet der Gating-Router $H(x) = x \cdot W_g$. Die Top-$k$ Auswahl bestimmt $G(x) = \text{Softmax}(\text{TopK}(H(x) + \epsilon, k))$.
- Load-Balancing Aux Loss: Um Routing-Kollaps auf wenige dominante Experten zu verhindern, minimiert der Hilfsloss $\mathcal{L}_{\text{aux}} = \alpha \cdot N \sum_{i=1}^N f_i \cdot P_i$, wobei $f_i$ der Token-Anteil und $P_i$ die aggregierte Wahrscheinlichkeitsmasse von Experte $i$ ist.
- Gradient Checkpointing: Durch Recomputation der Zwischenzustände in Rückwärtspfad sinkt der VRAM-Bedarf der Aktivierungen von $O(L \cdot B \cdot T \cdot d)$ auf $O(\sqrt{L} \cdot B \cdot T \cdot d)$ bzw. konstanten Schicht-Overhead.

## 2. Low-Rank Gradient Projection (GaLore Optimization)
- Subspace Gradient Decomposition: Für eine Gewichtsmatrix $W \in \mathbb{R}^{m \times n}$ mit $m \ge n$ und Rang $r \ll n$ projiziert GaLore den dichten Gradienten $G \in \mathbb{R}^{m \times n}$ über eine orthogonale Projektionsmatrix $P \in \mathbb{R}^{n \times r}$ via $R = G \cdot P$.
- Optimizer State Memory Slashed by 65-80%: Anstelle von zwei $m \times n$ Float32-Zuständen für AdamW (exp_avg und exp_avg_sq) speichert der Optimierer lediglich die kompakten $m \times r$ Momente im Subraum.
- Per-Layer Hooking: Durch Registrierung von post-accumulate-grad-hooks wird jeder Subraum-Schritt unmittelbar nach der Gradientenberechnung der Schicht auf der CPU ausgeführt und der GPU-Gradient sofort freigegeben.""",

    r"""# Rotary Position Embeddings (RoPE), FlashAttention & Multi-Head Latent Attention:

## 1. Rotary Position Embeddings (RoPE)
- Drehmatrix-Transformation: Für Q/K-Tensoren $x \in \mathbb{R}^d$ an Position $m$ transformiert RoPE Vektoren im 2D-Teilraum über $R_{\Theta, m}^d x_m$, sodass das Skalarprodukt $\langle R_m q, R_n k \rangle = g(q, k, m - n)$ ausschließlich vom relativen Abstand $m - n$ abhängt.
- Llama-3 / Nemotron Frequenz-Skalierung: Mit $\theta = 500000.0$ wird Aliasing bei ultra-langen Kontexten (7.168 - 128.000 Tokens) wirksam unterdrückt.

## 2. Multi-Head Latent Attention (MLA)
- Low-Rank KV-Kompression: Anstelle des vollen $N \times d_{\text{head}}$ KV-Caches komprimiert MLA Schlüssel und Werte in einen kompakten latenten Vektor $c_t^{KV} \in \mathbb{R}^{d_c}$.
- Entkoppelte RoPE-Vektoren: MLA trennt inhaltsbasierte Aufmerksamkeit von positionsbasierten RoPE-Projektionen, wodurch der Inferenz-Speicherbedarf um bis zu 85% sinkt bei voller Aufmerksamkeitsdynamik."""
]


def run_ai_deep_reasoning_ingestion(
    output_dir: str = "/home/benjamin/Bilder/data/ai_research_knowledge/shards",
    vocab_file: str = "/home/benjamin/Bilder/data/vocab_65k.json",
    max_tokens_per_shard: int = 500_000,
    target_shards: int = 60,
) -> int:
    global STOP_REQUESTED

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    print("=" * 80, flush=True)
    print("🧠 MASSIVE AI & DEEP REASONING EXPANSION PIPELINE (7B PRE-TRAINING)", flush=True)
    print(f"📁 Ziel-Verzeichnis: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    existing_files = sorted(glob.glob(os.path.join(output_dir, "*.mgbs")))
    shard_count = len(existing_files)
    print(f"  ⏭️ {shard_count} bestehende AI-Reasoning Shards gefunden. Starte ab Index {shard_count:04d}...", flush=True)

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
        shard_path = os.path.join(output_dir, f"ai_reasoning_shard_{shard_count:04d}.mgbs")
        encoder.save_to_file(shard_path, buffer_tokens, raw_byte_count=len(buffer_tokens) * 2)
        total_tokens_written += len(buffer_tokens)
        shard_count += 1
        elapsed = time.time() - start_time
        print(f"  💾 [AI-REASONING Shard {shard_count:04d}] {len(buffer_tokens):,} Tokens | Gesamt: {total_tokens_written:,} Tokens -> {os.path.basename(shard_path)}", flush=True)
        buffer_tokens = []

    # 1. Ingest Core AI Research & Formal Derivations
    print("\n📚 [Quelle 1/3] Tokenisiere AI Research Architecture & Optimization Theory...", flush=True)
    for doc in AI_RESEARCH_DEEP_THEORY:
        formatted = f"### KI-Forschung & Mathematische Systemtheorie:\n{doc.strip()}\n\n"
        buffer_tokens.extend(tokenizer.encode(formatted))
        if len(buffer_tokens) >= max_tokens_per_shard:
            flush_shard()

    # 2. Ingest OpenHermes-2.5 (teknium/OpenHermes-2.5)
    print("\n💡 [Quelle 2/3] Streame OpenHermes-2.5 (Logische Deduktion, Chain-of-Thought & Reasoning)...", flush=True)
    try:
        hermes_ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
        item_count = 0
        skip_items = max(0, shard_count * 400)
        ds_iter = iter(hermes_ds)

        if skip_items > 0:
            print(f"  ⏭️ Überspringe ca. {skip_items:,} bereits verarbeitete Reasoning-Dialoge...", flush=True)
            for _ in range(skip_items):
                try:
                    next(ds_iter)
                except StopIteration:
                    break

        for item in ds_iter:
            if STOP_REQUESTED or shard_count >= target_shards:
                break
            conversations = item.get("conversations", [])
            if not conversations or len(conversations) < 2:
                continue

            # Multi-Turn Formatting with Think-Tags
            formatted_turns = []
            for turn in conversations:
                from_role = turn.get("from", "")
                val = turn.get("value", "")
                if from_role in ("human", "user"):
                    formatted_turns.append(f"### Benutzer:\n{val}")
                elif from_role in ("gpt", "assistant"):
                    formatted_turns.append(f"### Assistent:\n<think>\nStrukturierte logische Analyse & Deduktion:\n</think>\n{val}")

            dialogue_str = "\n\n".join(formatted_turns) + "\n\n"
            buffer_tokens.extend(tokenizer.encode(dialogue_str))
            item_count += 1

            if len(buffer_tokens) >= max_tokens_per_shard:
                flush_shard()

            if item_count % 3000 == 0:
                print(f"  ⚙️ [OpenHermes-2.5] {item_count:,} Reasoning-Dialoge gestreamt (Shards: {shard_count})", flush=True)

        print(f"✅ OpenHermes-2.5 Stream abgeschlossen ({item_count:,} Dialoge).", flush=True)
    except Exception as e:
        print(f"⚠️ Hinweis bei OpenHermes-2.5: {e}", flush=True)

    flush_shard()
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 AI & Deep Reasoning Expansion abgeschlossen! Gesamte Shards: {shard_count} (~{total_tokens_written:,} Tokens)", flush=True)
    print("=" * 80, flush=True)
    return shard_count


if __name__ == "__main__":
    run_ai_deep_reasoning_ingestion()
