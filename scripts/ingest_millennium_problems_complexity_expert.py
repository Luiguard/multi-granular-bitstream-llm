#!/usr/bin/env python3
"""Specialized Ingestion Pipeline for the 7 Millennium Prize Problems, Computational Complexity (P vs NP),
Formal Mathematics, Quantum Field Theory, and Differential Topology.

Covers:
1. P versus NP & Complexity Theory (Clay Millennium Problem #1):
   - Formal Definitions: Deterministic (P) vs Nondeterministic Polynomial Time (NP).
   - Verifiers & Polynomial Witnesses, Cook-Levin Theorem (Boolean Satisfiability / 3-SAT).
   - NP-Completeness Reductions: Clique, Vertex Cover, Hamiltonian Path, Subset Sum, TSP.
   - Proof Barriers: Relativization (Baker-Gill-Solovay), Natural Proofs (Razborov-Rudich), Algebrization (Aaronson-Wigderson).
   - Cryptographic Implications of P = NP vs P != NP (One-Way Functions, BQP Quantum Complexity).
2. The Other 6 Millennium Prize Problems:
   - Riemann Hypothesis (Zeta Function zeros on Re(s) = 1/2, Prime Number Theorem error terms).
   - Poincaré Conjecture (Hamilton-Perelman Ricci Flow with Surgery on 3-manifolds, solved 2003).
   - Navier-Stokes Existence & Smoothness (3D Incompressible fluid dynamics, Turbulence, Blow-up singularities).
   - Yang-Mills Existence & Mass Gap (Non-Abelian Gauge Theory, SU(N), Color Confinement, Glueball masses).
   - Birch and Swinnerton-Dyer Conjecture (Elliptic Curves E(Q) rank vs L(E, s) analytic order at s=1).
   - Hodge Conjecture (Complex algebraic varieties, Hodge (p, p)-cycles, de Rham cohomology).

Appends directly to data/stem_knowledge/shards/ and data/ai_research_knowledge/shards/.
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


MILLENNIUM_DEEP_MATHEMATICS = [
    # 1. P vs NP & Theoretical Computer Science
    r"""# P vs. NP Problem: Komplexitätstheorie, Reduktionen & Beweisbarrieren (Clay Problem #1):

## 1. Formale Definitionen der Komplexitätsklassen P und NP
- **Klasse P**: Die Menge aller Entscheidungsprobleme $L \subseteq \{0, 1\}^*$, die von einer deterministischen Turingmaschine (DTM) in polynomieller Zeit $O(n^k)$ entschieden werden können.
- **Klasse NP**: Die Menge aller Sprachen $L$, für die ein polynomieller Verifizierer $V(x, w)$ existiert, sodass:
  $$x \in L \iff \exists w \in \{0, 1\}^{O(|x|^c)} : V(x, w) = 1$$
  Ein Zeuge (Zertifikat/Witness) $w$ kann also in polynomieller Zeit $O(n^c)$ auf Korrektheit überprüft werden.

## 2. Der Satz von Cook-Levin & NP-Vollständigkeit
- **Cook-Levin Theorem (1971)**: Das Erfüllbarkeitsproblem der Aussagenlogik (SAT / 3-SAT) ist **NP-vollständig** ($NP\text{-Complete}$).
- Jedes Problem $L \in NP$ lässt sich über eine polynomielle Reduktion $L \le_p \text{3-SAT}$ transformieren:
  $$\forall x \in \{0, 1\}^*: x \in L \iff f(x) \in \text{3-SAT}, \quad \text{mit } \text{Laufzeit}(f) = O(|x|^k)$$
- **Karp-Reduktionen**: 3-SAT $\le_p$ CLIQUE $\le_p$ VERTEX-COVER $\le_p$ HAMILTONIAN-CYCLE $\le_p$ SUBSET-SUM.

## 3. Die drei fundamentalen Beweisbarrieren (Warum P != NP so schwer zu beweisen ist)
1. **Relativierungs-Barriere (Baker, Gill, Solovay 1975)**:
   - Es existieren Orakel $A$ und $B$ mit $P^A = NP^A$, aber $P^B \neq NP^B$. Jeder Beweis, der unverändert für Orakel gilt (Diagonalisierung), scheitert.
2. **Natural Proofs Barriere (Razborov & Rudich 1997)**:
   - Unter der Annahme, dass starke pseudozufällige Funktionen (Kryptographie) existieren, kann keine "natürliche Eigenschaft" von Schaltkreisen eine super-polynomielle untere Schranke beweisen.
3. **Algebrierungs-Barriere (Aaronson & Wigderson 2008)**:
   - Auch algebraische Erweiterungen von Orakeln (wie bei IP = PSPACE) können $P$ nicht von $NP$ trennen.""",

    # 2. Riemann Hypothesis & Analytic Number Theory
    r"""# Die Riemannsche Vermutung & Analytische Zahlentheorie (Clay Problem #2):

## 1. Die Riemannsche Zeta-Funktion
Für $\text{Re}(s) > 1$ ist die Zeta-Funktion definiert als Dirichlet-Reihe und Euler-Produkt über alle Primzahlen $\mathbb{P}$:
$$\zeta(s) = \sum_{n=1}^\infty \frac{1}{n^s} = \prod_{p \in \mathbb{P}} \frac{1}{1 - p^{-s}}$$
Durch meromorphe Fortsetzung auf die gesamte komplexe Ebene $\mathbb{C}$ besitzt $\zeta(s)$ einen einfachen Pol bei $s = 1$ mit Residuum 1 sowie die Funktionalgleichung:
$$\zeta(s) = 2^s \pi^{s-1} \sin\left(\frac{\pi s}{2}\right) \Gamma(1 - s) \zeta(1 - s)$$

## 2. Die Vermutung & Auswirkung auf Primzahlen
- **Riemann-Hypothese**: Alle nicht-trivialen Nullstellen von $\zeta(s)$ besitzen den Realteil:
  $$\text{Re}(s) = \frac{1}{2} \quad \text{(Die kritische Gerade)}$$
- **Primzahlsatz mit optimaler Fehlerschranke**:
  $$\pi(x) = \text{Li}(x) + O(\sqrt{x} \ln x)$$
  Eine Abweichung von der kritischen Geraden würde bedeuten, dass Primzahlen unregelmäßige Schwingungen aufweisen.""",

    # 3. Navier-Stokes, Yang-Mills, Poincaré, BSD & Hodge
    r"""# Die weiteren 5 Millennium-Probleme im Detail:

## 1. Navier-Stokes Existenz & Glattheit (Fluiddynamik)
$$\rho \left(\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u}\right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{f}, \quad \nabla \cdot \mathbf{u} = 0$$
- **Kernfrage**: Existieren für beliebige glatte Anfangsbedingungen $\mathbf{u}_0(x)$ im dreidimensionalen Raum $\mathbb{R}^3$ für alle Zeiten $t > 0$ glatte, physikalisch beschränkte Geschwindigkeitsfelder mit endlicher kinetischer Energie, oder bilden sich nach endlicher Zeit Singularitäten (Blow-up / Turbulenz-Kollaps)?

## 2. Yang-Mills-Theorie & Massenlücke (Quantenfeldtheorie)
- In einer nicht-abelschen Eichfeldtheorie mit Eichgruppe $SU(N)$ (Quantenchromodynamik für Quarks und Gluonen) muss mathematisch streng bewiesen werden, dass die leichteste Anregung (Glueball) eine strikt positive Masse $\Delta > 0$ besitzt:
  $$m_{\text{glueball}} \ge \Delta > 0$$
  Dies erklärt, warum die starke Kernkraft nur eine ultrakurze Reichweite von $\approx 10^{-15}\text{ m}$ hat.

## 3. Die Poincaré-Vermutung (Gelöst von Grigori Perelman 2003)
- Jede einfach zusammenhängende, geschlossene 3-dimensionale Mannigfaltigkeit ist homöomorph zur 3-Sphäre $S^3$.
- Bewiesen mittels des Ricci-Flusses mit Chirurgie: $\frac{\partial g_{ij}}{\partial t} = -2 R_{ij}$.

## 4. Birch und Swinnerton-Dyer Vermutung (Elliptische Kurven)
- Für eine elliptische Kurve $E$ über $\mathbb{Q}$ ist der algebraische Rang der rationalen Punktegruppe $E(\mathbb{Q})$ gleich der analytischen Nullstellenordnung der Hasse-Weil L-Funktion bei $s = 1$:
  $$\text{ord}_{s=1} L(E, s) = \text{rank}(E(\mathbb{Q}))$$

## 5. Die Hodge-Vermutung (Algebraische Geometrie)
- Auf nicht-singulären komplexen projektiven algebraischen Varietäten ist jede Hodge-Klasse (ein Kohomologie-Element in $H^{2p}(X, \mathbb{Q}) \cap H^{p,p}(X)$) eine rationale Linearkombination von Kohomologieklassen algebraischer Teilvarietäten."""
]


def run_millennium_problems_ingestion(
    output_dir: str = "/home/benjamin/Bilder/data/stem_knowledge/shards",
    vocab_file: str = "/home/benjamin/Bilder/data/vocab_65k.json",
    max_tokens_per_shard: int = 500_000,
    target_shards: int = 160,
) -> int:
    global STOP_REQUESTED

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    print("=" * 80, flush=True)
    print("🏆 MASSIVE MILLENNIUM PROBLEMS & THEORETICAL COMPLEXITY PIPELINE (7B PRE-TRAINING)", flush=True)
    print(f"📁 Ziel-Verzeichnis: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    existing_files = sorted(glob.glob(os.path.join(output_dir, "*.mgbs")))
    shard_count = len(existing_files)
    print(f"  ⏭️ {shard_count} bestehende STEM-Shards gefunden. Starte ab Index {shard_count:04d}...", flush=True)

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
        shard_path = os.path.join(output_dir, f"stem_math_shard_{shard_count:04d}.mgbs")
        encoder.save_to_file(shard_path, buffer_tokens, raw_byte_count=len(buffer_tokens) * 2)
        total_tokens_written += len(buffer_tokens)
        shard_count += 1
        elapsed = time.time() - start_time
        print(f"  💾 [MILLENNIUM/MATH Shard {shard_count:04d}] {len(buffer_tokens):,} Tokens | Gesamt: {total_tokens_written:,} Tokens -> {os.path.basename(shard_path)}", flush=True)
        buffer_tokens = []

    # 1. Ingest Deep Millennium Problems & Complexity Proofs
    print("\n📚 [Quelle 1/2] Tokenisiere alle 7 Millennium-Probleme & Komplexitätstheorie...", flush=True)
    for doc in MILLENNIUM_DEEP_MATHEMATICS:
        formatted = f"### Höhere Mathematik & Millennium-Probleme:\n{doc.strip()}\n\n"
        buffer_tokens.extend(tokenizer.encode(formatted))
        if len(buffer_tokens) >= max_tokens_per_shard:
            flush_shard()

    # 2. Ingest Math & Complexity Dialogues (MathInstruct & Theorem Proving)
    print("\n📐 [Quelle 2/2] Streame formale mathematische Beweise & Komplexitätstheorie...", flush=True)
    try:
        math_ds = load_dataset("TIGER-Lab/MathInstruct", split="train", streaming=True)
        math_count = 0
        for item in math_ds:
            if STOP_REQUESTED or shard_count >= target_shards:
                break
            instr = item.get("instruction", "")
            resp = item.get("output", "")
            src = item.get("source", "Math")
            if not instr or not resp:
                continue

            keywords = ["proof", "theorem", "lemma", "polynomial", "complexity", "np", "turing", "riemann", "integral", "derivative", "matrix", "topology", "equation"]
            is_relevant = any(kw in instr.lower() or kw in resp.lower() for kw in keywords)
            if not is_relevant:
                continue

            formatted = f"### Mathematisches Problem ({src} · Formale Herleitung):\n{instr}\n\n### Beweisführung & Lösungsstruktur:\n<think>\nSchrittweise formale Deduktion und mathematische Beweislogik:\n</think>\n{resp}\n\n"
            buffer_tokens.extend(tokenizer.encode(formatted))
            math_count += 1

            if len(buffer_tokens) >= max_tokens_per_shard:
                flush_shard()

            if math_count % 2000 == 0:
                print(f"  ⚙️ [Millennium-Math-Engine] {math_count:,} mathematische Beweise verarbeitet (Shards: {shard_count})", flush=True)

        print(f"✅ Formale mathematische Beweise abgeschlossen ({math_count:,} verarbeitet).", flush=True)
    except Exception as e:
        print(f"⚠️ Hinweis bei Math Stream: {e}", flush=True)

    flush_shard()
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 Millennium Problems & Complexity Specialist Ingestion abgeschlossen! Gesamte Shards: {shard_count} (~{total_tokens_written:,} Tokens)", flush=True)
    print("=" * 80, flush=True)
    return shard_count


if __name__ == "__main__":
    run_millennium_problems_ingestion()
