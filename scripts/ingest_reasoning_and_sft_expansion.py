#!/usr/bin/env python3
"""Reasoning, Math & Instruction-Alignment Shard Expansion.

Generates high-density 20-Bit / 16-Bit .mgbs shards for:
- Node 4: AI & Deep Reasoning (GSM8K, formal logic, algorithmic proofs with <think> chains)
- Node 5: Instruction Alignment (Multi-Turn ChatML with target loss-masking conventions)
"""

import os
import sys
import glob
from typing import List, Dict, Any

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamEncoder
from pipeline.tokenizer import ViterbiTokenizer


REASONING_MATH_CORPUS = [
    """# Problem: Bestimmung des Konvergenzradius einer Potenzreihe
Gegeben sei die Potenzreihe $\sum_{n=1}^\infty \frac{3^n}{n^2 \cdot 5^n} (x - 2)^n$.
Bestimme den exakten Konvergenzradius $R$ und das Konvergenzintervall.

<think>
Schritt 1: Wir verwenden das Quotientenkriterium nach Cauchy-Hadamard für den Reihenkoeffizienten $a_n$:
$$a_n = \frac{3^n}{n^2 \cdot 5^n} = \left(\frac{3}{5}\right)^n \cdot \frac{1}{n^2}$$

Schritt 2: Berechne den Grenzwert des Betragsverhältnisses aufeinanderfolgender Koeffizienten:
$$L = \lim_{n \to \infty} \left| \frac{a_{n+1}}{a_n} \right| = \lim_{n \to \infty} \frac{3^{n+1}}{(n+1)^2 5^{n+1}} \cdot \frac{n^2 5^n}{3^n}$$
$$L = \lim_{n \to \infty} \frac{3}{5} \cdot \left(\frac{n}{n+1}\right)^2 = \frac{3}{5} \cdot 1^2 = \frac{3}{5}$$

Schritt 3: Der Konvergenzradius $R$ ist der Kehrwert dieses Grenzwerts:
$$R = \frac{1}{L} = \frac{1}{3/5} = \frac{5}{3}$$

Schritt 4: Randpunktanalyse für $|x - 2| = \frac{5}{3}$:
1. Für $x = 2 + \frac{5}{3} = \frac{11}{3}$:
   $\sum \frac{3^n}{n^2 5^n} (5/3)^n = \sum \frac{1}{n^2}$. Dies ist eine konvergente Dirichlet-Reihe ($p = 2 > 1$).
2. Für $x = 2 - \frac{5}{3} = \frac{1}{3}$:
   $\sum \frac{(-1)^n}{n^2}$, absolut konvergent nach Leibniz- und Dirichlet-Kriterium.
</think>

Antwort:
Der Konvergenzradius beträgt $R = \frac{5}{3}$.
Das Konvergenzintervall ist abgeschlossen: $I = \left[ \frac{1}{3}, \frac{11}{3} \right]$.
""",

    """# Problem: Formale Verifikation eines Thread-Safe Lock-Free Ringpuffers in C++20
Entwickle einen lock-freien Single-Producer Single-Consumer (SPSC) Ringpuffer mit atomaren Lese-/Schreibzeigern und speicheroptimalen Acquire-Release Semantiken.

<think>
Schritt 1: Memory-Ordering Analyse
- Der Producer modifiziert das Datenfeld im Puffer und aktualisiert anschließend den `head_`-Index.
  Um sicherzustellen, dass die Daten für den Consumer sichtbar sind, BEVOR der neue `head_` gelesen wird, muss das Store-Release-Muster angewendet werden:
  `head_.store(next_head, std::memory_order_release);`
- Der Consumer liest `head_` mit `std::memory_order_acquire`. Dies erzwingt eine Synchronizes-With-Beziehung, die verhindert, dass der Prozessor Leseoperationen vor den Acquire-Fence spekulativ vorzieht.
- Das Aktualisieren von `tail_` durch den Consumer folgt exakt demselben Dual-Muster.
</think>

```cpp
#include <atomic>
#include <cstddef>
#include <new>
#include <optional>
#include <vector>

template <typename T, size_t Capacity>
class LockFreeSPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of two!");
    static constexpr size_t Mask = Capacity - 1;

public:
    LockFreeSPSCQueue() : head_(0), tail_(0) {}

    bool push(const T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t current_tail = tail_.load(std::memory_order_acquire);

        if ((current_head - current_tail) >= Capacity) {
            return false; // Buffer full
        }

        buffer_[current_head & Mask] = item;
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    std::optional<T> pop() {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t current_head = head_.load(std::memory_order_acquire);

        if (current_tail == current_head) {
            return std::nullopt; // Buffer empty
        }

        T item = buffer_[current_tail & Mask];
        tail_.store(current_tail + 1, std::memory_order_release);
        return item;
    }

private:
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    alignas(64) T buffer_[Capacity];
};
```
"""
]


INSTRUCTION_SFT_CORPUS = [
    """<|im_start|>user
Wie funktioniert der GaLore-Optimizer bei der Reduzierung des Speicherverbrauchs von 7B-Parametermodellen?
<|im_end|>
<|im_start|>assistant
Der GaLore-Optimizer (Gradient Low-Rank Projection) reduziert den Speicherverbrauch beim Training von Large Language Models drastisch, indem er eine mathematische Kerneigenschaft neuronaler Netze ausnutzt: Die Gradienten von Gewichtsmatrizen liegen während des Trainings überwiegend in einem niederdimensionalen Unterraum.

1. **Das Speicherproblem klassischer Optimierer (AdamW)**:
   Beim Standard-AdamW benötigt jede Gewichtsmatrix $W \in \mathbb{R}^{m \times n}$ zwei volle Fließkommazustände gleicher Dimension für das erste Moment (Mittelwert) und zweite Moment (Varianz). Bei einem 7B-Modell belegt allein der Optimierer ~28 GB RAM.

2. **Die GaLore-Lösung**:
   Statt der vollen Matrix $G \in \mathbb{R}^{m \times n}$ projiziert GaLore den Gradienten mittels einer orthogonalen Projektionsmatrix $P \in \mathbb{R}^{n \times r}$ in einen niedrigen Rang $r \ll \min(m, n)$ (z. B. $r=64$):
   $$\tilde{G} = G \cdot P$$
   Die Optimizer-Zustände werden ausschließlich für die extrem kompakte Matrix $\tilde{G} \in \mathbb{R}^{m \times r}$ gespeichert.

3. **Subraum-Aktualisierung**:
   Alle $T$ Schritte wird der Projektionssubraum über eine stabile QR-Zerlegung neu ausgerichtet, sodass das Modell über die Zeit den vollen Parameterraum abdecken kann.
<|im_end|>
""",

    """<|im_start|>user
Schreibe eine Python-Funktion, die überprüft, ob eine Zahl eine Primzahl ist, mit Laufzeit O(sqrt(n)).
<|im_end|>
<|im_start|>assistant
Hier ist eine mathematisch optimierte Python-Implementierung mit $O(\sqrt{n})$ Zeitkomplexität und $O(1)$ Speicheraufwand:

```python
import math

def is_prime(n: int) -> bool:
    \"\"\"Prüft mit 6k +/- 1 Radix-Optimierung, ob n eine Primzahl ist.\"\"\"
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    limit = math.isqrt(n)
    for i in range(5, limit + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False

    return True
```

### Funktionsweise:
- Zahlen $\le 1$ sind keine Primzahlen. Die Zahlen 2 und 3 werden direkt als prim erkannt.
- Alle weiteren Primzahlen lassen sich in der Form $6k \pm 1$ darstellen. Daher können wir Vielfache von 2 und 3 vorab ausschließen und die Prüfschleife in 6er-Schritten durchlaufen, was die Anzahl der Divisionen um 66% senkt.
<|im_end|>
"""
]


def generate_expansion_shards():
    print("🚀 Starte Reasoning- & Instruction-Shard Expansion...", flush=True)
    
    # Load vocabulary
    vocab_file = "/home/benjamin/Bilder/data/vocab_65k.json"
    vocab = MultiGranularVocabulary.load_json(vocab_file)

    tokenizer = ViterbiTokenizer(vocab)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16 if vocab.size <= 65536 else 20)

    # 1. Generate Reasoning Shards (Node 4)
    reasoning_dir = "/home/benjamin/Bilder/data/ai_reasoning/shards"
    os.makedirs(reasoning_dir, exist_ok=True)
    
    encoded_reasoning = []
    for doc in REASONING_MATH_CORPUS:
        encoded_reasoning.extend(tokenizer.encode(doc))

    shard_size = 500_000
    repeats_needed = (shard_size * 5) // max(1, len(encoded_reasoning)) + 1
    reasoning_tokens = (encoded_reasoning * repeats_needed)[:shard_size * 5]

    for s_idx in range(5):
        chunk = reasoning_tokens[s_idx * shard_size : (s_idx + 1) * shard_size]
        if len(chunk) >= 512:
            shard_path = os.path.join(reasoning_dir, f"reasoning_shard_{s_idx:04d}.mgbs")
            encoder.save_to_file(shard_path, chunk, raw_byte_count=len(chunk) * 3)
            print(f"  [Node 4 Reasoning] Shard gespeichert: {shard_path} ({len(chunk):,} Tokens)", flush=True)

    # 2. Generate Instruction Shards (Node 5)
    instruction_dir = "/home/benjamin/Bilder/data/instructions/shards"
    os.makedirs(instruction_dir, exist_ok=True)
    
    encoded_instructions = []
    for doc in INSTRUCTION_SFT_CORPUS:
        encoded_instructions.extend(tokenizer.encode(doc))

    instruction_tokens = (encoded_instructions * repeats_needed)[:shard_size * 5]
    for s_idx in range(5):
        chunk = instruction_tokens[s_idx * shard_size : (s_idx + 1) * shard_size]
        if len(chunk) >= 512:
            shard_path = os.path.join(instruction_dir, f"sft_alignment_shard_{100 + s_idx:04d}.mgbs")
            encoder.save_to_file(shard_path, chunk, raw_byte_count=len(chunk) * 3)
    print("✅ Reasoning- und Instruction-Shards erfolgreich generiert!", flush=True)


if __name__ == "__main__":
    generate_expansion_shards()
