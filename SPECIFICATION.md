# KI-Spezifikation: Multi-Granularitäts-Token- & Bitstream-System

Diese Spezifikation definiert das Trainings- und Inferenzsystem für Multi-Granularitäts-Tokens und komprimierte Bitstreams.

---

## 1. Ziel des Systems

Definiere ein Trainings- und Inferenzsystem, das Text nicht als Zeichenfolge (UTF‑8 / BPE), sondern als mehrstufige Token-Sequenzen (Wörter, Phrasen, Satzmuster, Byte-Fallback) verarbeitet und diese in kompakte Binär-Bitstreams serialisiert.  
Das Modell soll:

* Bitstreams als Eingabe verarbeiten  
* Token-Sequenzen korrekt fortsetzen  
* Verlustfrei rekonstruierbaren Text erzeugen  
* Mit deutlich weniger Parametern und reduzierter Aufmerksamkeits-Komplexität ($O(T^2)$) arbeiten

---

## 2. Vokabular-Architektur

### Vokabular-Hierarchie:

* **Tier 0 – Byte-Fallback:**
  * 256 Tokens für alle möglichen Byte-Werte (`0x00`–`0xFF`)
  * Garantiert Zero-OOV für beliebige Eingaben (Binärdaten, Unicode, Emojis, Code)
* **Tier 1 – Wort-Tokens:**
  * Tokens für einzelne Wörter (nach Normalisierung)
* **Tier 2 – Phrasen-Tokens:**
  * Tokens für häufige N-Gramme ($N=2\dots5$), induziert über Pointwise Mutual Information (PMI) und Information Gain
* **Tier 3 – Satzmuster / Templates:**
  * Tokens für wiederkehrende Satzstrukturen (z. B. *„ich bin ein mensch“*, *„auf der grundlage von“*)

### Bitbreite:

Für ein Vokabular der Größe $|V|$ gilt:
$$\text{bitwidth} = \lceil \log_2 |V| \rceil$$
Alle Token-IDs werden mit dieser Bitbreite dicht gepackt.

### Constraints:

* Jede Token-ID ist eindeutig.  
* Alle Tokens sind verlustfrei dekodierbar.  
* Zero-OOV durch Tier-0-Byte-Fallback.

---

## 3. Segmentierung & Tokenisierung (Viterbi)

### Ziel:
Finde die global optimalen Token-Sequenzen für einen gegebenen Text, basierend auf dem Vokabular und statistischen Kennzahlen.

### Verfahren:
* Viterbi-Dynamische Programmierung zur Zerlegung des Textes in Tokens.
* Kostenfunktion kombiniert:
  $$\text{Cost}(t) = -\log_2 P(t) + \lambda \cdot \text{LengthPenalty}(t)$$
  * Häufigkeit (N-Gramm-Counts)
  * Pointwise Mutual Information (PMI)
  * Information Gain
  * Token-Länge (Bytes)
* Ergebnis ist eine Sequenz von Token-IDs, die:
  * Verlustfrei rekonstruierbar ist
  * Zero-OOV garantiert
  * Global optimal bzgl. der Kostenfunktion ist

### Beispiel:
* **Eingabe:** `"ich bin ein mensch"`
* **Ausgabe-Tokens:**
  * `0390` → `"ich "` (Tier: Phrase)
  * `0351` → `"bin ein mensch"` (Tier: Phrase)
* **Bitstream:** `110000110 101011111` (18 Bits)
* **Rekonstruktion:** `"ich bin ein mensch"` (100 % verlustfrei)

---

## 4. Bitstream-Format & I/O

### Bit-Packing:
* Token-IDs werden mit fester Bitbreite (z. B. 9–18 Bit) dicht gepackt.
* Keine Padding-Bits zwischen Tokens.

### Header (MGBS – Multi-Granular Bitstream Spec):
* Magic Bytes: `b"MGBS"` (4 Bytes)
* Version: uint16
* Bitbreite: uint8
* Vokabulargröße $|V|$: uint32
* Anzahl Tokens: uint64
* Anzahl Raw-Bytes: uint64

### Constraints:
* Jeder Bitstream muss vollständig und deterministisch dekodierbar sein.  
* Roundtrip: `Text → Tokens → Bitstream → Tokens → Text` ist 100 % verlustfrei.

---

## 5. Modellarchitektur

### Eingabe:  
`Bitstream → Token-IDs → Factorized Embedding → Transformer Blocks → Output Projection`

### Embedding-Schicht:
Verwende faktorisierte Embeddings:
$$W_e = E_{\text{vocab}} \times E_{\text{proj}}$$
* $E_{\text{vocab}} \in \mathbb{R}^{|V| \times r}$  
* $E_{\text{proj}} \in \mathbb{R}^{r \times d}$  
* Ziel: Reduktion der Parameterzahl bei gleicher Modellkapazität ($r \ll d$).

### Loss-Funktion:
**Byte-Weighted CrossEntropy Loss:**
$$\mathcal{L} = -\frac{\sum_{t=1}^T \text{bytes}(x_t) \cdot \log P(x_t \mid x_{<t})}{\sum_{t=1}^T \text{bytes}(x_t)}$$
* Tokens werden nach ihrer Byte-Länge gewichtet, um Entropie-Verzerrungen bei Multi-Wort-Einheiten zu verhindern.

---

## 6. Trainingspipeline

1. **Rohtext laden:** UTF-8 Korpus (z. B. Wikipedia, The Pile, OpenWebText).
2. **Normalisierung:** Whitespace, HTML-Bereinigung.
3. **N-Gramm-Mining:** Berechnung von Häufigkeiten ($N=1\dots5$), PMI und Information Gain.
4. **Vokabular aufbauen:** 4 Tiers (Bytes, Wörter, Phrasen, Templates) + Bitbreiten-Kalkulation.
5. **Tokenisierung:** Viterbi-Segmentierung aller Texte in optimierte Token-Sequenzen.
6. **Bitstream-Erzeugung:** Bit-Packing in `.mgbs`-Dateien.
7. **Training:** Streaming-Dataset mit `FactorizedEmbedding` und `ByteWeightedCrossEntropyLoss`.
8. **Evaluierung:** Verlustfreie Rekonstruktion, Perplexity pro Byte und Gradientenstabilität.

---

## 7. Constraints & Qualitätskriterien

### Hard Constraints:
* **Zero-OOV:** Jeder Input ist darstellbar (Byte-Fallback).  
* **Lossless Roundtrip:** Kein Informationsverlust bei Dekodierung.  
* **Deterministische Segmentierung:** Gleicher Input → gleiche Token-Sequenz.  
* **Keine Token-Kollisionen:** Jede ID eindeutig.  
* **Keine Padding-Bits im Bitstream.**

### Qualitätsmetriken:
* Kompressionsfaktor (Bytes UTF-8 $\to$ Bytes Bitstream)  
* Byte-Weighted NLL  
* Perplexity über Tokens & Bytes  
* Gradienten-Norm (Stabilität)  
* Vokabular-Abdeckung
