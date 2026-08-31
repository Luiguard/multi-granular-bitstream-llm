#!/usr/bin/env python3
"""Specialized Ingestion Pipeline for Vision-to-Code, OCR-Layout Parsing,
UI Bounding-Box Alignment, Figma JSON Translation, and Responsive Web Synthesis.

Covers:
1. OCR & Layout Token Representation to Clean Code:
   - Converting raw OCR tokens + (x, y, w, h) bounding boxes directly into semantic HTML5 and CSS Grid.
2. Figma / Penpot / UI Design Node Trees to Design Tokens:
   - Translating design node hierarchies, typography scales, HSL color tokens, and auto-layout into flexbox/grid.
3. Form & Table Document Extraction:
   - Scanning structured tabular invoices/forms into accessible HTML5 tables with ARIA and CSS micro-interactions.
4. Interactive Canvas & SVG Vector Alignment:
   - Translating 2D vector coordinates and path commands (M, L, C, Z) into reactive Canvas 2D / WebGL rendering.

Appends directly to data/cyber_web_knowledge/shards/ and data/instructions/shards/.
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


VISION_UI_TO_CODE_STANDARDS = [
    # 1. Bounding Box & OCR Layout-to-Code Specification
    r"""# Vision-to-Code Architektur: Bounding Boxes & OCR-Layout zu Semantischem HTML5/CSS3:

## 1. Schema für Vision/ONNX Bounding-Box Eingaben
Ein ONNX-Vision- oder OCR-Modell liefert normalisierte Koordinaten $[y_{\min}, x_{\min}, y_{\max}, x_{\max}] \in [0, 1000]$:
```json
{
  "viewport": { "width": 1920, "height": 1080 },
  "detected_elements": [
    { "class": "navbar", "bbox": [0, 0, 80, 1000], "content": "Home | Dashboard | Einstellungen" },
    { "class": "card", "bbox": [120, 50, 450, 480], "title": "Live Server Telemetrie", "metrics": "GPU: 78% | VRAM: 3.2GB" },
    { "class": "card", "bbox": [120, 520, 450, 950], "title": "Trainings-Fortschritt", "progress": "85%" },
    { "class": "button_primary", "bbox": [480, 50, 540, 250], "label": "⚡ Text Generieren" }
  ]
}
```

## 2. Direkte Übersetzung in modernes, responsives CSS Grid
```html
<header class="site-nav">
  <nav aria-label="Hauptmenü">
    <a href="#home">Home</a>
    <a href="#dashboard" aria-current="page">Dashboard</a>
    <a href="#settings">Einstellungen</a>
  </nav>
</header>
<main class="dashboard-grid">
  <article class="glass-card">
    <h2>Live Server Telemetrie</h2>
    <div class="metrics-row">
      <span class="badge">GPU: 78%</span>
      <span class="badge">VRAM: 3.2 GB</span>
    </div>
  </article>
  <article class="glass-card">
    <h2>Trainings-Fortschritt</h2>
    <div class="progress-bar" role="progressbar" aria-valuenow="85" aria-valuemin="0" aria-valuemax="100">
      <div class="progress-fill" style="width: 85%;">85%</div>
    </div>
  </article>
  <div class="action-bar">
    <button type="button" class="btn-primary-glow">⚡ Text Generieren</button>
  </div>
</main>
```""",

    # 2. Figma / Design Token Auto-Layout to CSS Custom Properties
    r"""# Figma Node Tree & Design Token Translation:

## 1. Übersetzung von Figma Auto-Layout in CSS
- **Figma Direction: Horizontal** $\longrightarrow$ `display: flex; flex-direction: row;`
- **Figma Direction: Vertical** $\longrightarrow$ `display: flex; flex-direction: column;`
- **Figma Item Spacing: 16px** $\longrightarrow$ `gap: 1rem;`
- **Figma Padding: [24, 32, 24, 32]** $\longrightarrow$ `padding: 1.5rem 2rem;`
- **Figma Primary Fill: #06B6D4 with 20% Opacity** $\longrightarrow$ `background: hsla(189, 94%, 43%, 0.2); backdrop-filter: blur(12px);`
- **Figma Corner Radius: 16px** $\longrightarrow$ `border-radius: 1rem;`
- **Figma Drop Shadow: Y: 8, Blur: 24, Spread: 0, #000 30%** $\longrightarrow$ `box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.3);`""",

    # 3. OCR Document Table & Invoice Extraction to Accessible Data Tables
    r"""# OCR Dokumenten- & Tabellenextraktion zu Barrierefreien HTML5-Tabellen:

```html
<div class="table-responsive-wrapper">
  <table class="data-table" aria-label="Extrahierte Abrechnungsdaten">
    <thead>
      <tr>
        <th scope="col">Position</th>
        <th scope="col">Beschreibung</th>
        <th scope="col" class="num-col">Menge</th>
        <th scope="col" class="num-col">Einzelpreis</th>
        <th scope="col" class="num-col">Gesamt</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th scope="row">01</th>
        <td>Bitstream MoE Training Shard Cluster</td>
        <td class="num-col">100</td>
        <td class="num-col">12,50 €</td>
        <td class="num-col font-bold">1.250,00 €</td>
      </tr>
    </tbody>
    <tfoot>
      <tr>
        <th scope="row" colspan="4">Gesamtbetrag (inkl. 19% MwSt.):</th>
        <td class="num-col font-bold total-highlight">1.487,50 €</td>
      </tr>
    </tfoot>
  </table>
</div>
```"""
]


def run_vision_ui_to_code_ingestion(
    output_dir: str = "/home/benjamin/Bilder/data/cyber_web_knowledge/shards",
    vocab_file: str = "/home/benjamin/Bilder/data/vocab_65k.json",
    max_tokens_per_shard: int = 500_000,
    target_shards: int = 240,
) -> int:
    global STOP_REQUESTED

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    print("=" * 80, flush=True)
    print("📸 MASSIVE VISION-TO-CODE & OCR-LAYOUT ALIGNMENT PIPELINE (7B PRE-TRAINING)", flush=True)
    print(f"📁 Ziel-Verzeichnis: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    existing_files = sorted(glob.glob(os.path.join(output_dir, "*.mgbs")))
    shard_count = len(existing_files)
    print(f"  ⏭️ {shard_count} bestehende Cyber/Web-Shards gefunden. Starte ab Index {shard_count:04d}...", flush=True)

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
        shard_path = os.path.join(output_dir, f"cyber_web_shard_{shard_count:04d}.mgbs")
        encoder.save_to_file(shard_path, buffer_tokens, raw_byte_count=len(buffer_tokens) * 2)
        total_tokens_written += len(buffer_tokens)
        shard_count += 1
        elapsed = time.time() - start_time
        print(f"  💾 [VISION/UI/OCR Shard {shard_count:04d}] {len(buffer_tokens):,} Tokens | Gesamt: {total_tokens_written:,} Tokens -> {os.path.basename(shard_path)}", flush=True)
        buffer_tokens = []

    # 1. Ingest Deep Vision-to-Code Standards
    print("\n📚 [Quelle 1/2] Tokenisiere OCR-Bounding Boxes, Figma Auto-Layout & UI AST Standards...", flush=True)
    for doc in VISION_UI_TO_CODE_STANDARDS:
        formatted = f"### Vision-to-Code & OCR Layout Architektur:\n{doc.strip()}\n\n"
        buffer_tokens.extend(tokenizer.encode(formatted))
        if len(buffer_tokens) >= max_tokens_per_shard:
            flush_shard()

    # 2. Ingest Multimodal UI/Code/SVG Dialogues
    print("\n🖼️ [Quelle 2/2] Streame UI-to-Code, SVG Coordinates & Layout Transformationen...", flush=True)
    try:
        code_ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
        v_count = 0
        for item in code_ds:
            if STOP_REQUESTED or shard_count >= target_shards:
                break
            instr = item.get("instruction", "")
            resp = item.get("output", "")
            if not instr or not resp:
                continue

            keywords = ["svg", "canvas", "layout", "ui", "design", "css", "html", "ocr", "table", "button", "modal", "navbar", "card", "responsive", "box", "image"]
            is_relevant = any(kw in instr.lower() or kw in resp.lower() for kw in keywords)
            if not is_relevant:
                continue

            formatted = f"### Benutzer (Vision / OCR UI-Spezifikation):\n{instr}\n\n### Assistent (Master Vision-to-Code Architect):\n<think>\nExtrahiere UI-Hierarchie, Bounding-Box Geometrie und generiere semantischen, barrierefreien Code:\n</think>\n{resp}\n\n"
            buffer_tokens.extend(tokenizer.encode(formatted))
            v_count += 1

            if len(buffer_tokens) >= max_tokens_per_shard:
                flush_shard()

            if v_count % 2000 == 0:
                print(f"  ⚙️ [Vision-UI-Engine] {v_count:,} Vision/UI Transformationen verarbeitet (Shards: {shard_count})", flush=True)

        print(f"✅ Vision & UI Transformationen abgeschlossen ({v_count:,} verarbeitet).", flush=True)
    except Exception as e:
        print(f"⚠️ Hinweis bei Vision Stream: {e}", flush=True)

    flush_shard()
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 Vision-to-Code Specialist Ingestion abgeschlossen! Gesamte Shards: {shard_count} (~{total_tokens_written:,} Tokens)", flush=True)
    print("=" * 80, flush=True)
    return shard_count


if __name__ == "__main__":
    run_vision_ui_to_code_ingestion()
