#!/usr/bin/env python3
"""Specialized Ingestion Pipeline for Modern Web Engineering, Semantic HTML5, Advanced CSS3,
UI/UX Design Systems, TypeScript, WebGL, Canvas, Responsive Layouts, and Frontend Frameworks.

Covers:
1. Semantic HTML5 & WCAG 2.1 AAA Accessibility:
   - Landmarks, ARIA roles, form validation, native elements vs generic divs, SEO metadata.
2. Advanced CSS3, Architecture & Design Tokens:
   - CSS Grid (auto-fit, minmax, subgrid, grid-template-areas), Flexbox, Container Queries (@container).
   - Modern Aesthetics: Glassmorphism (backdrop-filter: blur), HSL Design Tokens, CSS Custom Properties, Dark Mode.
   - GPU-accelerated Keyframe Animations (@keyframes), 3D Transforms, will-change, smooth transitions.
3. JavaScript ES6+ & TypeScript:
   - Event Loop, Promises, Async/Await, WebSockets, Web Workers, Proxy/Reflect, Strict Typing.
4. Web APIs, Canvas 2D & WebGL:
   - Hardware-accelerated 2D Canvas rendering, Shaders (GLSL), RequestAnimationFrame game/render loops.
5. Web Performance & Core Web Vitals:
   - LCP, FID/INP, CLS, Critical CSS, Lazy Loading (images & iframes), Minification, Tree-Shaking.

Appends seamlessly to data/cyber_web_knowledge/shards/ without overwriting existing shards.
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


WEB_FRONTEND_DEEP_KNOWLEDGE = [
    # 1. Semantic HTML5 & Modern Accessibility (a11y)
    r"""# Semantic HTML5, Microdata & WCAG 2.1 AAA Accessibility Architecture:

## 1. Landmark Hierarchie & Document Outline
Ein barrierefreies und suchmaschinenoptimiertes (SEO) HTML5-Dokument verzichtet auf 'Div-Suppe' und nutzt semantische Landmarks:
```html
<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="State-of-the-Art Web Architecture & UI/UX">
  <title>Modern Web Systems</title>
</head>
<body>
  <header role="banner" class="site-header">
    <nav role="navigation" aria-label="Hauptnavigation">
      <ul>
        <li><a href="#dashboard" aria-current="page">Dashboard</a></li>
        <li><a href="#analytics">Analytics</a></li>
      </ul>
    </nav>
  </header>
  <main id="main-content" role="main">
    <article class="content-card">
      <header>
        <h1>Moderne Frontend-Architektur</h1>
        <p class="meta">Veröffentlicht am <time datetime="2026-08-30">30. August 2026</time></p>
      </header>
      <section aria-labelledby="section-grid">
        <h2 id="section-grid">CSS Grid & Container Queries</h2>
        <p>Modulare Komponenten passen sich an ihren Eltern-Container an.</p>
      </section>
    </article>
    <aside role="complementary" aria-label="Systemstatus">
      <h2>Live Telemetrie</h2>
    </aside>
  </main>
  <footer role="contentinfo">
    <p>&copy; 2026 Multi-Granular Engine. Alle Rechte vorbehalten.</p>
  </footer>
</body>
</html>
```

## 2. Barrierefreie Formular-Architektur & Native Validierung
- Jedes Input-Element MUSS ein explizites `<label for="id">` besitzen.
- Fehlermeldungen werden via `aria-describedby="error-id"` und `aria-invalid="true"` mit dem Feld verknüpft.
- Dynamische Statusmeldungen nutzen `aria-live="polite"` (bzw. `"assertive"` bei kritischen Alerts).""",

    # 2. Modern CSS3: Grid, Glassmorphism, HSL Design Tokens & Animations
    r"""# Advanced CSS3: Design Tokens, Glassmorphism, Responsive Grid & Micro-Animations:

## 1. Hierarchisches HSL Design Token System (:root Variables)
```css
:root {
  /* HSL Color Tokens: Hue, Saturation, Lightness */
  --primary-h: 217;
  --primary-s: 91%;
  --primary-l: 60%;
  --primary: hsl(var(--primary-h) var(--primary-s) var(--primary-l));
  --primary-glow: hsla(var(--primary-h), var(--primary-s), var(--primary-l), 0.35);

  --surface-bg: hsl(222, 47%, 11%);
  --surface-card: hsla(217, 33%, 17%, 0.75);
  --border-glass: hsla(217, 33%, 35%, 0.4);

  /* Typography & Spacing Scale */
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --space-xs: 0.25rem;
  --space-sm: 0.5rem;
  --space-md: 1.0rem;
  --space-lg: 1.5rem;
  --space-xl: 2.5rem;

  /* Transitions */
  --ease-spring: cubic-bezier(0.175, 0.885, 0.32, 1.275);
  --duration-fast: 150ms;
  --duration-normal: 250ms;
}

/* 2. Glassmorphism & GPU Accelerated Card Component */
.glass-panel {
  background: var(--surface-card);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--border-glass);
  border-radius: 1rem;
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
  transition: transform var(--duration-normal) var(--ease-spring),
              box-shadow var(--duration-normal) ease;
  will-change: transform;
}

.glass-panel:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 40px 0 var(--primary-glow);
}

/* 3. Responsive CSS Grid mit Subgrid & Container Queries */
.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr));
  gap: var(--space-lg);
  container-type: inline-size;
}

@container (min-width: 600px) {
  .card-wide {
    grid-column: span 2;
  }
}

/* 4. Pulse & Shimmer Animations */
@keyframes pulseGlow {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.6; transform: scale(1.05); }
}

.pulse-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: hsl(142, 76%, 45%);
  box-shadow: 0 0 12px hsl(142, 76%, 45%);
  animation: pulseGlow 2s infinite ease-in-out;
}
```""",

    # 3. Modern JavaScript ES6+, TypeScript & Reactive State
    r"""# TypeScript, Asynchronous State & Real-Time WebSockets:

## 1. Vollständiger Reactive State Store in TypeScript (Zero Dependencies)
```typescript
type Listener<T> = (state: T) => void;

export class ObservableStore<T extends Record<string, any>> {
  private state: T;
  private listeners: Set<Listener<T>> = new Set();

  constructor(initialState: T) {
    this.state = Object.freeze({ ...initialState });
  }

  public getState(): Readonly<T> {
    return this.state;
  }

  public setState(partial: Partial<T> | ((prev: T) => Partial<T>)): void {
    const nextUpdates = typeof partial === 'function' ? partial(this.state) : partial;
    this.state = Object.freeze({ ...this.state, ...nextUpdates });
    this.notify();
  }

  public subscribe(listener: Listener<T>): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  private notify(): void {
    for (const listener of this.listeners) {
      listener(this.state);
    }
  }
}
```

## 2. Hardware-Accelerated 2D Canvas Chart Renderer (High FPS)
```javascript
export function renderHighPerformanceChart(canvas, dataPoints, strokeColor = '#06b6d4') {
  const ctx = canvas.getContext('2d', { alpha: true });
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  if (!dataPoints || dataPoints.length < 2) return;

  ctx.beginPath();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = strokeColor;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';

  const stepX = w / (dataPoints.length - 1);
  const minVal = Math.min(...dataPoints);
  const maxVal = Math.max(...dataPoints);
  const range = (maxVal - minVal) || 1;

  dataPoints.forEach((val, i) => {
    const x = i * stepX;
    const y = h - ((val - minVal) / range) * (h - 20) - 10;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke();
}
```"""
]


def run_web_frontend_ingestion(
    output_dir: str = "/home/benjamin/Bilder/data/cyber_web_knowledge/shards",
    vocab_file: str = "/home/benjamin/Bilder/data/vocab_65k.json",
    max_tokens_per_shard: int = 500_000,
    target_shards: int = 200,
) -> int:
    global STOP_REQUESTED

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    print("=" * 80, flush=True)
    print("🎨 MASSIVE WEB FRONTEND, HTML5, CSS3 & UI/UX SPECIALIST PIPELINE (7B PRE-TRAINING)", flush=True)
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
        print(f"  💾 [WEB/UI/UX Shard {shard_count:04d}] {len(buffer_tokens):,} Tokens | Gesamt: {total_tokens_written:,} Tokens -> {os.path.basename(shard_path)}", flush=True)
        buffer_tokens = []

    # 1. Ingest Deep Frontend & CSS Architecture Standards
    print("\n📚 [Quelle 1/2] Tokenisiere Semantisches HTML5, Modern CSS3, Glassmorphism & UI/UX Standards...", flush=True)
    for doc in WEB_FRONTEND_DEEP_KNOWLEDGE:
        formatted = f"### Web Engineering & UI/UX Architektur:\n{doc.strip()}\n\n"
        buffer_tokens.extend(tokenizer.encode(formatted))
        if len(buffer_tokens) >= max_tokens_per_shard:
            flush_shard()

    # 2. Ingest Frontend & Web Instructions (HTML, CSS, JS, TS, React, Vue, Canvas)
    print("\n🌐 [Quelle 2/2] Streame HTML5, CSS3, JavaScript ES6+, TypeScript & Web APIs...", flush=True)
    try:
        code_ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
        web_count = 0
        for item in code_ds:
            if STOP_REQUESTED or shard_count >= target_shards:
                break
            instr = item.get("instruction", "")
            resp = item.get("output", "")
            if not instr or not resp:
                continue

            keywords = ["html", "css", "javascript", "typescript", "react", "vue", "dom", "canvas", "grid", "flexbox", "style", "ui", "ux", "responsive", "frontend", "web"]
            is_relevant = any(kw in instr.lower() or kw in resp.lower() for kw in keywords)
            if not is_relevant:
                continue

            formatted = f"### Benutzer (Web Development & UI/UX):\n{instr}\n\n### Assistent (Master Frontend Engineer):\n<think>\nAnalysiere HTML5-Semantik, CSS-Layouts, Responsive Design und Best Practices:\n</think>\n{resp}\n\n"
            buffer_tokens.extend(tokenizer.encode(formatted))
            web_count += 1

            if len(buffer_tokens) >= max_tokens_per_shard:
                flush_shard()

            if web_count % 2000 == 0:
                print(f"  ⚙️ [Web-Frontend] {web_count:,} Web-Instruktionen verarbeitet (Shards: {shard_count})", flush=True)

        print(f"✅ Frontend & Web Dialoge abgeschlossen ({web_count:,} verarbeitet).", flush=True)
    except Exception as e:
        print(f"⚠️ Hinweis bei Web Stream: {e}", flush=True)

    flush_shard()
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 Web Frontend Specialist Ingestion abgeschlossen! Gesamte Shards: {shard_count} (~{total_tokens_written:,} Tokens)", flush=True)
    print("=" * 80, flush=True)
    return shard_count


if __name__ == "__main__":
    run_web_frontend_ingestion()
