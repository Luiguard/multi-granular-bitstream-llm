#!/usr/bin/env python3
"""
Specialized Ingestion Pipeline for Rate-Limits, Throttling Mitigation,
Anti-Scraping Resilience, Multi-Source Fallbacks, and Web Data Engineering.

Teaches the 7.45B MoE Model:
1. HTTP Throttling Status Codes (429, 403, 503, 504, 402).
2. Rate-Limiting Algorithms (Token Bucket, Leaky Bucket, Sliding Window Counter).
3. Resilience & Bypass Strategies:
   - Exponential Backoff with Decorrelated Jitter
   - Multi-Engine Fallback Cascades (DuckDuckGo -> Wikipedia -> ArXiv -> Wayback Machine)
   - User-Agent & Browser Fingerprint Pool Rotation
   - Archive & Mirror Retrievals (Wayback Machine / WebCache)
   - Distributed Scraping & IP Rotation Pool Architectures
   - Cache-First (LRU / TTL) Strategy

Appends to data/cyber_web_knowledge/shards/.
"""

import glob
import json
import os
import sys
import time

sys.path.insert(0, "/home/benjamin/Bilder")

from datasets import load_dataset
from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.bitstream import BitstreamEncoder

TARGET_DIR = "/home/benjamin/Bilder/data/cyber_web_knowledge/shards"
os.makedirs(TARGET_DIR, exist_ok=True)

VOCAB_PATH = "/home/benjamin/Bilder/data/vocab_65k.json"
if not os.path.exists(VOCAB_PATH):
    VOCAB_PATH = "/home/benjamin/Bilder/vocab.json"

VOCAB = MultiGranularVocabulary.load_json(VOCAB_PATH)
TOKENIZER = ViterbiTokenizer(VOCAB)
ENCODER = BitstreamEncoder(vocab_size=VOCAB.size, bit_width=16)

SHARD_SIZE = 500_000
TOTAL_TARGET_SHARDS = 340  # 300 exist, adding 40 deep rate-limit engineering shards


def get_next_shard_index(directory: str) -> int:
    existing = glob.glob(os.path.join(directory, "cyber_web_shard_*.mgbs"))
    indices = []
    for f in existing:
        try:
            base = os.path.basename(f)
            num = int(base.replace("cyber_web_shard_", "").replace(".mgbs", ""))
            indices.append(num)
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def generate_rate_limiting_curriculum():
    """Deep algorithmic and architectural training patterns for rate limiting and mitigation."""
    modules = [
        # Module 1: HTTP 429 & Exponential Backoff with Jitter
        r"""# Architektur-Leitfaden: HTTP 429 Too Many Requests & Exponential Backoff mit Jitter

## 1. Das Problem: Server-seitige Ratenbegrenzung (Rate Limiting)
Webserver und APIs schützen ihre Infrastruktur durch Algorithmen wie:
- **Token Bucket**: Feste Token-Nachfüllrate; Anfragen verbrauchen Tokens.
- **Leaky Bucket**: Konstante Abflussrate; Puffer für kurzzeitige Spikes.
- **Sliding Window Log / Counter**: Zeitfenster-basierte Drosselung pro IP / API-Key.

Wenn das Limit überschritten wird, antwortet der Server mit **HTTP 429 (Too Many Requests)** oder **HTTP 503 (Service Unavailable)**, häufig begleitet vom Header `Retry-After: <Sekunden>`.

## 2. Die Lösung: Exponential Backoff mit Random Jitter
Stumpfes Wiederholen führt zum "Thundering Herd Problem" (Serverüberlastung). Die mathematisch optimale Strategie lautet **Full Jitter**:

$$\text{Sleep}(i) = \text{Uniform}(0, \min(T_{\max}, T_{\text{base}} \cdot 2^i))$$

### Python Referenz-Implementierung:
```python
import time
import random
import urllib.request
import urllib.error

def fetch_with_backoff(url: str, max_retries: int = 4, base_delay: float = 1.0, max_delay: float = 30.0):
    delay = base_delay
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ResilientCrawler/2.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 504, 502):
                retry_after = e.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_time = float(retry_after)
                else:
                    sleep_time = random.uniform(0.5, min(max_delay, delay))
                print(f"[Warnung] HTTP {e.code} empfangen. Warte {sleep_time:.2f}s (Versuch {attempt+1}/{max_retries})...")
                time.sleep(sleep_time)
                delay *= 2.0
            else:
                raise e
    raise RuntimeError("Maximale Wiederholungen überschritten.")
```""",

        # Module 2: Multi-Source Fallback Cascade
        r"""# Architektur-Leitfaden: Multi-Engine & Multi-Source Fallback Kaskade

## 1. Das Problem: Ausfall oder Blockade einer primären Datenquelle
Wenn eine Suchmaschine (z.B. DuckDuckGo) temporäre CAPTCHAs anzeigt oder IP-Adressen sperrt, darf die autonome KI-Pipeline nicht stoppen.

## 2. Die Lösung: Kaskadierende Failover-Architektur
Die Anfrage wandert hierarchisch durch voneinander unabhängige Primär- und Sekundärquellen:

```mermaid
flowchart TD
    Q[Suchanfrage] --> E1[1. DuckDuckGo HTML Lite]
    E1 -->|Erfolg| Ret[Suchergebnisse]
    E1 -->|HTTP 429/403| E2[2. Wikipedia REST API DE & EN]
    E2 -->|Erfolg| Ret
    E2 -->|Unvollständig| E3[3. ArXiv Open Science API]
    E3 -->|Erfolg| Ret
    E3 -->|Kein Treffer| E4[4. Wayback Machine Archive Snapshot]
    E4 --> Ret
```

### Vorteile der Multi-Source Kaskade:
1. **Zero Single-Point-of-Failure**: Der Ausfall eines Anbieters stoppt die Inferenz nicht.
2. **Quellen-Diversität**: Kombination aus aktuellen Web-Snippets, enzyklopädischem Tiefenwissen und wissenschaftlichen Primärquellen.
3. **Bandbreiten-Schonung**: Anfragen werden gestreut.""",

        # Module 3: Anti-Scraping WAF Bypasses & Fingerprint Rotation
        r"""# Architektur-Leitfaden: User-Agent-Rotation, TLS Fingerprinting & WAF-Resilienz

## 1. Erkennungsmechanismen moderner Web Application Firewalls (Cloudflare, Akamai, Datadome)
- **Static User-Agent Matching**: Erkennung von veralteten oder fehlenden Client-Headern (`python-requests`, `curl`, `Java/1.8`).
- **HTTP/2 & TLS Client Hello Fingerprinting (JA3 / JA4)**: Überprüfung von Cipher Suites und Extension-Reihenfolgen.
- **Header Order & Case Consistency**: Browser wie Chrome und Firefox senden spezifische Header-Reihenfolgen (`Sec-CH-UA`, `Sec-Fetch-Dest`).

## 2. Robuste Gegenmaßnahmen für Datenpipelines:
1. **Header-Pool Rotation**: Dynamischer Wechsel zwischen modernen Browser-Signaturen (Chrome 128+, Firefox 129+, Safari 17+).
2. **Referer & Origin Spoofing**: Setzen von kontextuell plausiblen Referern (z.B. `https://duckduckgo.com/` oder `https://www.google.com/`).
3. **Session- & Cookie-Persistenz**: Beibehalten von Session-Tokens zur Vermeidung wiederholter Authentifizierungs-Challenges.
4. **Wayback Machine Fallback**: Bei unüberwindbaren WAF-Blockaden (`HTTP 403`) Abruf des letzten gecachten Snapshots über `https://archive.org/wayback/available?url=...`.""",

        # Module 4: Cache-First Architecture & Deduplication
        r"""# Architektur-Leitfaden: Cache-First In-Memory & TTL Deduplizierung

## 1. Das Problem: Redundante Netzwerkanfragen
Häufige Anfragen nach denselben Konzepten verbrauchen unnötige Bandbreite und provozieren Rate Limits.

## 2. Die Lösung: Zweistufiges Cache-Design (Memory LRU + Disk Persistent)
- **Time-to-Live (TTL)**: Temporäres Vorhalten von Webseiten (z.B. 15 Minuten bis 24 Stunden).
- **Inverted Hash Index**: Schneller $O(1)$ Lookup über URL-Hashes (SHA-256).
- **Conditional GET Requests**: Nutzung von `If-None-Match` (ETag) und `If-Modified-Since` zur Minimierung des Datenübertragungsvolumens (HTTP 304 Not Modified)."""
    ]

    for mod in modules:
        yield f"### System:\nDu bist ein Master Data & Distributed Systems Architect. Beherrsche Web-Scraping, Ratenbegrenzungen (HTTP 429), Fallback-Kaskaden und Anti-Throttling Resilienz im Detail.\n\n{mod}\n\n"


def main():
    print("=" * 80)
    print("🌐 RATE-LIMIT MITIGATION & ANTI-THROTTLING ARCHITECTURE INGESTION")
    print(f"📁 Ziel-Verzeichnis: {TARGET_DIR}")
    print("=" * 80)

    start_idx = get_next_shard_index(TARGET_DIR)
    print(f"  ⏭️ Starte ab Shard-Index {start_idx:04d} (Ziel: {TOTAL_TARGET_SHARDS})...")

    current_shard_idx = start_idx
    shard_tokens = []
    total_tokens = 0

    def flush_shard():
        nonlocal current_shard_idx, shard_tokens, total_tokens
        if not shard_tokens:
            return
        filename = f"cyber_web_shard_{current_shard_idx:04d}.mgbs"
        filepath = os.path.join(TARGET_DIR, filename)

        ENCODER.save_to_file(filepath, shard_tokens, raw_byte_count=len(shard_tokens) * 2)
        total_tokens += len(shard_tokens)
        print(f"  💾 [RATE-LIMIT Shard {current_shard_idx:04d}] {len(shard_tokens):,} Tokens -> {filename}", flush=True)
        current_shard_idx += 1
        shard_tokens = []

    # 1. Ingest Theoretical & Algorithmic Modules
    print("\n📚 [Quelle 1/2] Tokenisiere Rate-Limit Algorithmen & Multi-Engine Resilienz...")
    for _ in range(80):
        for pattern in generate_rate_limiting_curriculum():
            tokens = list(TOKENIZER.encode(pattern))
            shard_tokens.extend(tokens)
            if len(shard_tokens) >= SHARD_SIZE:
                flush_shard()
                if current_shard_idx >= TOTAL_TARGET_SHARDS:
                    break
        if current_shard_idx >= TOTAL_TARGET_SHARDS:
            break

    # 2. Ingest Networking & Distributed Systems Dataset
    if current_shard_idx < TOTAL_TARGET_SHARDS:
        print("\n🌐 [Quelle 2/2] Streame Distributed Networking & Systems Code (Evol-Instruct-Code)...")
        try:
            code_ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
            for item in code_ds:
                instr = item.get("instruction", "")
                resp = item.get("output", "")
                if not instr or not resp:
                    continue

                keywords = ["http", "rate limit", "crawler", "scrape", "request", "proxy", "async", "retry", "socket", "network", "backoff", "cache", "token bucket", "api"]
                if any(kw in instr.lower() or kw in resp.lower() for kw in keywords):
                    text = f"### Benutzer (Netzwerk & Scraping Problem):\n{instr}\n\n### Assistent (Master Systems Architect):\n<think>\nAnalysiere Ratenbegrenzungen, HTTP Statuscodes, Fallback-Strategien und optimiere die Netzwerk-Resilienz:\n</think>\n{resp}\n\n"
                    tokens = list(TOKENIZER.encode(text))
                    shard_tokens.extend(tokens)
                    if len(shard_tokens) >= SHARD_SIZE:
                        flush_shard()
                        if current_shard_idx >= TOTAL_TARGET_SHARDS:
                            break
        except Exception as e:
            print(f"⚠️ Hinweis bei Dataset-Stream: {e}")

    flush_shard()
    print("=" * 80)
    print(f"🎉 Rate-Limit & Resilience Ingestion abgeschlossen! Gesamte Shards: {current_shard_idx} (~{total_tokens:,} Tokens)")
    print("=" * 80)


if __name__ == "__main__":
    main()
