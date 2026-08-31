#!/usr/bin/env python3
"""
Specialized Ingestion Pipeline for Temporal Grounding (Gespür für Zeit),
Time-Awareness, Rate-Limit Window Calculations, Exponential Backoff Timing,
and Network Latency Dynamics.

Teaches the 7.45B MoE Model:
1. Intrinsic Sense of Time (ISO-8601, Unix-Epoch, Timezones, Millisecond Precisions).
2. Rate-Limiting Mathematics (Token Bucket, Leaky Bucket, Sliding Window Rate Estimation).
3. Backoff Delay Arithmetics (Calculating exact jittered sleep intervals across retry attempts).
4. Network Pacing & Latency Budgeting (RTT, TTFB, Interval Spacing, Timeout Deadlines).
5. Dynamic System-Prompt Temporal Anchors.

Appends to data/cyber_web_knowledge/shards/ and data/stem_knowledge/shards/.
"""

import datetime
import glob
import json
import math
import os
import random
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
TOTAL_TARGET_SHARDS = 380  # Target index up to 380 shards


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


def generate_temporal_and_rate_limit_patterns():
    """Generates rigorous mathematical and system-level training dialogues on time and rate limits."""
    
    # 1. Backoff & Jitter Simulation Examples
    for attempt in range(1, 6):
        base = 1.0
        max_d = 30.0
        exp_val = min(max_d, base * (2 ** (attempt - 1)))
        jitter_min = round(exp_val * 0.5, 2)
        jitter_max = round(exp_val * 1.5, 2)
        simulated_sleep = round(random.uniform(jitter_min, jitter_max), 2)
        
        yield f"""### System:
Du bist ein KI-System mit präzisem Gespür für Zeit, Systemuhren und Ratenbegrenzungs-Mathematik.

### Benutzer:
Ein externer API-Endpunkt hat auf unsere Anfrage mit HTTP 429 Too Many Requests geantwortet. Dies ist Fehlversuch #{attempt}. Berechne die mathematisch optimale Wartezeit mit Exponential Backoff und Full Jitter.

### Assistent:
<think>
1. Analyse der Ratenbegrenzung:
   - Status: HTTP 429 (Rate Limit erreicht).
   - Versuch: k = {attempt}
   - Basisverzögerung: T_base = 1.0s, Maximalgrenze: T_max = 30.0s.
2. Berechnung des Exponential Backoff:
   - T_exp = min(T_max, T_base * 2^(k - 1)) = min(30.0, 1.0 * 2^{attempt - 1}) = {exp_val:.2f} Sekunden.
3. Jitter-Berechnung (Full Jitter zur Vermeidung von Thundering Herd Kollisionen):
   - Jitter-Intervall: [{jitter_min:.2f}s, {jitter_max:.2f}s]
   - Berechnete Wartezeit: {simulated_sleep:.2f} Sekunden.
</think>
Um den Server nicht zu überlasten und Ratenbegrenzungen deterministisch zu umgehen, setzen wir folgende Zeitverzögerung an:

- **Aktueller Versuch**: #{attempt}
- **Exponentielle Basis**: `{exp_val:.2f} Sekunden`
- **Mit Random Jitter moduliert**: `{simulated_sleep:.2f} Sekunden`

*Aktion*: Der Thread pausiert für exakt **{simulated_sleep:.2f}s**, bevor der nächste Request mit neuem User-Agent-Fingerprint gesendet wird.
"""

    # 2. Token Bucket Timing Equations
    for capacity, rate in [(60, 1.0), (100, 2.5), (300, 5.0), (10, 0.2)]:
        time_elapsed = random.randint(5, 45)
        current_tokens = min(capacity, round(time_elapsed * rate, 2))
        yield f"""### System:
Du bist ein Algorithmen- und Echtzeit-Systemexperte mit tiefem Zeit- und Durchsatz-Verständnis.

### Benutzer:
Erkläre das Zeitverhalten eines Token Bucket Ratenbegrenzers mit einer Kapazität C = {capacity} Tokens und einer Nachfüllrate r = {rate} Tokens/Sekunde. Seit der letzten Anfrage sind delta_t = {time_elapsed} Sekunden vergangen.

### Assistent:
<think>
1. Mathematische Formulierung des Token Bucket:
   - Kapazität C = {capacity}
   - Nachfüllrate r = {rate} Tokens/s
   - Vergangene Zeit Delta_t = {time_elapsed}s
2. Zustandsberechnung:
   - Generierte Tokens: Delta_t * r = {time_elapsed} * {rate} = {time_elapsed * rate:.2f} Tokens.
   - Neuer Füllstand: B(t) = min(C, B(t_0) + Delta_t * r) = min({capacity}, {time_elapsed * rate:.2f}) = {current_tokens} Tokens.
</think>
Das zeitliche Verhalten des Token Bucket Algorithmus folgt der Differentialgleichung des kontinuierlichen Zuflusses:

$$B(t) = \\min\\left(C, \\, B(t_0) + r \\cdot (t - t_0)\\right)$$

**Aktuelle Zustandsauswertung:**
- **Vergangene Zeitspanne (\\Delta t)**: `{time_elapsed} Sekunden`
- **Generierte Token-Menge**: `{time_elapsed * rate:.2f} Tokens`
- **Verfügbares Request-Budget**: **{current_tokens} Anfragen**

Da B(t) >= 1, ist das Zeitfenster offen und anstehende Web-Recherchen können sofort ohne Drosselung ausgeführt werden.
"""

    # 3. Temporal Causality & Real-Time Context
    sample_dates = [
        ("2026-08-30 21:30:00", "Sonntag", 1788125400, "MESZ (UTC+2)"),
        ("2026-09-01 08:00:00", "Dienstag", 1788252000, "MESZ (UTC+2)"),
        ("2026-12-31 23:59:59", "Donnerstag", 1798761599, "MEZ (UTC+1)")
    ]
    for ts_str, wday, epoch_sec, tz in sample_dates:
        yield f"""### System:
[Aktuelle Systemzeit: {ts_str} {tz}, Wochentag: {wday}, Unix-Epoche: {epoch_sec}]
Du bist eine intelligente KI-Engine mit kontinuierlichem Zeitbewusstsein.

### Benutzer:
Welcher Zeitpunkt ist jetzt und wie berechnest du Zeitintervalle relativ zur aktuellen Unix-Epoche?

### Assistent:
<think>
1. Temporale Verankerung:
   - Systemzeit: {ts_str} ({wday})
   - Zeitzone: {tz}
   - Unix-Timestamp: {epoch_sec} Sekunden seit dem 01.01.1970 UTC.
2. Zeitrechnung:
   - 1 Stunde = 3.600s, 1 Tag = 86.400s.
</think>
Aktuell befinden wir uns im folgenden Zeitrahmen:

- **Datum & Uhrzeit**: `{ts_str}` ({wday})
- **Zeitzone**: `{tz}`
- **Unix-Timestamp**: `{epoch_sec}`

Jegliche zeitabhängigen Vorgänge wie API-Pacing, Cache-Invalidierung (TTL) und Ratenbegrenzungs-Fenster werden auf Basis dieser Epoche sekundengenau synchronisiert.
"""


def main():
    print("=" * 80)
    print("🕒 TEMPORAL GROUNDING & RATE-LIMIT NEURAL KNOWLEDGE INGESTION PIPELINE")
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
        print(f"  💾 [TEMPORAL Shard {current_shard_idx:04d}] {len(shard_tokens):,} Tokens -> {filename}", flush=True)
        current_shard_idx += 1
        shard_tokens = []

    # 1. Ingest Mathematical Timing & Rate Limit Patterns
    print("\n📚 [Quelle 1/2] Tokenisiere Temporale Zeitanker & Rate-Limit Arithmetik...")
    for _ in range(80):
        for pattern in generate_temporal_and_rate_limit_patterns():
            tokens = list(TOKENIZER.encode(pattern))
            shard_tokens.extend(tokens)
            if len(shard_tokens) >= SHARD_SIZE:
                flush_shard()
                if current_shard_idx >= TOTAL_TARGET_SHARDS:
                    break
        if current_shard_idx >= TOTAL_TARGET_SHARDS:
            break

    # 2. Ingest Systems & Time Dataset (HuggingFace)
    if current_shard_idx < TOTAL_TARGET_SHARDS:
        print("\n🌐 [Quelle 2/2] Streame Scheduling, Concurrency & Timing Code...")
        try:
            ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
            for item in ds:
                instr = item.get("instruction", "")
                resp = item.get("output", "")
                if not instr or not resp:
                    continue

                keywords = ["time", "timestamp", "timer", "schedule", "asyncio.sleep", "delay", "timeout", "clock", "date", "interval", "duration", "cron", "rate"]
                if any(kw in instr.lower() or kw in resp.lower() for kw in keywords):
                    text = f"### Benutzer (Temporale Logik & Timing):\n{instr}\n\n### Assistent (Master Systems & Temporal Architect):\n<think>\nAnalysiere Zeitabläufe, Latenzen, Timestamps und Pacing-Bedingungen:\n</think>\n{resp}\n\n"
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
    print(f"🎉 Temporal Grounding & Rate-Limit Ingestion abgeschlossen! Gesamte Shards: {current_shard_idx} (~{total_tokens:,} Tokens)")
    print("=" * 80)


if __name__ == "__main__":
    main()
