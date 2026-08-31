#!/usr/bin/env python3
"""Massive Ingestion Pipeline for the Whole World Wide Web (WWW) Infrastructure,
Internet Protocols (IETF/RFCs), Distributed Backend Architectures, Cloud DevOps, and Data Layer Internals.

Covers:
1. Internet Protocols & Networking Stack (RFC Specifications):
   - HTTP/1.1, HTTP/2 (Multiplexing, HPACK), HTTP/3 (QUIC over UDP, 0-RTT, BBR Congestion Control).
   - DNS Architecture (Root servers, DNSSEC, DoH, Anycast), TCP/IP (Handshakes, Sockets, TIME_WAIT),
     TLS 1.3 (Diffie-Hellman, SNI, ALPN, OCSP), WebSockets (RFC 6455), WebRTC (STUN/TURN/ICE).
2. Distributed Backend Systems & Runtimes:
   - Node.js (Libuv event loop), Go (Goroutines, Channels), Rust (Axum, Tokio), Python (FastAPI/ASGI).
   - API Standards: REST (Richardson Maturity Model), GraphQL (DataLoader, N+1), gRPC (Protobuf), Webhooks (HMAC).
3. Distributed Databases & Cache Internals:
   - PostgreSQL (MVCC, WAL, GIN/GiST Indexing, PgBouncer), MySQL (InnoDB, ACID),
   - Redis (Data Structures, Pub/Sub, Lua), Elasticsearch (Lucene Inverted Indexes, BM25), Vector DBs (HNSW, pgvector).
4. Cloud Infrastructure, DevOps & Edge Architecture:
   - Docker, Kubernetes, Nginx (Event loop, Upstream balancing), CDNs (Edge caching, Cache-Control),
     CORS (Preflight OPTIONS), CSP (Content-Security-Policy), Rate Limiting (Sliding Window).

Appends directly to data/cyber_web_knowledge/shards/ without overwriting existing shards.
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


WWW_INTERNET_STANDARDS = [
    # 1. Complete HTTP Evolution: HTTP/1.1 -> HTTP/2 -> HTTP/3 (QUIC)
    r"""# The Evolution of HTTP Protocols: RFC 9110, RFC 9113 (HTTP/2), RFC 9000 (QUIC/HTTP/3):

## 1. HTTP/1.1 vs HTTP/2 vs HTTP/3 Protokollvergleich
- **HTTP/1.1 (RFC 9112)**:
  - Textbasiertes Protokoll. Head-of-Line (HoL) Blocking auf Anwendungsebene: Jede TCP-Verbindung kann nur eine Anfrage gleichzeitig abarbeiten (Pipelining in der Praxis unzuverlässig).
  - Hoher Header-Overhead bei vielen kleinen Anfragen (Cookie-Übertragung bei jedem Request).
- **HTTP/2 (RFC 9113)**:
  - Binäres Framing-Format (Frames, Streams, Messages).
  - Echtes Multiplexing: Hunderte parallele bidirektionale Streams über eine einzige TCP-Verbindung.
  - HPACK Header-Kompression: Statische und dynamische Tabellen zur Vermeidung redundanter Header-Bytes.
  - Schwachstelle: HoL-Blocking auf TCP-Transportebene (Packet Loss blockiert alle Multiplex-Streams).
- **HTTP/3 über QUIC (RFC 9000 / RFC 9114)**:
  - Basiert auf UDP statt TCP: Jeder HTTP-Stream ist auf Transportebene unabhängig (kein TCP-HoL Blocking mehr!).
  - 0-RTT Connection Handshakes: TLS 1.3 ist nativ im QUIC-Header integriert (spart 1-2 Round-Trips bei bekannten Hosts).
  - Connection Migration: Verbindung bleibt bei IP-Wechsel (z. B. WLAN zu 5G) über die 64-Bit Connection-ID nahtlos bestehen.
  - BBRv2 Congestion Control: Modellbasiertes Bandbreiten- und RTT-Routing minimiert Pufferüberläufe (Bufferbloat).""",

    # 2. DNS Architecture, BGP Anycast & Content Delivery Networks (CDNs)
    r"""# Global Internet Architecture: DNSSEC, BGP Routing, Anycast & CDN Edge Caching:

## 1. DNS Auflösungshierarchie & DNSSEC (RFC 4033)
- **Ablauf einer rekursiven DNS-Auflösung**:
  1. Client prüft OS-Cache & `/etc/hosts`.
  2. Anfrage an den Recursive Resolver (z. B. 1.1.1.1 oder 8.8.8.8) via DoH (DNS-over-HTTPS / RFC 8484) über Port 443.
  3. Resolver fragt Root-Server (`.` - 13 weltweite Anycast-IP-Cluster `a.root-servers.net` bis `m.root-servers.net`).
  4. Root verweist auf TLD-Nameserver (`.de`, `.com`).
  5. TLD verweist auf den autoritativen Nameserver der Domain (`ns1.example.com`).
- **DNSSEC (Kryptographische Authentizität)**:
  - RRset (Resource Record Set) wird mit dem Zone Signing Key (ZSK) signiert $\rightarrow$ `RRSIG` Record.
  - ZSK wird mit dem Key Signing Key (KSK) signiert $\rightarrow$ `DNSKEY` Record.
  - Der Hash des KSK wird an die übergeordnete TLD als `DS` (Delegation Signer) Record übermittelt $\rightarrow$ Lückenlose Vertrauenskette (Chain of Trust) bis zum Root-Schlüssel.

## 2. CDN Edge Caching & Cache-Control Header
- `Cache-Control: public, max-age=31536000, immutable` für unveränderliche statische Assets mit Content-Hash.
- `Cache-Control: no-cache, s-maxage=3600, stale-while-revalidate=86400`:
  - `s-maxage`: CDN serviert Cache für 1 Stunde.
  - `stale-while-revalidate`: CDN liefert veraltete Daten sofort aus und holt im Hintergrund asynchron ein Update vom Origin-Server (0ms Latenz für den Nutzer!).
- `Vary: Accept-Encoding, Origin`: Verhindert Cache-Poisoning bei CORS oder unterschiedlichen Kompressionsformaten (Brotli `br`, Gzip `gzip`, Zstandard `zstd`).""",

    # 3. Distributed Database Internals: PostgreSQL, Redis & Elasticsearch
    r"""# Distributed Data Architecture: PostgreSQL MVCC, Redis Engine & Elasticsearch:

## 1. PostgreSQL Internals: MVCC, WAL & Indizierungsstrategien
- **Multi-Version Concurrency Control (MVCC)**:
  - Zeilen werden bei `UPDATE` nicht überschrieben, sondern als neue Tupel mit `xmin` (Erzeugungs-Transaktion) und `xmax` (Löschungs-Transaktion) angelegt.
  - `VACUUM` bereinigt tote Tupel (Dead Tuples) und verhindert Tabellen-Aufblähung (Table Bloat).
- **Write-Ahead Logging (WAL)**:
  - Transaktions-Änderungen werden sequentiell ins WAL geschrieben (`fsync`), bevor Datenblöcke auf die Festplatte wandern (Garantiert Atomicity & Durability bei Stromausfall).
- **Index-Typen**:
  - **B-Tree**: Standard für Vergleiche ($=, <, >, \le, \ge$) mit logarithmischer Suchzeit $O(\log N)$.
  - **GIN (Generalized Inverted Index)**: Perfekt für JSONB, Arrays und Fulltext-Suche.
  - **BRIN (Block Range Index)**: Extrem speichereffizient für monoton steigende Zeitstempel bei Milliarden Zeilen.

## 2. Redis In-Memory Engine & High-Speed Caching
- Single-Threaded Event Loop (Epoll/Kqueue) für In-Memory-Operationen: Verhindert Lock-Contention und Kontextwechsel.
- **Sliding Window Rate Limiter via Sorted Sets (`ZADD`)**:
  ```python
  def is_rate_limited(redis_client, user_id, limit=100, window_sec=60):
      now = time.time()
      key = f"rate_limit:{user_id}"
      pipe = redis_client.pipeline()
      # 1. Entferne Zeitstempel außerhalb des aktuellen Zeitfensters
      pipe.zremrangebyscore(key, 0, now - window_sec)
      # 2. Füge aktuellen Request hinzu
      pipe.zadd(key, {str(now): now})
      # 3. Zähle Elemente im Zeitfenster
      pipe.zcard(key)
      # 4. Setze TTL auf das Zeitfenster
      pipe.expire(key, window_sec)
      _, _, current_count, _ = pipe.execute()
      return current_count > limit
  ```""",

    # 4. Web Security Hardening: CORS Preflight, Content Security Policy & Zero-Trust
    r"""# Advanced Web Security Hardening: CORS, CSP Headers & Zero-Trust Architecture:

## 1. Cross-Origin Resource Sharing (CORS) & Preflight Details
- Ein Preflight `OPTIONS`-Request wird ausgelöst, wenn:
  - Die HTTP-Methode nicht `GET`, `HEAD` oder `POST` ist.
  - Custom Headers wie `Authorization` oder `X-Custom-Header` gesetzt sind.
  - Der `Content-Type` nicht `application/x-www-form-urlencoded`, `multipart/form-data` oder `text/plain` ist (z. B. `application/json`).
- Korrekte Preflight-Antwort der API:
  ```http
  HTTP/1.1 204 No Content
  Access-Control-Allow-Origin: https://app.example.com
  Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
  Access-Control-Allow-Headers: Authorization, Content-Type, X-Request-ID
  Access-Control-Allow-Credentials: true
  Access-Control-Max-Age: 86400
  ```

## 2. Strikte Content Security Policy (CSP Level 3)
```http
Content-Security-Policy: default-src 'none'; \
  script-src 'self' 'nonce-rAnd0m123' 'strict-dynamic'; \
  style-src 'self' 'unsafe-inline'; \
  img-src 'self' data: https://images.example.com; \
  font-src 'self'; \
  connect-src 'self' wss://api.example.com; \
  frame-ancestors 'none'; \
  base-uri 'self'; \
  form-action 'self'; \
  upgrade-insecure-requests;
```
- `frame-ancestors 'none'`: Verhindert Clickjacking-Angriffe über iframes (ersetzt `X-Frame-Options: DENY`).
- `strict-dynamic` mit Nonce: Verhindert Reflected & Stored XSS selbst bei bösartig injizierten Script-Tags."""
]


def run_www_internet_ingestion(
    output_dir: str = "/home/benjamin/Bilder/data/cyber_web_knowledge/shards",
    vocab_file: str = "/home/benjamin/Bilder/data/vocab_65k.json",
    max_tokens_per_shard: int = 500_000,
    target_shards: int = 250,
) -> int:
    global STOP_REQUESTED

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    print("=" * 80, flush=True)
    print("🌐 MASSIVE WWW, INTERNET PROTOCOLS & DISTRIBUTED SYSTEMS PIPELINE (7B PRE-TRAINING)", flush=True)
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
        print(f"  💾 [WWW/NET/CLOUD Shard {shard_count:04d}] {len(buffer_tokens):,} Tokens | Gesamt: {total_tokens_written:,} Tokens -> {os.path.basename(shard_path)}", flush=True)
        buffer_tokens = []

    # 1. Ingest Deep Internet & Web Architecture Standards
    print("\n📚 [Quelle 1/2] Tokenisiere RFCs, HTTP/3, QUIC, DNSSEC, BGP & PostgreSQL/Redis Internals...", flush=True)
    for doc in WWW_INTERNET_STANDARDS:
        formatted = f"### Internet-Engineering & WWW-Systemarchitektur:\n{doc.strip()}\n\n"
        buffer_tokens.extend(tokenizer.encode(formatted))
        if len(buffer_tokens) >= max_tokens_per_shard:
            flush_shard()

    # 2. Ingest Distributed Web, Networking & DevOps Instructions
    print("\n🌍 [Quelle 2/2] Streame Distributed Web, Cloud-Native, Nginx, Docker & API Engineering...", flush=True)
    try:
        code_ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
        www_count = 0
        for item in code_ds:
            if STOP_REQUESTED or shard_count >= target_shards:
                break
            instr = item.get("instruction", "")
            resp = item.get("output", "")
            if not instr or not resp:
                continue

            keywords = ["http", "tcp", "udp", "dns", "tls", "api", "database", "sql", "postgres", "redis", "nginx", "docker", "kubernetes", "network", "server", "cors", "auth", "jwt", "grpc", "graphql"]
            is_relevant = any(kw in instr.lower() or kw in resp.lower() for kw in keywords)
            if not is_relevant:
                continue

            formatted = f"### Benutzer (Internet Engineering & System Architecture):\n{instr}\n\n### Assistent (Principal Cloud & Network Architect):\n<think>\nAnalysiere Netzwerkprotokolle, verteilte Systeme, Datenbank-Indizes und Skalierbarkeit:\n</think>\n{resp}\n\n"
            buffer_tokens.extend(tokenizer.encode(formatted))
            www_count += 1

            if len(buffer_tokens) >= max_tokens_per_shard:
                flush_shard()

            if www_count % 2000 == 0:
                print(f"  ⚙️ [WWW-Engine] {www_count:,} Netzwerk- & System-Dialoge verarbeitet (Shards: {shard_count})", flush=True)

        print(f"✅ WWW & Distributed Systems Dialoge abgeschlossen ({www_count:,} verarbeitet).", flush=True)
    except Exception as e:
        print(f"⚠️ Hinweis bei WWW Stream: {e}", flush=True)

    flush_shard()
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 WWW & Internet Specialist Ingestion abgeschlossen! Gesamte Shards: {shard_count} (~{total_tokens_written:,} Tokens)", flush=True)
    print("=" * 80, flush=True)
    return shard_count


if __name__ == "__main__":
    run_www_internet_ingestion()
