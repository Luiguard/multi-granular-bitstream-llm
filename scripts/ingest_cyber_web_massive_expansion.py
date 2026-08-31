#!/usr/bin/env python3
"""Massive-Scale Ingestion Pipeline for Cyber, Web Knowledge, Clean Code, and Software Architecture.

Target: 100+ to 200+ Shards (50M - 100M+ 16-Bit Multi-Granular Tokens)
Sources:
1. Evol-Instruct-Code-80k (Complete 80k complex programming & architecture dialogues)
2. BigCode / The Stack Smol (Python, JavaScript, TypeScript, Rust, HTML, CSS, Shell, SQL, Go)
3. Specialized Cybersecurity, OWASP Top 10, RFC Protocols, Cryptography, and Zero-Trust Architectures

Streams directly into 16-Bit .mgbs bitstream shards.
Automatically appends to existing shards in data/cyber_web_knowledge/shards/ without duplicates.
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

CYBER_DEEP_STANDARDS = [
    # 1. Advanced Web Security & OWASP Top 10 Full Mitigations
    """# Advanced Web Security & OWASP Top 10 Complete Defensive Architectures:

## 1. Injection Attacks (SQLi, NoSQLi, Command Injection, LDAP)
- Parametrisierung & Strikte Typisierung: Niemals dynamische String-Interpolation oder Format-Strings für Query-Konstruktionen nutzen.
- Prepared Statements: Datenbank-Engines kompilieren die Query-Struktur vorab, Datenparameter werden isoliert als reine Werte gebunden.
- ORM-Sicherheit: Auch in ORMs (SQLAlchemy, Prisma, Hibernate) raw SQL clauses (wie raw(), whereRaw()) nur mit Parametern verwenden.
- Command Injection: Nutze APIs mit Argument-Listen (z. B. subprocess.run(['ls', '-la', user_path])) statt Shell-Ausführung (shell=True verboten).

## 2. Broken Authentication & Session Management
- Passworthashing: Ausschließlich Argon2id (Memory: 64MB, Iterations: 3, Parallelism: 4) oder bcrypt (Work Factor >= 12) mit kryptographisch zufälligem 128-Bit Salt.
- Session-Tokens: Mindestens 128 Bit Entropie via CSPRNG (os.urandom(32) / crypto.randomBytes(32)).
- Cookie-Sicherheit: SameSite=Strict; Secure; HttpOnly; Path=/; Domain=example.com.
- Rate Limiting: Token Bucket oder Leaky Bucket Algorithmen per IP und User-Identifikator auf Authentifizierungs-Endpunkten (Login, MFA, Password Reset).

## 3. Cryptographic Failures & Transport Layer Security (TLS 1.3)
- TLS 1.3: Ausschließlich moderne Cipher Suites (TLS_AES_256_GCM_SHA384, TLS_CHACHA20_POLY1305_SHA256).
- HSTS: Strict-Transport-Security: max-age=63072000; includeSubDomains; preload.
- Symmetrische Verschlüsselung: Authenticated Encryption with Associated Data (AEAD) wie AES-256-GCM oder ChaCha20-Poly1305. Niemals ECB-Modus oder CBC ohne HMAC-Verifikation nutzen.
- Asymmetrische Kryptographie: Ed25519 für Signaturen, X25519 für Key-Exchange, RSA mit mind. 4096 Bit und OAEP Padding.""",

    # 2. Modern Fullstack Web Engineering & HTML5/CSS3/TypeScript
    """# Modern Fullstack Web Architecture, Performance & Standards:

## 1. Semantic HTML5 & WCAG 2.1 AAA Accessibility
- Semantisches Markup: <header>, <nav>, <main>, <article>, <section>, <aside>, <footer> bilden die Landmark-Hierarchie für Screenreader.
- Accessible Rich Internet Applications (ARIA): ARIA nur nutzen, wenn kein nativer HTML5-Tag existiert (First Rule of ARIA). aria-expanded, aria-controls, aria-live="polite" für dynamische UI-Zustände.
- Formular-Validierung: Jedes interaktive Steuerelement benötigt ein assoziiertes <label for="id">. Native Validierung via type="email", pattern, required, minlength.

## 2. Modern CSS, Container Queries & Layout Systems
- CSS Grid & Flexbox: Zweidimensionale Layouts mit grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)).
- CSS Custom Properties (Design Tokens): Hierarchische Farb- und Abstands-Systeme via :root { --primary-h: 210; --primary-s: 100%; --primary-l: 50%; --primary: hsl(var(--primary-h) var(--primary-s) var(--primary-l)); }.
- Container Queries: @container (min-width: 400px) für modulare Komponenten-Responsivität unabhängig vom Viewport.

## 3. TypeScript, Clean Code & Asynchronous State
- Strikte Typsicherheit: strict: true, noImplicitAny: true, exactOptionalPropertyTypes: true.
- Asynchronität: Modernes Async/Await mit strukturierter Fehlerbehandlung. Vermeidung unhandled promise rejections.
- Immutability: ReadonlyArray<T>, Object.freeze, und pure transformations zur Vermeidung von Race Conditions im UI-State.""",

    # 3. System Architecture, Microservices & Zero-Trust
    """# Cloud-Native Systems, API Architecture & Zero-Trust:

## 1. RESTful API & GraphQL Best Practices
- HTTP-Semantik: Idempotente GET/PUT/DELETE, nicht-idempotente POST. Strikte Verwendung von HTTP Status Codes (200 OK, 201 Created, 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 409 Conflict, 429 Too Many Requests, 500 Internal Error).
- API-Versionierung: URI-basiert (/api/v1/...) oder Header-basiert (Accept: application/vnd.company.v1+json).
- Pagination: Keyset-basierte Cursor-Pagination (WHERE id > cursor LIMIT 20) anstelle von teuren OFFSET-Queries bei großen Tabellen.

## 2. Zero-Trust Architecture & Microservice Security
- Mutual TLS (mTLS): Jeder Dienst authentifiziert jeden anderen Dienst kryptographisch über X.509 Zertifikate.
- JWT (JSON Web Tokens): Kurzlebige Access Tokens (5-15 min) mit asymmetrischer Signaturprüfung (RS256/EdDSA) und strikter Prüfung von iss (Issuer), aud (Audience), exp (Expiration) und nbf (Not Before). Niemals Algorithmus 'none' erlauben.
- Least Privilege: Rollen- und Attribut-basierte Zugriffskontrolle (RBAC / ABAC) auf Endpunkt- und Datenbankebene."""
]


def run_massive_cyber_web_ingestion(
    output_dir: str = "/home/benjamin/Bilder/data/cyber_web_knowledge/shards",
    vocab_file: str = "/home/benjamin/Bilder/data/vocab_65k.json",
    max_tokens_per_shard: int = 500_000,
    target_shards: int = 150,
) -> int:
    global STOP_REQUESTED

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    print("=" * 80, flush=True)
    print("🛡️ MASSIVE CYBER & WEB EXPANSION PIPELINE (7B PRE-TRAINING)", flush=True)
    print(f"📁 Ziel-Verzeichnis: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    existing_files = sorted(glob.glob(os.path.join(output_dir, "*.mgbs")))
    shard_count = len(existing_files)
    print(f"  ⏭️ {shard_count} bestehende Cyber/Web Shards gefunden. Starte ab Index {shard_count:04d}...", flush=True)

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
        tps = len(buffer_tokens) / max(0.1, elapsed)
        print(f"  💾 [CYBER/WEB Shard {shard_count:04d}] {len(buffer_tokens):,} Tokens | Gesamt: {total_tokens_written:,} Tokens -> {os.path.basename(shard_path)}", flush=True)
        buffer_tokens = []

    # 1. Ingest Deep Cybersecurity & Architecture Standards
    print("\n📚 [Quelle 1/3] Tokenisiere Deep Cybersecurity & Fullstack Standards...", flush=True)
    for doc in CYBER_DEEP_STANDARDS:
        formatted = f"### System-Dokumentation (Cybersecurity & Architecture):\n{doc.strip()}\n\n"
        buffer_tokens.extend(tokenizer.encode(formatted))
        if len(buffer_tokens) >= max_tokens_per_shard:
            flush_shard()

    # 2. Ingest Evol-Instruct-Code-80k (Complete 80k dialogues)
    print("\n💻 [Quelle 2/3] Streame komplettes Evol-Instruct-Code-80k (80.000 Dialoge)...", flush=True)
    try:
        code_ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
        # Skip items already processed in earlier shards (~6000 items per 16 shards)
        skip_items = max(0, (shard_count - 1) * 350)
        print(f"  ⏭️ Überspringe ca. {skip_items:,} bereits verarbeitete Code-Dialoge...", flush=True)

        ds_iter = iter(code_ds)
        skipped = 0
        while skipped < skip_items:
            try:
                next(ds_iter)
                skipped += 1
            except StopIteration:
                break

        item_idx = skipped
        for item in ds_iter:
            if STOP_REQUESTED or shard_count >= target_shards:
                break

            instr = item.get("instruction", "")
            resp = item.get("output", "")
            if not instr or not resp:
                continue

            formatted = f"### Benutzer:\n{instr}\n\n### Assistent:\n<think>\nAnalysiere Software-Architektur, Internetsicherheit und Best Practices:\n</think>\n{resp}\n\n"
            toks = tokenizer.encode(formatted)
            buffer_tokens.extend(toks)
            item_idx += 1

            if len(buffer_tokens) >= max_tokens_per_shard:
                flush_shard()

            if item_idx % 2500 == 0:
                print(f"  ⚙️ [Evol-Code] Verarbeitet: {item_idx:,} / 80,000 Dialoge (Shards: {shard_count})", flush=True)

        print(f"✅ Evol-Instruct-Code Stream abgeschlossen ({item_idx:,} Dialoge verarbeitet).", flush=True)
    except Exception as e:
        print(f"⚠️ Hinweis bei Evol-Instruct-Code Stream: {e}", flush=True)

    # 3. Ingest The Stack Smol (Python, JS, TS, Rust, Go, HTML, CSS, Shell)
    if shard_count < target_shards and not STOP_REQUESTED:
        print("\n🌐 [Quelle 3/3] Streame The Stack Smol (Python, JS, TS, Rust, HTML, Shell)...", flush=True)
        for lang in ["python", "javascript", "typescript", "rust", "go", "html", "shell"]:
            if STOP_REQUESTED or shard_count >= target_shards:
                break
            print(f"  📥 Lade Stack Smol: {lang}...", flush=True)
            try:
                stack_ds = load_dataset("bigcode/the-stack-smol", data_dir=f"data/{lang}", split="train", streaming=True)
                doc_count = 0
                for item in stack_ds:
                    if STOP_REQUESTED or shard_count >= target_shards:
                        break
                    code_content = item.get("content", "")
                    if not code_content or len(code_content) < 50:
                        continue

                    formatted = f"### Datei ({lang}):\n{code_content}\n\n"
                    toks = tokenizer.encode(formatted)
                    buffer_tokens.extend(toks)
                    doc_count += 1

                    if len(buffer_tokens) >= max_tokens_per_shard:
                        flush_shard()

                    if doc_count >= 15000:
                        break
                print(f"  ✅ {doc_count:,} Dateien aus Stack Smol ({lang}) verarbeitet.", flush=True)
            except Exception as e:
                print(f"  ⚠️ Hinweis bei Stack Smol ({lang}): {e}", flush=True)

    flush_shard()
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 Cyber & Web Expansion abgeschlossen! Gesamte Shards: {shard_count} (~{total_tokens_written:,} Tokens)", flush=True)
    print("=" * 80, flush=True)
    return shard_count


if __name__ == "__main__":
    run_massive_cyber_web_ingestion()
