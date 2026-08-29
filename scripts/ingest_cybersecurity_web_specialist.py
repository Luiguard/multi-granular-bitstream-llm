#!/usr/bin/env python3
"""Specialized Ingestion Pipeline for Modern Web Engineering, Clean Code Best Practices, and Cybersecurity.

Ingests:
1. Web Development: Semantic HTML5, Modern CSS, JavaScript ES6+, TypeScript, Responsive Design, Accessibility (WCAG).
2. Cybersecurity & Defenses: OWASP Top 10 (SQLi, XSS, CSRF, SSRF, Auth bypass), Cryptography (AES, RSA, ECC, Argon2), Network Protocols (TLS 1.3, HTTPS, SSH), Zero-Trust.
3. Software Engineering Best Practices: SOLID, Clean Code, RESTful API design, Docker, Git workflows, Unit Testing.

Shards into 16-Bit .mgbs bitstream files.
"""

import os
import sys
import glob
from typing import List, Dict

import datasets
from datasets import load_dataset

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamEncoder
from pipeline.tokenizer import ViterbiTokenizer


CYBER_WEB_CURATED_KNOWLEDGE = [
    # 1. Modern Web Development & HTML5 Best Practices
    {
        "topic": "HTML5 & Modern Web Best Practices",
        "text": """HTML5 und Moderne Web-Entwicklung Best Practices:
1. Semantische Struktur: Verwende stets semantische HTML5-Elemente (<header>, <nav>, <main>, <article>, <section>, <aside>, <footer>) anstelle von generischen <div>-Containern, um Barrierefreiheit (a11y) und Suchmaschinenoptimierung (SEO) zu maximieren.
2. Formulare & Validierung: Formularfelder müssen stets ein eindeutiges <label for="id"> besitzen. Nutze native Validierungsattribute wie required, type="email", pattern und autocomplete.
3. Responsive Design & CSS: Nutze modernes CSS mit Flexbox und CSS Grid, CSS Custom Properties (Variablen) und Media Queries (@media (max-width: 768px)). Vermeide Inline-Styles und setze auf mobile-first CSS-Architekturen.
4. JavaScript & Asynchronität: Verwende strikt ES6+ Syntax (const/let, Arrow Functions, Destructuring, Template Literals) und modernes Async/Await für sauberes Fehlermanagement mit try/catch.
5. Web Performance: Optimiere Ladezeiten durch Lazy-Loading von Bildern (loading="lazy"), Minifizierung von Assets und asynchrones Laden von Skripten (async / defer)."""
    },
    # 2. Cybersecurity & OWASP Top 10 Defenses
    {
        "topic": "Cybersecurity & Secure Coding",
        "text": """Internetsicherheit, Kryptographie & OWASP Top 10 Schutzmaßnahmen:
1. SQL-Injection (SQLi) Vermeidung: Verwende niemals String-Konkatenation für SQL-Queries. Nutze ausnahmslos Prepared Statements und parametrisierte Abfragen (z.B. cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))).
2. Cross-Site Scripting (XSS) Abwehr: Alle dynamischen Benutzereingaben müssen vor dem Rendern im HTML-Kontext strikt kontextsensitiv maskiert (HTML-escaped) werden. Implementiere eine strenge Content Security Policy (CSP) per HTTP-Header (Content-Security-Policy: default-src 'self').
3. Cross-Site Request Forgery (CSRF): Sichere alle zustandsverändernden POST/PUT/DELETE Anfragen durch kryptographische CSRF-Tokens ab und setze Cookies stets mit SameSite=Strict, Secure und HttpOnly Flags.
4. Moderne Kryptographie & Passwort-Hashing: Speichere Passwörter niemals im Klartext oder mit schwachen Algorithmen wie MD5 oder SHA-1. Verwende ausschließlich speicherintensive KDFs wie Argon2id oder bcrypt mit individuellem Salt.
5. Netzwerk- & Transportsicherheit: Erzwinge verschlüsselte Verbindungen über TLS 1.3 mit HSTS (HTTP Strict Transport Security: max-age=31536000; includeSubDomains; preload).
6. Authentifizierung & Zero-Trust: Implementiere Multi-Faktor-Authentifizierung (MFA/2FA via TOTP/FIDO2) und das Least-Privilege-Prinzip bei Rollen- und Rechtevergabe."""
    },
    # 3. Clean Code & Software Engineering Architecture
    {
        "topic": "Clean Code & SOLID Principles",
        "text": """Clean Code Prinzipien und Software-Architektur:
1. Single Responsibility Principle (SRP): Eine Klasse oder Funktion darf nur genau einen Grund zur Änderung haben.
2. Open/Closed Principle (OCP): Software-Entwicklungen sollten offen für Erweiterungen, aber geschlossen für Modifikationen sein.
3. Liskov Substitution Principle (LSP): Subtypen müssen sich nahtlos anstelle ihrer Basistypen einsetzen lassen, ohne das Programmverhalten zu brechen.
4. Interface Segregation Principle (ISP): Clients dürfen nicht von Schnittstellen abhängig sein, die sie nicht verwenden.
5. Dependency Inversion Principle (DIP): Höhere Module dürfen nicht von niedrigen Modulen abhängen; beide müssen von Abstraktionen abhängen.
6. Error Handling & Testing: Schreibe robuste Unit-Tests (TDD/BDD), behandle Exceptions explizit und vermeide stille Fehler wie leere catch-Blöcke."""
    },
]


def ingest_cyber_and_web(
    output_dir: str = "/home/benjamin/Bilder/data/cyber_web_knowledge/shards",
    vocab_file: str = "/home/benjamin/Bilder/data/vocab_65k.json",
    max_tokens_per_shard: int = 500000,
):
    print("=" * 80, flush=True)
    print("🛡️ SPEZIAL-DATENSATZ: INTERNETSICHERHEIT, WEB-DEVELOPMENT & BEST PRACTICES", flush=True)
    print("=" * 80, flush=True)

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    vocab = MultiGranularVocabulary.load_json(vocab_file)
    tokenizer = ViterbiTokenizer(vocab)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16)

    shard_idx = 0
    buffer_tokens = []

    def flush_shard():
        nonlocal shard_idx, buffer_tokens
        if not buffer_tokens:
            return
        shard_path = os.path.join(output_dir, f"cyber_web_shard_{shard_idx:04d}.mgbs")
        encoder.save_to_file(shard_path, buffer_tokens, raw_byte_count=len(buffer_tokens) * 2)
        print(f"  💾 [CYBER/WEB Shard {shard_idx:04d}] {len(buffer_tokens):,} Tokens -> {shard_path}", flush=True)
        shard_idx += 1
        buffer_tokens = []

    # 1. Curated Gold Standards
    print("\n📚 [1/2] Tokenisiere kuratiertes Wissen zu HTML5, CSS3, OWASP & Clean Code...", flush=True)
    for entry in CYBER_WEB_CURATED_KNOWLEDGE:
        toks = tokenizer.encode(entry["text"])
        buffer_tokens.extend(toks)
    flush_shard()

    # 2. Ingest Code & Security Dialogue Instruction Dataset
    print("\n💻 [2/2] Lade & Tokenisiere Fullstack- & Security-Dialoge...", flush=True)
    try:
        code_ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
        count = 0
        for item in code_ds:
            instr = item.get("instruction", "")
            resp = item.get("output", "")
            formatted = f"### Benutzer:\n{instr}\n\n### Assistent:\n<think>\nAnalysiere Software-Architektur, Internetsicherheit und Best Practices:\n</think>\n{resp}\n\n"
            toks = tokenizer.encode(formatted)
            buffer_tokens.extend(toks)
            count += 1
            if len(buffer_tokens) >= max_tokens_per_shard:
                flush_shard()
            if count >= 6000:
                break
        print(f"✅ {count:,} Software Engineering & Security-Instruktionen gestreamt!", flush=True)
    except Exception as e:
        print(f"⚠️ Code Stream Hinweis: {e}", flush=True)

    flush_shard()
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 Cyber-, Web- & Best-Practice Shards erfolgreich erstellt in: {output_dir}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    ingest_cyber_and_web()
