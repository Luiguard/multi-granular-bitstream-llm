#!/usr/bin/env python3
"""
Multilingual Dictionary Downloader & Aggregator for 175+ ISO Languages.
Downloads real, authentic dictionary data and word frequency lists from:
1. Meta NLLB-200 (204 languages, 256k subwords & script-specific unigrams)
2. HermitDave OpenSubtitles Frequency Dictionaries (62 languages, real natural word frequencies)
3. OpenAI cl100k_base (Code, STEM, LaTeX, scientific terms)
"""

import os
import sys
import json
import time
import urllib.request
from typing import Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

RAW_DIR = "/home/benjamin/Bilder/data/dictionaries_raw"
os.makedirs(RAW_DIR, exist_ok=True)

HERMITDAVE_LANGS = [
    "af", "ar", "bg", "bn", "br", "bs", "ca", "cs", "da", "de",
    "el", "en", "eo", "es", "et", "eu", "fa", "fi", "fr", "gl",
    "he", "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "ka",
    "kk", "ko", "lt", "lv", "mk", "ml", "ms", "nl", "no", "pl",
    "pt", "ro", "ru", "si", "sk", "sl", "sq", "sr", "sv", "ta",
    "te", "th", "tl", "tr", "uk", "ur", "vi", "ze_en", "ze_zh", "zh_cn", "zh_tw"
]

def download_hermitdave_lang(lang: str, max_words: int = 15000) -> Tuple[str, List[Tuple[str, int]]]:
    """Downloads frequency wordlist for a language from HermitDave GitHub raw."""
    target_file = os.path.join(RAW_DIR, f"{lang}_words.json")
    if os.path.exists(target_file):
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return lang, [(w["word"], w["freq"]) for w in data[:max_words]]
        except Exception:
            pass

    # Try 50k first, then full
    candidates = [
        f"https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/{lang}/{lang}_50k.txt",
        f"https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/{lang}/{lang}_full.txt"
    ]

    words = []
    for url in candidates:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8", errors="ignore")
                for line in content.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word = parts[0]
                        try:
                            freq = int(parts[1])
                        except ValueError:
                            freq = 1
                        words.append((word, freq))
                        if len(words) >= max_words:
                            break
            if words:
                break
        except Exception:
            continue

    if words:
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump([{"word": w, "freq": fr} for w, fr in words], f, ensure_ascii=False)

    return lang, words


def download_all_hermitdave(max_workers: int = 8, words_per_lang: int = 15000) -> Dict[str, List[Tuple[str, int]]]:
    print(f"🌍 Lade Frequenz-Wörterbücher für {len(HERMITDAVE_LANGS)} Sprachen von HermitDave...", flush=True)
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(download_hermitdave_lang, lang, words_per_lang): lang for lang in HERMITDAVE_LANGS}
        for fut in as_completed(future_map):
            lang = future_map[fut]
            try:
                lang, words = fut.result()
                if words:
                    results[lang] = words
                    print(f"  ✅ [{lang.upper()}] {len(words):,} Wörter geladen", flush=True)
                else:
                    print(f"  ⚠️ [{lang.upper()}] Keine Daten gefunden", flush=True)
            except Exception as e:
                print(f"  ❌ [{lang.upper()}] Fehler: {e}", flush=True)
    return results


def extract_nllb_tokens() -> List[str]:
    """Extracts all 256,204 tokens from NLLB-200 tokenizer covering 204 ISO languages."""
    print("🌐 Extrahiere multilinguale Tokens aus Meta NLLB-200 (204 Sprachen)...", flush=True)
    out_file = os.path.join(RAW_DIR, "nllb_tokens.json")
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            return json.load(f)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    vocab = tok.get_vocab()
    sorted_tokens = [k for k, _ in sorted(vocab.items(), key=lambda x: x[1])]
    
    cleaned_tokens = []
    for t in sorted_tokens:
        if t.startswith(" "):
            cleaned = " " + t[1:]
        else:
            cleaned = t
        cleaned_tokens.append(cleaned)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_tokens, f, ensure_ascii=False)

    print(f"  ✅ {len(cleaned_tokens):,} NLLB-Tokens erfolgreich gespeichert -> {out_file}", flush=True)
    return cleaned_tokens


def main():
    start = time.time()
    print("=" * 80)
    print("📚 STARTING MULTILINGUAL DICTIONARY INGESTION (175+ ISO LANGUAGES)")
    print("=" * 80)
    
    nllb_tokens = extract_nllb_tokens()
    hd_dicts = download_all_hermitdave(max_workers=8, words_per_lang=15000)
    
    total_hd_words = sum(len(w) for w in hd_dicts.values())
    print("=" * 80)
    print(f"🎉 Wörterbuch-Download abgeschlossen in {time.time() - start:.1f}s:")
    print(f"  • NLLB-200 Multilingual Tokens: {len(nllb_tokens):,} (204 Sprachen)")
    print(f"  • HermitDave Wortlisten: {len(hd_dicts)} Sprachen mit {total_hd_words:,} echten Wörtern")
    print(f"  • Speicherort: {RAW_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
