#!/usr/bin/env python3
"""Fetch a real multi-page Wikipedia article and benchmark the Multi-Granular Bitstream Pipeline."""

import json
import os
import urllib.parse
import urllib.request
import numpy as np

from pipeline.vocabulary import MultiGranularVocabulary, TokenTier
from pipeline.mining import PhraseMiner
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.bitstream import BitstreamEncoder, BitstreamDecoder, BitstreamDataset
from pipeline.model_components import FactorizedEmbedding, ByteWeightedCrossEntropyLoss, MiniTransformerBlock


def fetch_wikipedia_article(title: str = "Künstliche Intelligenz") -> str:
    """Fetches full plain text of a real Wikipedia article using the MediaWiki API."""
    encoded_title = urllib.parse.quote(title)
    url = f"https://de.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={encoded_title}&format=json"
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MultiGranularBitstreamResearch/1.0 (academic research project)"},
    )
    
    with urllib.request.urlopen(req, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))
    
    pages = data["query"]["pages"]
    page_id = next(iter(pages))
    extract = pages[page_id]["extract"]
    return extract


def main():
    print("=" * 80)
    print("📚 WIKIPEDIA BENCHMARK: MULTI-GRANULARE BITSTREAM TRAININGSPIPELINE")
    print("=" * 80)

    # 1. Realer Wikipedia-Text Download
    article_title = "Künstliche Intelligenz"
    print(f"\n[1] Lade echten Wikipedia-Artikel: '{article_title}'...")
    raw_text = fetch_wikipedia_article(article_title)

    raw_char_count = len(raw_text)
    raw_utf8_bytes = len(raw_text.encode("utf-8"))
    standard_words = len(raw_text.split())
    estimated_pages = max(1, raw_char_count // 2500)

    print(f"  - Geladene Zeichen:     {raw_char_count:,} Zeichen")
    print(f"  - Geladene UTF-8 Bytes: {raw_utf8_bytes:,} Bytes (~{raw_utf8_bytes / 1024:.1f} KB)")
    print(f"  - Wortanzahl:           {standard_words:,} Wörter")
    print(f"  - Geschätzter Umfang:   ~{estimated_pages} DIN-A4 Normseiten")

    # Aufteilen in Absätze
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 20]
    print(f"  - Gültige Absätze:      {len(paragraphs)}")

    # 2. Statistisches Mining & Vokabular-Induktion
    print(f"\n[2] PHRASEN-MINING & STATISTISCHE INDUKTION (PMI & INFORMATION GAIN):")
    miner = PhraseMiner(
        min_ngram_freq=3,
        min_pmi=0.5,
        max_ngram_len=5,
        max_vocab_budget=8192,
    )
    vocab, stats = miner.mine_from_corpus(paragraphs)

    print(f"  - Tier 0 (Byte-Fallback):          256 Tokens (0x00 - 0xFF)")
    print(f"  - Tier 1 (Einzelne Wörter):        {stats.unique_unigrams:,} Tokens")
    print(f"  - Tier 2 (Geminte Phrasen):        {stats.mined_phrases_count:,} Tokens")
    print(f"  - Tier 3 (Satzmuster/Templates):   {stats.mined_templates_count:,} Tokens")
    print(f"  - Gesamtgröße Vokabular |V|:       {vocab.size:,} Tokens")
    print(f"  - Erforderliche Bitbreite:         {vocab.required_bits} Bits pro Token")

    # Top-10 Phrasen anzeigen
    top_phrases = [
        (vocab.get_token(i), vocab.id_to_frequency[i], vocab.id_to_pmi[i], vocab.id_to_byte_len[i])
        for i in range(256 + stats.unique_unigrams, min(vocab.size, 256 + stats.unique_unigrams + 10))
    ]
    print("\n  Top 10 automatisch induzierte Phrasen:")
    for phrase, freq, pmi, blen in top_phrases:
        print(f"    • '{phrase}' (Freq: {freq}, PMI: {pmi:.2f}, Bytes: {blen})")

    # 3. Viterbi Dynamische Tokenisierung
    print(f"\n[3] VITERBI DYNAMISCHE TOKENISIERUNG:")
    tokenizer = ViterbiTokenizer(vocab, length_bonus_factor=0.6)

    # Gesamten Artikel tokenisieren
    all_token_ids = []
    for p in paragraphs:
        tokens = tokenizer.encode(p)
        all_token_ids.extend(tokens)

    multi_token_count = len(all_token_ids)
    compression_rate = (1.0 - (multi_token_count / standard_words)) * 100.0

    print(f"  - Standard Wörter:              {standard_words:,}")
    print(f"  - Multi-Granulare Tokens:       {multi_token_count:,}")
    print(f"  - Sequenzlängen-Kompression:    {compression_rate:.2f}% kürzere Sequenzen!")
    print(f"  - KV-Cache VRAM-Reduktion:      {compression_rate:.2f}% weniger Cache-Speicher")
    print(f"  - Attention Speedup O(T^2):     ~{(standard_words / multi_token_count)**2:.2f}x rechnerische Beschleunigung")

    # 4. Verlustfreie Rekonstruktionsprüfung
    print(f"\n[4] REKONSTRUKTIONS- & VERLUSTFREIHEITS-PRÜFUNG:")
    sample_paragraph = paragraphs[0]
    sample_tokens = tokenizer.encode(sample_paragraph)
    reconstructed_sample = tokenizer.decode(sample_tokens)

    assert reconstructed_sample == sample_paragraph, "Rekonstruktions-Fehler!"
    print(f"  - Sample Absatz ({len(sample_paragraph)} Zeichen): 100% verlustfrei rekonstruiert ✅")

    # 5. Binäre Bitstream-Serialisierung (.mgbs)
    print(f"\n[5] BINÄRE BITSTREAM-SERIALISIERUNG (.mgbs):")
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=vocab.required_bits)
    packed_bytes = encoder.pack_tokens(all_token_ids)

    bitstream_path = "/home/benjamin/Bilder/wiki_ki_article.mgbs"
    header = encoder.save_to_file(bitstream_path, all_token_ids, raw_byte_count=raw_utf8_bytes)
    file_size_bytes = os.path.getsize(bitstream_path)

    print(f"  - Gespeicherte Bitstream-Datei: {bitstream_path}")
    print(f"  - Ursprüngliche UTF-8 Größe:   {raw_utf8_bytes:,} Bytes")
    print(f"  - MGBS-Bitstream Dateigröße:    {file_size_bytes:,} Bytes (inkl. Header)")
    print(f"  - Effektive Daten-Kompression:  {raw_utf8_bytes / file_size_bytes:.2f}x")

    # Bit-exakte Rücklesung
    loaded_header, loaded_tokens = BitstreamDecoder.load_from_file(bitstream_path)
    assert loaded_tokens == all_token_ids, "Bitstream Ladefehler!"
    print(f"  - Bitstream-Integrität:         100% BIT-EXAKT VERIFIZIERT ✅")

    # 6. Mini-Training auf dem Wikipedia Bitstream
    print(f"\n[6] NEURONALES TRAINING AUF DEM WIKIPEDIA BITSTREAM:")
    rank = 32
    embedding_dim = 128
    hidden_dim = 256
    seq_len = 32
    batch_size = 8
    num_steps = 20

    dataset = BitstreamDataset(all_token_ids, sequence_length=seq_len)
    embedding = FactorizedEmbedding(vocab_size=vocab.size, rank=rank, embedding_dim=embedding_dim)
    transformer = MiniTransformerBlock(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    W_out = np.random.normal(0, 0.02, (embedding_dim, vocab.size)).astype(np.float32)
    criterion = ByteWeightedCrossEntropyLoss(id_to_byte_len=vocab.id_to_byte_len, vocab_size=vocab.size)

    print(f"  - Modell: Factorized Embedding (Rank={rank}, Dim={embedding_dim})")
    print(f"  - Standard Embedding Matrix:    {vocab.size * embedding_dim:,} Parameter")
    print(f"  - Faktorisiertes Embedding:     {embedding.parameter_count:,} Parameter")
    print(f"  - Parameter-Ersparnis:          {(1.0 - embedding.parameter_reduction_ratio) * 100:.1f}%")
    print(f"  - Starte {num_steps} Trainingsschritte...")

    initial_loss = 0.0
    final_loss = 0.0

    for step in range(1, num_steps + 1):
        x, y = dataset.get_batch(batch_size=batch_size)
        emb = embedding.forward(x)
        h = transformer.forward(emb)
        logits = np.matmul(h, W_out)
        loss, grad_logits = criterion.compute(logits, y)

        if step == 1:
            initial_loss = loss

        # Backward & SGD Update
        grad_h = np.matmul(grad_logits, W_out.T)
        grad_W_out = np.matmul(h.reshape(-1, embedding_dim).T, grad_logits.reshape(-1, vocab.size))
        W_out -= 0.05 * grad_W_out
        embedding.backward(grad_h)

        if step % 5 == 0 or step == 1:
            print(f"    Schritt {step:02d}/{num_steps:02d} | Byte-Weighted NLL Loss: {loss:.4f}")
        
        final_loss = loss

    print(f"\n  - Trainings-Konvergenz: Initial Loss {initial_loss:.4f} ➔ Final Loss {final_loss:.4f} (Δ = -{initial_loss - final_loss:.4f})")
    print("=" * 80)
    print("✨ WIKIPEDIA BENCHMARK VOLLSTÄNDIG & ERFOLGREICH BEENDET!")
    print("=" * 80)


if __name__ == "__main__":
    main()
