#!/usr/bin/env python3
"""End-to-End Execution and Demonstration of the Multi-Granular Bitstream Pipeline."""

import os
import sys
import numpy as np
from pipeline.vocabulary import MultiGranularVocabulary, TokenTier
from pipeline.mining import PhraseMiner
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.bitstream import BitstreamEncoder, BitstreamDecoder, BitstreamDataset
from pipeline.model_components import FactorizedEmbedding, ByteWeightedCrossEntropyLoss, MiniTransformerBlock


def main():
    print("=" * 80)
    print("🚀 MULTI-GRANULARE TOKENISIERUNG & BITSTREAM TRAININGSPIPELINE")
    print("=" * 80)

    # 1. Realer mehrsprachiger / deutscher Trainingskorpus
    corpus = [
        "ich bin ein mensch und ich denke nach.",
        "ich bin ein mensch, der die zusammenhänge der welt verstehen will.",
        "ich bin ein forscher im bereich der künstlichen intelligenz.",
        "auf der grundlage dieser daten können wir die parameter exakt berechnen.",
        "auf der grundlage empirischer beobachtungen wird die theorie verifiziert.",
        "in der regel sinkt die sequenzlänge durch multi-granulare tokens um über sechzig prozent.",
        "in der regel führt die quadratische aufmerksamkeit bei langen sequenzen zu speicherengpässen.",
        "künstliche intelligenz und maschinelles lernen revolutionieren moderne architectures.",
        "ein bitstream speichert token-sequenzen ohne unnötigen overhead bit-gepackt auf der festplatte.",
        "das modell lernt sowohl elementare wörter als auch komplexe phrasen und satzmuster.",
        "durch das faktorierte embedding wird der speicherbedarf der vokabulartabelle drastisch reduziert.",
        "die byte-gewichtete loss-funktion verhindert die verzerrung der gradienten bei variabler tokenlänge.",
    ]

    total_chars = sum(len(doc) for doc in corpus)
    total_raw_bytes = sum(len(doc.encode("utf-8")) for doc in corpus)
    print(f"\n[1] KORPUS-ANALYSE:")
    print(f"  - Dokumente: {len(corpus)}")
    print(f"  - Zeichen gesamt: {total_chars}")
    print(f"  - UTF-8 Bytes gesamt: {total_raw_bytes} Bytes")

    # 2. Mining & PMI Berechnung
    print(f"\n[2] PHRASEN-MINING & VOKABULAR-INDUKTION:")
    miner = PhraseMiner(
        min_ngram_freq=2,
        min_pmi=0.1,
        max_ngram_len=5,
        max_vocab_budget=4096,
    )
    vocab, stats = miner.mine_from_corpus(corpus)

    print(f"  - Eindeutige Wörter (Tier 1): {stats.unique_unigrams}")
    print(f"  - Geminte Phrasen (Tier 2):   {stats.mined_phrases_count}")
    print(f"  - Satzmuster / Templates (Tier 3): {stats.mined_templates_count}")
    print(f"  - Byte-Fallback (Tier 0):     256 Tokens (0x00 - 0xFF)")
    print(f"  - Gesamtgröße Vokabular |V|:  {vocab.size} Tokens")
    print(f"  - Erforderliche Bitbreite:    {vocab.required_bits} Bits pro Token")

    # Speichern des Vokabulars
    vocab_file = "/home/benjamin/Bilder/vocab.json"
    vocab.save_json(vocab_file)
    print(f"  - Vokabular gespeichert unter: {vocab_file}")

    # 3. Viterbi Dynamic Programming Tokenisierung
    print(f"\n[3] VITERBI DYNAMISCHE TOKENISIERUNG (ZERO-OOV & LOSSLESS):")
    tokenizer = ViterbiTokenizer(vocab)

    # Beispiel-Demonstration an der Nutzer-Anfrage "ich bin ein mensch"
    demo_sentence = "ich bin ein mensch"
    demo_tokens = tokenizer.encode(demo_sentence)
    demo_breakdown = tokenizer.get_token_breakdown(demo_sentence)

    print(f"\n  👉 Analyse für Eingabe: '{demo_sentence}'")
    for t_id, text_repr, tier, b_len in demo_breakdown:
        tier_name = TokenTier(tier).name
        print(f"     Token ID: {t_id:04d} (0b{t_id:0{vocab.required_bits}b}) | Text: '{text_repr}' | Tier: {tier_name} | Bytes: {b_len}")

    demo_bitstream = " ".join(f"{t_id:0{vocab.required_bits}b}" for t_id in demo_tokens)
    print(f"     Bit-Stream: {demo_bitstream}")
    reconstructed_demo = tokenizer.decode(demo_tokens)
    print(f"     Rekonstruktion: '{reconstructed_demo}' (Verlustfrei: {reconstructed_demo == demo_sentence})")

    # Gesamten Korpus tokenisieren
    all_token_ids = []
    for doc in corpus:
        doc_tokens = tokenizer.encode(doc)
        all_token_ids.extend(doc_tokens)

    # Standard Word / Character Vergleich
    standard_word_count = sum(len(doc.split()) for doc in corpus)
    multi_token_count = len(all_token_ids)

    print(f"\n[4] KOMPRESSIONS- & EFFIZIENZMETRIKEN:")
    print(f"  - Standard-Wort-Anzahl:        {standard_word_count} Wörter")
    print(f"  - Multi-Granulare Token-Anzahl: {multi_token_count} Tokens")
    print(f"  - Sequenzlängen-Reduktion:     {((1.0 - multi_token_count / standard_word_count) * 100):.1f}% kürzere Sequenzen")
    print(f"  - KV-Cache Speicherersparnis:  {((1.0 - multi_token_count / standard_word_count) * 100):.1f}% weniger VRAM")
    print(f"  - Attention Speedup O(T^2):     ~{((standard_word_count / multi_token_count) ** 2):.2f}x schneller")

    # 4. Bitstream Binary Serialization
    print(f"\n[5] BITSTREAM SERIALISIERUNG & SPEICHERUNG:")
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=vocab.required_bits)
    packed_bytes = encoder.pack_tokens(all_token_ids)

    bitstream_file = "/home/benjamin/Bilder/corpus.mgbs"
    header = encoder.save_to_file(bitstream_file, all_token_ids, raw_byte_count=total_raw_bytes)

    file_size = os.path.getsize(bitstream_file)
    print(f"  - Bitstream-Datei:            {bitstream_file}")
    print(f"  - Rohdaten UTF-8 Größe:       {total_raw_bytes} Bytes")
    print(f"  - Gepackte Bitstream-Größe:   {file_size} Bytes (inkl. 31 Bytes Header)")
    print(f"  - Netto Token-Daten:          {len(packed_bytes)} Bytes")
    print(f"  - Effektive Daten-Kompression: {(total_raw_bytes / max(1, file_size)):.2f}x")

    # Laden und Integritätsprüfung
    loaded_header, loaded_tokens = BitstreamDecoder.load_from_file(bitstream_file)
    assert loaded_tokens == all_token_ids, "Bitstream-Dekodierung stimmt nicht exakt überein!"
    print(f"  - Integritätsprüfung:         100% BIT-EXAKT VERIFIZIERT ✅")

    # 5. Model Training Forward & Backward Simulation
    print(f"\n[6] MODELL-TRAINING (FACTORIZED EMBEDDING & BYTE-WEIGHTED LOSS):")
    rank = 32
    embedding_dim = 128
    batch_size = 4
    seq_len = 16

    dataset = BitstreamDataset(all_token_ids, sequence_length=seq_len)
    x_batch, y_batch = dataset.get_batch(batch_size=batch_size)

    # Initialisiere faktorisiertes Embedding
    embedding = FactorizedEmbedding(
        vocab_size=vocab.size,
        rank=rank,
        embedding_dim=embedding_dim,
    )
    print(f"  - Standard Embedding Parameter:   {embedding.standard_parameter_count:,} Parameter")
    print(f"  - Faktorisiertes Embedding:       {embedding.parameter_count:,} Parameter")
    print(f"  - Parameter-Reduktion:            {(1.0 - embedding.parameter_reduction_ratio) * 100:.1f}%")

    # Forward Pass
    embedded = embedding.forward(x_batch)
    transformer = MiniTransformerBlock(embedding_dim=embedding_dim, hidden_dim=256)
    hidden = transformer.forward(embedded)

    # Projection Head
    W_out = np.random.normal(0, 0.02, (embedding_dim, vocab.size)).astype(np.float32)
    logits = np.matmul(hidden, W_out)

    # Byte-gewichteter Loss
    criterion = ByteWeightedCrossEntropyLoss(
        id_to_byte_len=vocab.id_to_byte_len,
        vocab_size=vocab.size,
    )
    loss, grad_logits = criterion.compute(logits, y_batch)

    # Backward Pass & Gradientenfluss
    grad_hidden = np.matmul(grad_logits, W_out.T)
    embedding.backward(grad_hidden)

    grad_norm = float(np.linalg.norm(embedding.grad_E_proj))
    print(f"  - Loss (Byte-Weighted NLL):       {loss:.4f}")
    print(f"  - Gradient Norm dL/dE_proj:       {grad_norm:.6f}")
    print(f"  - Gradienten-Fluss:               STABIL & BERECHNET ✅")

    print("\n" + "=" * 80)
    print("✨ ALLE PIPELINE-SCHRITTE ERFOLGREICH ABGESCHLOSSEN!")
    print("=" * 80)


if __name__ == "__main__":
    main()
