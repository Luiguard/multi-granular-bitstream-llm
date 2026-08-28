#!/usr/bin/env python3
"""Step 2: Supervised Instruction & Reasoning Fine-Tuning (SFT) on Bitstream Shards."""

import argparse
import glob
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamDecoder
from pipeline.tokenizer import ViterbiTokenizer
from scripts.train_distributed import MultiGranularCausalTransformer, ShardedBitstreamDataset


def train_instruction_tuning():
    parser = argparse.ArgumentParser(description="Step 2: Bitstream Instruction Tuning (SFT)")
    parser.add_argument("--base_model", type=str, default="./multi_granular_model.pt", help="Pfad zum vortrainierten Basismodell")
    parser.add_argument("--vocab_file", type=str, default="./data/vocab_65k.json", help="Pfad zu vocab.json")
    parser.add_argument("--instruction_shards", type=str, default="./data/instructions/shards", help="Pfad zu den Instruction .mgbs Shards")
    parser.add_argument("--output_model", type=str, default="./multi_granular_instruct_model.pt", help="Zielpfad für das Instruct-Modell")
    parser.add_argument("--epochs", type=int, default=3, help="Anzahl Epochen für Feintuning")
    parser.add_argument("--lr", type=float, default=5e-5, help="Niedrigere Lernrate für SFT")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
    args = parser.parse_args()

    print("=" * 80)
    print("🎯 SCHRITT 2: INSTRUCTION- & REASONING-FEINTUNING (SFT) AUF BITSTREAMS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if not os.path.exists(args.vocab_file):
        args.vocab_file = "./vocab.json"

    vocab = MultiGranularVocabulary.load_json(args.vocab_file)
    shard_files = glob.glob(os.path.join(args.instruction_shards, "*.mgbs"))

    if not shard_files:
        print(f"❌ Keine Instruction-Shards in {args.instruction_shards} gefunden!")
        print("Erzeuge zuerst Shards mit: python scripts/ingest_instructions.py")
        return

    dataset = ShardedBitstreamDataset(shard_files, seq_len=128)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialisiere Modell
    model = MultiGranularCausalTransformer(
        vocab_size=vocab.size,
        rank=64,
        d_model=512,
        n_layers=6,
        n_heads=8,
    ).to(device)

    # Lade vortrainiertes Basismodell aus Schritt 1
    if os.path.exists(args.base_model):
        print(f"✅ Lade vortrainierte Basisgewichte aus Schritt 1: {args.base_model}")
        model.load_state_dict(torch.load(args.base_model, map_location=device, weights_only=True), strict=False)
    else:
        print(f"⚠️ Basismodell {args.base_model} nicht gefunden, starte frisches Training.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Byte weights
    byte_weights = torch.ones(vocab.size, dtype=torch.float32, device=device)
    for t_id in range(vocab.size):
        byte_weights[t_id] = float(max(1, vocab.get_byte_len(t_id)))

    print(f"\n🚀 Starte Instruction-Tuning über {args.epochs} Epochen ({len(dataset):,} Dialog-Samples)...")
    model.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        batches = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x_batch)
                flat_logits = logits.view(-1, vocab.size)
                flat_targets = y_batch.view(-1)

                weights = byte_weights[flat_targets]
                loss_unreduced = F.cross_entropy(flat_logits, flat_targets, reduction="none")
                loss = torch.sum(loss_unreduced * weights) / torch.sum(weights)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            batches += 1

        print(f"  [SFT Epoche {epoch}/{args.epochs}] Durchschnittlicher Loss: {total_loss / max(1, batches):.4f}")

    # Speichere finales Instruct-Modell
    torch.save(model.state_dict(), args.output_model)
    print(f"\n🎉 INSTRUCT-MODELL ERFOLGREICH GESPEICHERT: {args.output_model}")
    print("Das Modell kann jetzt direkt mit Ollama verwendet werden!")


if __name__ == "__main__":
    train_instruction_tuning()
