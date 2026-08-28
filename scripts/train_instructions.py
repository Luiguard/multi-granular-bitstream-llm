#!/usr/bin/env python3
"""Step 2: Supervised Instruction & Reasoning Fine-Tuning (SFT) with Live Telemetry."""

import argparse
import glob
import json
import os
import sys
import time
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamDecoder
from train_model import MultiGranularCausalTransformer, ShardedBitstreamDataset, update_live_status


def train_instruction_tuning():
    parser = argparse.ArgumentParser(description="Step 2: Bitstream Instruction Tuning (SFT)")
    parser.add_argument("--base_model", type=str, default="/home/benjamin/Bilder/multi_granular_model.pt", help="Pfad zum Basismodell")
    parser.add_argument("--vocab_file", type=str, default="/home/benjamin/Bilder/data/vocab_65k.json", help="Pfad zu vocab.json")
    parser.add_argument("--instruction_shards", type=str, default="/home/benjamin/Bilder/data/instructions/shards", help="Pfad zu den Instruction Shards")
    parser.add_argument("--output_model", type=str, default="/home/benjamin/Bilder/multi_granular_instruct_model.pt", help="Zielpfad für das Instruct-Modell")
    parser.add_argument("--target_steps", type=int, default=5000, help="Ziel-Schritte für SFT")
    parser.add_argument("--lr", type=float, default=2e-4, help="Lernrate für SFT")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
    args = parser.parse_args()

    print("=" * 80, flush=True)
    print("🎯 SCHRITT 2: INSTRUCTION- & REASONING-FEINTUNING (SFT) MIT LIVE STATUS", flush=True)
    print("=" * 80, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    if not os.path.exists(args.vocab_file):
        args.vocab_file = "/home/benjamin/Bilder/vocab.json"

    vocab = MultiGranularVocabulary.load_json(args.vocab_file)
    shard_files = sorted(glob.glob(os.path.join(args.instruction_shards, "*.mgbs")) + glob.glob("/home/benjamin/Bilder/data/biology_math/shards/*.mgbs"))

    if not shard_files:
        print(f"❌ Keine Instruction-Shards in {args.instruction_shards} gefunden!", flush=True)
        return

    status_file = "/home/benjamin/Bilder/data/training_status.json"
    dataset = ShardedBitstreamDataset(shard_files, seq_len=64)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MultiGranularCausalTransformer(
        vocab_size=vocab.size,
        rank=64,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=1536,
        max_seq_len=128,
    ).to(device)

    if os.path.exists(args.base_model):
        print(f"✅ Lade vortrainierte Basisgewichte aus Schritt 1: {args.base_model}", flush=True)
        model.load_state_dict(torch.load(args.base_model, map_location=device, weights_only=True))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    byte_weights = torch.ones(vocab.size, dtype=torch.float32, device=device)
    for t_id in range(vocab.size):
        byte_weights[t_id] = float(max(1, vocab.get_byte_len(t_id)))

    model.train()
    step = 0
    start_time = time.time()
    loss_history: List[float] = []

    print(f"\n🚀 Starte SFT Instruction Tuning ({args.target_steps:,} Schritte auf 2.82M Dialog-Tokens)...", flush=True)
    data_iter = iter(dataloader)

    while step < args.target_steps:
        try:
            x_batch, y_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x_batch, y_batch = next(data_iter)

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

        step += 1
        loss_val = float(loss.item())
        loss_history.append(loss_val)

        elapsed = time.time() - start_time
        tokens_processed = step * args.batch_size * 64
        tps = tokens_processed / max(0.1, elapsed)

        update_live_status(
            status_file=status_file,
            epoch=2,
            max_epochs=2,
            step=step,
            total_steps=args.target_steps,
            current_loss=loss_val,
            loss_history=loss_history,
            tokens_per_sec=tps,
            elapsed_time=elapsed,
            shards_count=len(shard_files),
        )

        if step % 20 == 0:
            print(f"  [SFT Step {step:04d}/{args.target_steps}] Loss: {loss_val:.4f} | TPS: {int(tps)} | VRAM: {torch.cuda.memory_allocated() / (1024*1024):.1f} MB", flush=True)

    # Speichere finales Instruct-Modell
    torch.save(model.state_dict(), args.output_model)
    print(f"\n🎉 INSTRUCT-MODELL ERFOLGREICH GESPEICHERT: {args.output_model}", flush=True)

    try:
        os.system("notify-send '🚀 SFT Instruct-Modell' '🎉 Schritt 2 (Dialog- & Reasoning-Tuning) erfolgreich abgeschlossen!' -u critical 2>/dev/null")
    except Exception:
        pass


if __name__ == "__main__":
    train_instruction_tuning()
