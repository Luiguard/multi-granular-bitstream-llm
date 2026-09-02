#!/usr/bin/env python3
"""NVIDIA Nemotron-Style Resource-Efficient Training Script for Multi-Granular Bitstreams.

Integrates:
- Grouped Query Attention (GQA) & SwiGLU Feed-Forward Networks
- Rotary Position Embeddings (RoPE)
- PyTorch Native SDPA (FlashAttention-2 Kernel)
- Gradient Checkpointing (Enables training 300M+ models on a 6GB Laptop GPU!)
- Real-time Web Dashboard Synchronization
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamDecoder
from pipeline.tokenizer import ViterbiTokenizer
from pipeline.nemotron_components import NemotronBitstreamLM


class ShardedBitstreamDataset(Dataset):
    def __init__(self, shard_files: List[str], seq_len: int = 128):
        self.shard_files = sorted(shard_files)
        self.seq_len = seq_len
        self.token_arrays: List[np.ndarray] = []
        self.cumulative_lengths: List[int] = [0]

        total_tokens = 0
        for s_file in self.shard_files:
            try:
                header, tokens = BitstreamDecoder.load_from_file(s_file)
                arr = np.array(tokens, dtype=np.int64)
                if len(arr) > seq_len:
                    self.token_arrays.append(arr)
                    valid_samples = len(arr) - seq_len
                    total_tokens += valid_samples
                    self.cumulative_lengths.append(total_tokens)
            except Exception as e:
                print(f"Warnung bei Shard {s_file}: {e}", flush=True)

        self.total_samples = total_tokens

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = index
        shard_idx = 0
        for i in range(len(self.cumulative_lengths) - 1):
            if self.cumulative_lengths[i] <= idx < self.cumulative_lengths[i + 1]:
                shard_idx = i
                break

        offset = idx - self.cumulative_lengths[shard_idx]
        arr = self.token_arrays[shard_idx]

        x_np = arr[offset : offset + self.seq_len]
        y_np = arr[offset + 1 : offset + self.seq_len + 1]

        return torch.from_numpy(x_np), torch.from_numpy(y_np)


def update_dashboard_status(
    status_file: str,
    epoch: int,
    max_epochs: int,
    step: int,
    total_steps: int,
    current_loss: float,
    loss_history: List[float],
    tokens_per_sec: float,
    elapsed_time: float,
    shards_count: int,
):
    progress_pct = (step / max(1, total_steps)) * 100.0
    remaining_steps = max(0, total_steps - step)
    seconds_per_step = elapsed_time / max(1, step)
    eta_seconds = remaining_steps * seconds_per_step

    mins = int(eta_seconds // 60)
    secs = int(eta_seconds % 60)
    eta_str = f"{mins:02d}:{secs:02d} min"

    data = {
        "epoch": epoch,
        "max_epochs": max_epochs,
        "step": step,
        "total_steps": total_steps,
        "progress_percent": round(progress_pct, 1),
        "eta_str": eta_str,
        "tokens_per_sec": int(tokens_per_sec),
        "current_loss": round(current_loss, 4),
        "shards_processed": shards_count,
        "loss_history": [round(v, 3) for v in loss_history[-30:]],
    }

    try:
        temp_file = status_file + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(data, f)
        os.replace(temp_file, status_file)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="NVIDIA Nemotron-Style Bitstream Trainer")
    parser.add_argument("--vocab_file", type=str, default="./data/vocab_65k.json", help="Pfad zu vocab.json")
    parser.add_argument("--shards_dir", type=str, default="./data/shards", help="Verzeichnis mit .mgbs Shards")
    parser.add_argument("--d_model", type=int, default=768, help="Modell Dimension (768 für ~150M Modell)")
    parser.add_argument("--n_layers", type=int, default=12, help="Transformer Schichten")
    parser.add_argument("--n_heads", type=int, default=12, help="Query Heads")
    parser.add_argument("--n_kv_heads", type=int, default=4, help="KV Heads (GQA)")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequenzlänge")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
    parser.add_argument("--epochs", type=int, default=5, help="Epochen")
    parser.add_argument("--lr", type=float, default=3e-4, help="Lernrate")
    args = parser.parse_args()

    print("=" * 80, flush=True)
    print("🚀 NVIDIA NEMOTRON-STYLE ULTRA-EFFIZIENTER BITSTREAM TRAINER", flush=True)
    print("=" * 80, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    if not os.path.exists(args.vocab_file):
        args.vocab_file = "./vocab.json"

    vocab = MultiGranularVocabulary.load_json(args.vocab_file)
    print(f"  - Vokabulargröße: {vocab.size:,} Tokens (16 Bit)", flush=True)

    shard_files = glob.glob(os.path.join(args.shards_dir, "*.mgbs"))
    if not shard_files:
        shard_files = ["./wiki_ki_article.mgbs"]

    dataset = ShardedBitstreamDataset(shard_files, seq_len=args.seq_len)
    total_steps = (len(dataset) // args.batch_size) * args.epochs
    print(f"  - Geladene Samples: {len(dataset):,}, Steps: {total_steps:,}", flush=True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Initialisiere Nemotron Modell mit Gradient Checkpointing
    model = NemotronBitstreamLM(
        vocab_size=vocab.size,
        rank=64,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_model * 3,  # SwiGLU 3x Dimension
        max_seq_len=args.seq_len * 2,
        use_gradient_checkpointing=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Modell-Parameter: {total_params:,} Parameter", flush=True)
    print(f"  - Nemotron Features: GQA (3:1 Ratio), SwiGLU, RoPE, SDPA FlashAttention, Gradient Checkpointing", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    byte_weights = torch.ones(vocab.size, dtype=torch.float32, device=device)
    for t_id in range(vocab.size):
        byte_weights[t_id] = float(max(1, vocab.get_byte_len(t_id)))

    model.train()
    step = 0
    start_time = time.time()
    loss_history = []
    status_file = "./data/training_status.json"

    print("\n🚀 Starte Nemotron-Style Training auf RTX 3060 GPU...", flush=True)

    for epoch in range(1, args.epochs + 1):
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x_batch)
                flat_logits = logits.view(-1, vocab.size)
                flat_targets = y_batch.view(-1)

                token_weights = byte_weights[flat_targets]
                loss_unreduced = F.cross_entropy(flat_logits, flat_targets, reduction="none")
                loss = torch.sum(loss_unreduced * token_weights) / torch.sum(token_weights)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            loss_val = float(loss.item())
            loss_history.append(loss_val)

            elapsed = time.time() - start_time
            tps = (step * args.batch_size * args.seq_len) / max(0.1, elapsed)

            update_dashboard_status(
                status_file=status_file,
                epoch=epoch,
                max_epochs=args.epochs,
                step=step,
                total_steps=total_steps,
                current_loss=loss_val,
                loss_history=loss_history,
                tokens_per_sec=tps,
                elapsed_time=elapsed,
                shards_count=len(shard_files),
            )

            if step % 20 == 0:
                vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
                print(f"  [Nemotron Epoche {epoch}/{args.epochs}] Step {step:04d}/{total_steps} | Loss: {loss_val:.4f} | TPS: {int(tps)} | VRAM: {vram_mb:.1f} MB", flush=True)

    torch.save(model.state_dict(), "./nemotron_bitstream_model.pt")
    print(f"\n🎉 Nemotron-Style Modell erfolgreich gespeichert: ./nemotron_bitstream_model.pt", flush=True)


if __name__ == "__main__":
    main()
