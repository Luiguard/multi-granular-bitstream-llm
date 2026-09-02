#!/usr/bin/env python3
"""Cluster-Scale & Distributed Training Script for Multi-Granular Bitstream Transformer.

Supports Single-GPU, Multi-GPU (DDP), and FSDP on arbitrary cluster hardware.
"""

import argparse
import glob
import math
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamDecoder
from pipeline.tokenizer import ViterbiTokenizer


class ShardedBitstreamDataset(Dataset):
    """Memory-mapped dataset that streams .mgbs shards directly from NVMe."""

    def __init__(self, shard_files: List[str], seq_len: int = 512):
        self.shard_files = sorted(shard_files)
        self.seq_len = seq_len
        self.token_arrays: List[np.ndarray] = []
        self.cumulative_lengths: List[int] = [0]

        total_tokens = 0
        for s_file in self.shard_files:
            _, tokens = BitstreamDecoder.load_from_file(s_file)
            arr = np.array(tokens, dtype=np.int64)
            self.token_arrays.append(arr)
            valid_samples = max(0, len(arr) - seq_len)
            total_tokens += valid_samples
            self.cumulative_lengths.append(total_tokens)

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

        return torch.from_numpy(x_np).long(), torch.from_numpy(y_np).long()


class MultiGranularCausalTransformer(nn.Module):
    """Scalable Causal Transformer with Factorized Embeddings for Multi-Granular Bitstreams."""

    def __init__(
        self,
        vocab_size: int,
        rank: int = 64,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Factorized Input Embedding
        self.E_vocab = nn.Embedding(vocab_size, rank)
        self.E_proj = nn.Linear(rank, d_model, bias=False)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer Layers
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Factorized Output Head
        self.norm_final = nn.LayerNorm(d_model)
        self.head_proj = nn.Linear(d_model, rank, bias=False)
        self.head_out = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)

        compact = self.E_vocab(x)
        h = self.drop(self.E_proj(compact) + self.pos_embedding(positions))

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=causal_mask, is_causal=True)
        h = self.norm_final(h)

        logits = self.head_out(self.head_proj(h))
        return logits


def main():
    parser = argparse.ArgumentParser(description="Multi-Granular Distributed Trainer")
    parser.add_argument("--vocab_file", type=str, required=True, help="Pfad zu vocab.json")
    parser.add_argument("--shards_dir", type=str, required=True, help="Verzeichnis mit .mgbs Shard-Dateien")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Speicherort für Checkpoints")
    parser.add_argument("--rank_dim", type=int, default=64, help="Embedding Factorization Rank")
    parser.add_argument("--d_model", type=int, default=768, help="Modell Dimension")
    parser.add_argument("--n_layers", type=int, default=12, help="Anzahl Transformer Schichten")
    parser.add_argument("--n_heads", type=int, default=12, help="Anzahl Attention Heads")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequenzlänge")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch-Größe pro GPU")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient Accumulation Steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Lernrate")
    parser.add_argument("--epochs", type=int, default=5, help="Anzahl Epochen")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Distributed setup
    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if is_distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        global_rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if global_rank == 0:
        print("=" * 80)
        print("🚀 MULTI-GRANULAR BITSTREAM CLUSTER TRAINER")
        print("=" * 80)
        print(f"  - Device: {device}")
        print(f"  - Distributed: {is_distributed}")

    vocab = MultiGranularVocabulary.load_json(args.vocab_file)
    shard_files = glob.glob(os.path.join(args.shards_dir, "*.mgbs"))

    if not shard_files:
        raise ValueError(f"Keine .mgbs Dateien in {args.shards_dir} gefunden!")

    dataset = ShardedBitstreamDataset(shard_files, seq_len=args.seq_len)
    sampler = DistributedSampler(dataset) if is_distributed else None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    raw_model = MultiGranularCausalTransformer(
        vocab_size=vocab.size,
        rank=args.rank_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_model * 4,
        max_seq_len=args.seq_len * 2,
    ).to(device)

    if is_distributed:
        model: nn.Module = DDP(raw_model, device_ids=[local_rank])
    else:
        model = raw_model

    total_params = sum(p.numel() for p in raw_model.parameters())

    if global_rank == 0:
        print(f"  - Modell-Parameter: {total_params:,} Parameter")
        print(f"  - Vokabulargröße |V|: {vocab.size:,} Tokens (16 Bit)")
        print(f"  - Trainings-Samples: {len(dataset):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Byte weights
    byte_weights = torch.ones(vocab.size, dtype=torch.float32, device=device)
    for t_id in range(vocab.size):
        byte_weights[t_id] = float(max(1, vocab.get_byte_len(t_id)))

    # Training Loop
    model.train()
    step = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        batches = 0
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                flat_logits = logits.view(-1, vocab.size)
                flat_targets = y.view(-1)

                weights = byte_weights[flat_targets]
                loss_unreduced = F.cross_entropy(flat_logits, flat_targets, reduction="none")
                loss = torch.sum(loss_unreduced * weights) / torch.sum(weights)
                loss = loss / args.grad_accum

            scaled_loss = scaler.scale(loss)
            assert isinstance(scaled_loss, torch.Tensor)
            scaled_loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1

            epoch_loss += loss.item() * args.grad_accum
            batches += 1

            if global_rank == 0 and step % 25 == 0 and step > 0:
                vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
                print(f"  [Epoche {epoch}/{args.epochs}] Step {step:04d} | Loss: {loss.item() * args.grad_accum:.4f} | VRAM: {vram_mb:.1f} MB")

        avg_loss = epoch_loss / max(1, batches)
        if global_rank == 0:
            print(f"✅ Epoche {epoch} abgeschlossen | Durchschnittlicher Loss: {avg_loss:.4f}")
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch}.pt")
            torch.save(raw_model.state_dict(), checkpoint_path)
            print(f"  💾 Checkpoint gespeichert: {checkpoint_path}")

    if global_rank == 0:
        final_path = os.path.join(args.output_dir, "best_model.pt")
        torch.save(raw_model.state_dict(), final_path)
        print(f"\n🎉 Training erfolgreich abgeschlossen! Finales Modell: {final_path}")


if __name__ == "__main__":
    main()
