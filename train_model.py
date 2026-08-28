#!/usr/bin/env python3
"""Fast-Track Multi-Granular Bitstream Trainer for 15-Minute Preview."""

import glob
import json
import math
import os
import sys
import time
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.vocabulary import MultiGranularVocabulary, TokenTier
from pipeline.bitstream import BitstreamHeader, BitstreamDecoder
from pipeline.tokenizer import ViterbiTokenizer


class ShardedBitstreamDataset(torch.utils.data.Dataset):
    """Robust, ultra-fast streaming dataset that reads .mgbs shards directly."""

    def __init__(self, shard_files: List[str], seq_len: int = 64):
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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


class MultiGranularCausalTransformer(nn.Module):
    """Causal Transformer with Factorized Embeddings and Byte-Weighted Projection."""

    def __init__(
        self,
        vocab_size: int,
        rank: int = 64,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1536,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.E_vocab = nn.Embedding(vocab_size, rank)
        self.E_proj = nn.Linear(rank, d_model, bias=False)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)

        self.norm_final = nn.LayerNorm(d_model)
        self.head_proj = nn.Linear(d_model, rank, bias=False)
        self.head_out = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)

        compact_emb = self.E_vocab(x)
        h = self.E_proj(compact_emb) + self.pos_embedding(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=causal_mask, is_causal=True)
        h = self.norm_final(h)

        logits = self.head_out(self.head_proj(h))
        return logits


def update_live_status(
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
        "current_loss": round(float(current_loss), 4),
        "shards_processed": shards_count,
        "loss_history": [round(float(v), 3) for v in loss_history[-30:]],
    }

    try:
        temp_file = status_file + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(data, f)
        os.replace(temp_file, status_file)
    except Exception:
        pass


def train_fast_track_preview():
    print("=" * 80, flush=True)
    print("⚡ MULTI-GRANULARER VORGESCHMACK-TRAINER (15 MINUTEN ZIELZEIT)", flush=True)
    print("=" * 80, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Rechengerät: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    vocab_file = "/home/benjamin/Bilder/data/vocab_65k.json"
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    vocab = MultiGranularVocabulary.load_json(vocab_file)
    print(f"  - Vokabulargröße |V|: {vocab.size:,} Tokens (16 Bit)", flush=True)

    status_file = "/home/benjamin/Bilder/data/training_status.json"
    shard_files = sorted(glob.glob("/home/benjamin/Bilder/data/shards/*.mgbs"))
    if not shard_files:
        shard_files = ["/home/benjamin/Bilder/wiki_ki_article.mgbs"]

    seq_len = 64
    batch_size = 16
    rank = 64
    d_model = 512
    n_layers = 6
    n_heads = 8
    target_steps = 15000  # Exakt 15.000 Schritte für ~15 Minuten Laufzeit!

    dataset = ShardedBitstreamDataset(shard_files, seq_len=seq_len)
    print(f"  - Geladene Samples: {len(dataset):,}, Ziel-Schritte: {target_steps:,}", flush=True)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    model = MultiGranularCausalTransformer(
        vocab_size=vocab.size,
        rank=rank,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=seq_len * 2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Modell-Parameter: {total_params:,} Parameter", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    byte_weights = torch.ones(vocab.size, dtype=torch.float32, device=device)
    for t_id in range(vocab.size):
        byte_weights[t_id] = float(max(1, vocab.get_byte_len(t_id)))

    model.train()
    step = 0
    start_time = time.time()
    loss_history: List[float] = []

    print("\n🚀 Starte 15-Minuten Fast-Track Training...", flush=True)
    data_iter = iter(dataloader)

    while step < target_steps:
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

        # Update live telemetry continuously
        elapsed = time.time() - start_time
        tokens_processed = step * batch_size * seq_len
        tps = tokens_processed / max(0.1, elapsed)

        update_live_status(
            status_file=status_file,
            epoch=1,
            max_epochs=1,
            step=step,
            total_steps=target_steps,
            current_loss=loss_val,
            loss_history=loss_history,
            tokens_per_sec=tps,
            elapsed_time=elapsed,
            shards_count=len(shard_files),
        )

        if step % 20 == 0:
            print(f"  Step {step:05d}/{target_steps} | Loss: {loss_val:.4f} | TPS: {int(tps)} | VRAM: {torch.cuda.memory_allocated() / (1024*1024):.1f} MB", flush=True)

    # Modell speichern
    model_save_path = "/home/benjamin/Bilder/multi_granular_model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n💾 Modell erfolgreich gespeichert unter: {model_save_path}", flush=True)

    # Desktop Notification
    try:
        os.system("notify-send '🚀 Fast-Track LLM' '🎉 15-Minuten Vorgeschmack-Modell erfolgreich fertiggestellt!' -u critical 2>/dev/null")
    except Exception:
        pass

    final_data = {
        "epoch": 1,
        "max_epochs": 1,
        "step": target_steps,
        "total_steps": target_steps,
        "progress_percent": 100.0,
        "eta_str": "00:00 min (FERTIG)",
        "tokens_per_sec": int(tps),
        "current_loss": round(float(loss_val), 4),
        "shards_processed": len(shard_files),
        "loss_history": [round(float(v), 3) for v in loss_history[-30:]],
        "status": "COMPLETED",
    }
    try:
        with open(status_file, "w") as f:
            json.dump(final_data, f)
    except Exception:
        pass


if __name__ == "__main__":
    train_fast_track_preview()
