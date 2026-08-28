#!/usr/bin/env python3
"""PyTorch Causal Transformer Training on Sharded Multi-Granular Bitstreams."""

import glob
import math
import os
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
    """Memory-efficient streaming dataset that reads .mgbs shards directly from NVMe."""

    def __init__(self, shard_files: List[str], seq_len: int = 256):
        self.shard_files = sorted(shard_files)
        self.seq_len = seq_len
        self.token_arrays: List[np.ndarray] = []
        self.cumulative_lengths: List[int] = [0]

        total_tokens = 0
        for s_file in self.shard_files:
            header, tokens = BitstreamDecoder.load_from_file(s_file)
            arr = np.array(tokens, dtype=np.int64)
            self.token_arrays.append(arr)
            valid_samples = max(0, len(arr) - seq_len)
            total_tokens += valid_samples
            self.cumulative_lengths.append(total_tokens)

        self.total_samples = total_tokens

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Find which shard contains sample idx
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
    """Causal Transformer with Factorized Embeddings and Byte-Weighted Projection."""

    def __init__(
        self,
        vocab_size: int,
        rank: int = 64,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 1536,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # 1. Factorized Embedding Layer (V x r) @ (r x d)
        self.E_vocab = nn.Embedding(vocab_size, rank)
        self.E_proj = nn.Linear(rank, d_model, bias=False)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # 2. Transformer Decoder Blocks
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

        # 3. Factorized Output Projection Head
        self.norm_final = nn.LayerNorm(d_model)
        self.head_proj = nn.Linear(d_model, rank, bias=False)
        # Weight tying with E_vocab for maximum parameter efficiency
        self.head_out = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)

        # Factorized embedding lookup
        compact_emb = self.E_vocab(x)  # (B, T, r)
        h = self.E_proj(compact_emb) + self.pos_embedding(positions)  # (B, T, d)

        # Causal Attention Mask (Upper triangular = -inf)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        # Forward through transformer layers
        h = self.transformer(h, mask=causal_mask, is_causal=True)
        h = self.norm_final(h)

        # Factorized Output Logits: (B, T, d) -> (B, T, r) -> (B, T, V)
        logits = self.head_out(self.head_proj(h))
        return logits


def train_model():
    print("=" * 80)
    print("🧠 MULTI-GRANULARER BITSTREAM TRAINER (GPU / CUDA)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Rechengerät: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    vocab_file = "/home/benjamin/Bilder/data/vocab_65k.json"
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/data/vocab_4096.json"

    print(f"  - Lade Vokabular aus: {vocab_file}")
    vocab = MultiGranularVocabulary.load_json(vocab_file)
    print(f"  - Vokabulargröße |V|: {vocab.size:,} Tokens (16 Bit)")

    shard_files = glob.glob("/home/benjamin/Bilder/data/shards/*.mgbs")
    if not shard_files:
        print("❌ Keine Shard-Dateien in data/shards/ gefunden!")
        return

    seq_len = 128
    batch_size = 16
    rank = 64
    d_model = 512
    n_layers = 6
    n_heads = 8
    epochs = 3

    print(f"  - Gefundene .mgbs Shards: {len(shard_files)} Dateien")
    dataset = ShardedBitstreamDataset(shard_files, seq_len=seq_len)
    print(f"  - Trainings-Samples gesamt: {len(dataset):,}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
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
    print(f"  - Modell-Parameter: {total_params:,} Parameter (~{total_params * 2 / (1024*1024):.1f} MB FP16)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Byte-Gewichte vorbereiten
    byte_weights = torch.ones(vocab.size, dtype=torch.float32, device=device)
    for t_id in range(vocab.size):
        byte_weights[t_id] = float(max(1, vocab.get_byte_len(t_id)))

    print("\n🚀 Starte Modell-Training...")
    model.train()
    step = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches_processed = 0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x_batch)  # (B, T, V)
                flat_logits = logits.view(-1, vocab.size)
                flat_targets = y_batch.view(-1)

                # Byte-Weighted Cross Entropy
                token_weights = byte_weights[flat_targets]
                loss_unreduced = F.cross_entropy(flat_logits, flat_targets, reduction="none")
                loss = torch.sum(loss_unreduced * token_weights) / torch.sum(token_weights)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            batches_processed += 1
            step += 1

            if step % 20 == 0:
                print(f"  [Epoche {epoch}/{epochs}] Step {step:04d} | Byte-Weighted Loss: {loss.item():.4f} | VRAM: {torch.cuda.memory_allocated() / (1024*1024):.1f} MB")

        avg_loss = epoch_loss / max(1, batches_processed)
        print(f"✅ Epoche {epoch} beendet | Durchschnitts-Loss: {avg_loss:.4f}")

    # Modell speichern
    model_save_path = "/home/benjamin/Bilder/multi_granular_model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"\n💾 Modell erfolgreich gespeichert unter: {model_save_path}")

    # Textgenerierungs-Test
    print("\n[TEST] Textgenerierungs-Test aus Bitstream:")
    model.eval()
    tokenizer = ViterbiTokenizer(vocab)
    prompt = "künstliche intelligenz ist"
    prompt_tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    generated_tokens = list(prompt_tokens)
    with torch.no_grad():
        for _ in range(15):
            curr_input = torch.tensor([generated_tokens[-seq_len:]], dtype=torch.long, device=device)
            logits = model(curr_input)
            next_token = int(torch.argmax(logits[0, -1, :]).item())
            generated_tokens.append(next_token)

    reconstructed_text = tokenizer.decode(generated_tokens)
    print(f"  - Prompt:        '{prompt}'")
    print(f"  - Generiert:     '{reconstructed_text}'")
    print("=" * 80)


if __name__ == "__main__":
    train_model()
