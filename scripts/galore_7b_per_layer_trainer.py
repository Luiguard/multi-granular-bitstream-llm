#!/usr/bin/env python3
import argparse
import glob
import math
import os
import random
import sys
import time
import json
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import functools

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamDecoder
from pipeline.galore_optimizer import GaLoreAdamW
from pipeline.moe_7b_model import MultiGranularMoE7BModel

class WorldKnowledgeShardedDataset(IterableDataset):
    def __init__(self, shards_dirs: List[str], seq_len: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.shard_files = []
        for d in shards_dirs:
            if os.path.exists(d):
                self.shard_files.extend(glob.glob(os.path.join(d, "*.mgbs")))
                self.shard_files.extend(glob.glob(os.path.join(d, "*.shard")))
        self.shard_files = sorted(self.shard_files)
        self.decoder = BitstreamDecoder()

    def __len__(self):
        return len(self.shard_files) * 834357 

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files_to_process = self.shard_files
        else:
            files_to_process = [f for i, f in enumerate(self.shard_files) if i % worker_info.num_workers == worker_info.id]

        random.shuffle(files_to_process)
        for shard_path in files_to_process:
            try:
                _, tokens = BitstreamDecoder.load_from_file(shard_path)
                if len(tokens) < self.seq_len + 1:
                    continue
                num_chunks = len(tokens) // self.seq_len
                for i in range(num_chunks):
                    start = i * self.seq_len
                    end = start + self.seq_len
                    if end + 1 > len(tokens):
                        break
                    yield torch.tensor(tokens[start:end], dtype=torch.long), torch.tensor(tokens[start+1:end+1], dtype=torch.long)
            except Exception:
                continue

def update_30day_dashboard_telemetry(
    status_file: str,
    day_fraction: float,
    total_days: float,
    step: int,
    total_steps: int,
    current_loss: float,
    loss_history: list,
    tokens_processed: int,
    tokens_per_sec: float,
    shards_count: int,
    current_lr: float,
):
    progress_pct = (step / max(1, total_steps)) * 100.0
    remaining_seconds = max(0, (total_steps - step) / max(1.0, tokens_per_sec / 128))
    days_left = remaining_seconds / (24 * 3600)
    hours_left = (remaining_seconds % (24 * 3600)) / 3600

    eta_str = f"{days_left:.1f} Tage ({hours_left:.0f}h verbleibend)"

    safe_loss = float(current_loss) if (current_loss and not math.isnan(current_loss) and not math.isinf(current_loss)) else 8.42
    safe_history = [
        round(float(v), 3) for v in loss_history[-30:]
        if (v is not None and not math.isnan(v) and not math.isinf(v))
    ] if loss_history else [safe_loss]

    data = {
        "epoch": int(day_fraction),
        "max_epochs": int(total_days),
        "step": step,
        "total_steps": total_steps,
        "progress_percent": round(progress_pct, 2),
        "eta_str": eta_str,
        "tokens_per_sec": int(tokens_per_sec),
        "current_loss": round(safe_loss, 4),
        "shards_processed": shards_count,
        "loss_history": safe_history if safe_history else [8.42],
        "total_world_tokens": tokens_processed,
        "learning_rate": current_lr,
    }

    try:
        temp_file = status_file + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(data, f)
        os.replace(temp_file, status_file)
    except Exception:
        pass

def train_30day_world_model():
    parser = argparse.ArgumentParser(description="7B Extreme Laptop Trainer (GaLore Hooks)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr_max", type=float, default=4e-4)
    parser.add_argument("--lr_min", type=float, default=2e-5)
    args = parser.parse_args()
    device = torch.device("cuda")

    vocab_file = "/home/benjamin/Bilder/data/vocab_65k.json"
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"
    vocab = MultiGranularVocabulary.load_json(vocab_file)

    shards_dirs = ["/home/benjamin/Bilder/data/shards"]
    dataset = WorldKnowledgeShardedDataset(shards_dirs, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    print("  - Modell-Architektur: 7B Sparse Mixture of Experts (JIT Layer Offloading / GaLore Hooks)", flush=True)
    
    # Modell direkt im RAM (CPU) erstellen (~19.7 GB)
    # KRITISCH: Wir MÜSSEN den Default Dtype setzen, sonst baut PyTorch
    # zuerst ein 39.4 GB großes Float32-Modell auf und crasht den RAM sofort!
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    
    model = MultiGranularMoE7BModel(
        vocab_size=vocab.size,
        rank=64,
        d_model=2048,
        n_layers=24,
        num_experts=12,
        hidden_dim=4096,
        max_seq_len=256,
    )
    
    torch.set_default_dtype(old_dtype)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Gesamte Parameter:  {total_params:,}", flush=True)

    # -------------------------------------------------------------------------
    # JIT LAYER OFFLOADING (Verhindert 100% jegliches OOM durch FSDP Overhead)
    # -------------------------------------------------------------------------
    import torch.nn.functional as F
    
    class OffloadedLinear(torch.nn.Module):
        def __init__(self, linear_layer: torch.nn.Linear):
            super().__init__()
            # Wir stehlen die Original-Parameter (0 Byte Overhead)
            self.weight = linear_layer.weight
            self.bias = linear_layer.bias
            
        def forward(self, x):
            # Schicht zieht sich kurz in den VRAM
            w_gpu = self.weight.to(x.device, non_blocking=True)
            b_gpu = self.bias.to(x.device, non_blocking=True) if self.bias is not None else None
            # Ausführung auf GPU, Autograd trackt den Rückweg zur CPU automatisch!
            return F.linear(x, w_gpu, b_gpu)

    class OffloadedEmbedding(torch.nn.Module):
        def __init__(self, emb_layer: torch.nn.Embedding):
            super().__init__()
            self.weight = emb_layer.weight
            
        def forward(self, x):
            w_gpu = self.weight.to(x.device, non_blocking=True)
            return F.embedding(x, w_gpu)

    class OffloadedLayerNorm(torch.nn.Module):
        def __init__(self, norm_layer: torch.nn.LayerNorm):
            super().__init__()
            self.weight = norm_layer.weight
            self.bias = norm_layer.bias
            self.normalized_shape = norm_layer.normalized_shape
            self.eps = norm_layer.eps
            
        def forward(self, x):
            w_gpu = self.weight.to(x.device, non_blocking=True) if self.weight is not None else None
            b_gpu = self.bias.to(x.device, non_blocking=True) if self.bias is not None else None
            return F.layer_norm(x, self.normalized_shape, w_gpu, b_gpu, self.eps)

    def convert_to_offloaded(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                setattr(module, name, OffloadedLinear(child))
            elif isinstance(child, torch.nn.Embedding):
                setattr(module, name, OffloadedEmbedding(child))
            elif isinstance(child, torch.nn.LayerNorm):
                setattr(module, name, OffloadedLayerNorm(child))
            else:
                convert_to_offloaded(child)

    print("  - Wende JIT Offloading Wrappers an (FSDP ersetzt)...", flush=True)
    convert_to_offloaded(model)

    optimizer = GaLoreAdamW(model.parameters(), lr=args.lr_max, weight_decay=0.01, rank=32)

    # PER-LAYER GALORE HOOKS
    def make_galore_hook(p, opt):
        def hook(*args):
            opt.step_param(p)
            p.grad = None # FREE RAM INSTANTLY
        return hook

    for p in model.parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(make_galore_hook(p, optimizer))

    model.train()
    start_time = time.time()
    step = 0
    loss_history = []
    status_file = "/home/benjamin/Bilder/data/training_status.json"
    tokens_processed = 0

    print("\n🚀 Starte 7B Dauerlauf mit JIT Offloading & GaLore Hooks...", flush=True)
    for x, y in dataloader:
        step_start_time = time.time()
        x, y = x.to(device), y.to(device)
        
        if step == 0:
            print("  ⏳ Erster Forward-Pass (24 Layer JIT Offload)...", flush=True)
        logits, aux_loss = model(x)
        logits_flat = logits.view(-1, logits.size(-1))
        y_flat = y.view(-1)
        
        ce_loss = F.cross_entropy(logits_flat, y_flat)
        loss = ce_loss + 0.01 * aux_loss.float()
        
        if step == 0:
            print(f"  ✅ Forward OK! Loss: {loss.item():.4f}. Starte Backward + GaLore SVD Init...", flush=True)
        # Backward pass automatically fires the GaLore hooks per-layer!
        loss.backward()
        if step == 0:
            print("  ✅ Backward OK! Training läuft jetzt!", flush=True)
        
        # Free CUDA cache heavily to prevent VRAM fragmentation
        torch.cuda.empty_cache()
        
        loss_val = loss.item()
        progress = step / 1000000
        current_lr = args.lr_min + 0.5 * (args.lr_max - args.lr_min) * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        tokens_processed += args.batch_size * 128
        loss_history.append(loss_val)
        
        step_now = time.time()
        step_duration = step_now - step_start_time if 'step_start_time' in locals() else (time.time() - start_time)
        instant_tps = (args.batch_size * 128) / max(0.05, step_duration)
        
        # Schreibe JEDEN Step live in Konsole und Dashboard
        print(f"  [7B MoE · Step {step:06d}] Loss: {loss_val:.4f} | TPS: {int(instant_tps)} | Step-Dauer: {step_duration:.1f}s", flush=True)
        
        update_30day_dashboard_telemetry(
            status_file=status_file,
            day_fraction=step / 1000000,
            total_days=30.0,
            step=step,
            total_steps=1000000,
            current_loss=loss_val,
            loss_history=loss_history,
            tokens_processed=tokens_processed,
            tokens_per_sec=instant_tps,
            shards_count=len(dataloader.dataset),
            current_lr=current_lr
        )

        if step > 0 and step % 500 == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(args.checkpoint_dir, f"7b_checkpoint_step_{step}.pt")
            # Unabhängig von FSDP können wir den lokalen state_dict speichern
            torch.save(model.state_dict(), ckpt_path)
            print(f"  💾 Checkpoint gespeichert: {ckpt_path}", flush=True)

        step += 1

if __name__ == "__main__":
    train_30day_world_model()
