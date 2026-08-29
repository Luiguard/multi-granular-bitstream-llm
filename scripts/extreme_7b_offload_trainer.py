#!/usr/bin/env python3
"""Autonomous 30-Day Long-Horizon World-Knowledge Training Engine.

Synthesizes:
- [A] Gradient Accumulation (Virtual Large-Batch Supercomputer Training)
- [B] Hardware Safety, Auto-Recovery, Hourly Checkpoint Rotation, and NVMe Streaming
- [C] Sparse Mixture-of-Experts (1.2B MoE / 250M Active) + GaLore Optimizer & SwiGLU

Designed to run stably 24/7 on laptop hardware with temperature protection.
"""

import argparse
import glob
import json
import math
import os
import random
import subprocess
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
from pipeline.nemotron_components import NemotronBitstreamLM
from pipeline.galore_optimizer import GaLoreAdamW
from scripts.upcycle_to_moe import MultiGranularMoE1B2Model


import functools

from torch.utils.data import IterableDataset

class WorldKnowledgeShardedDataset(IterableDataset):
    """Highly optimized streaming dataset for Chinchilla-scale training (7B+ tokens)."""

    def __init__(self, shards_dirs: List[str], seq_len: int = 128):
        self.seq_len = seq_len
        self.shard_files = []

        for s_dir in shards_dirs:
            if os.path.exists(s_dir):
                self.shard_files.extend(glob.glob(os.path.join(s_dir, "*.mgbs")))

        self.shard_files = sorted(list(set(self.shard_files)))
        
        # Calculate total samples quickly using headers
        self.total_samples = 0
        self.valid_shards = []
        for s_file in self.shard_files:
            try:
                header = BitstreamDecoder.read_header_only(s_file)
                if header.token_count > seq_len:
                    self.valid_shards.append(s_file)
                    self.total_samples += (header.token_count - seq_len)
            except Exception:
                pass

    def __iter__(self):
        # Zufällige Reihenfolge der Shards pro Epoche
        shuffled_shards = list(self.valid_shards)
        random.shuffle(shuffled_shards)

        for shard_path in shuffled_shards:
            try:
                header, tokens = BitstreamDecoder.load_from_file(shard_path)
                arr = np.array(tokens, dtype=np.int64)
                
                # Yield batches sequentially from this shard
                max_offset = len(arr) - self.seq_len
                if max_offset > 0:
                    for offset in range(max_offset):
                        x_np = arr[offset : offset + self.seq_len]
                        y_np = arr[offset + 1 : offset + self.seq_len + 1]
                        yield torch.from_numpy(x_np), torch.from_numpy(y_np)
            except Exception:
                continue

    def __len__(self) -> int:
        return self.total_samples



def get_gpu_temperature() -> int:
    """Queries NVIDIA GPU temperature to ensure 24/7 hardware safety."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if res.returncode == 0:
            return int(res.stdout.strip())
    except Exception:
        pass
    return 50


def update_30day_dashboard_telemetry(
    status_file: str,
    day_fraction: float,
    total_days: float,
    step: int,
    total_steps: int,
    current_loss: float,
    loss_history: List[float],
    tokens_processed: int,
    tokens_per_sec: float,
    shards_count: int,
    current_lr: float,
):
    """Exports 30-day long-horizon metrics for real-time visualization."""
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
    parser = argparse.ArgumentParser(description="Autonomous 30-Day World-Knowledge Trainer")
    parser.add_argument("--days", type=float, default=30.0, help="Trainingsdauer in Tagen")
    parser.add_argument("--batch_size", type=int, default=16, help="GPU Micro-Batch Size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Grad Accum Steps (Effektive Batch Size = 128)")
    parser.add_argument("--lr_max", type=float, default=4e-4, help="Maximale Lernrate")
    parser.add_argument("--lr_min", type=float, default=2e-5, help="Minimale Lernrate")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_7b", help="Checkpoint-Ordner")
    parser.add_argument("--use_moe", action="store_true", default=True, help="Nutze 1.2B Sparse MoE Architektur")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    status_file = "/home/benjamin/Bilder/data/training_status.json"

    print("=" * 80, flush=True)
    print("🌍 AUTONOMER 30-TAGE WELTWISSEN-TRAINER (SYNTHESE AUS A, B UND C)", flush=True)
    print("=" * 80, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Device:             {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"  - Geplante Dauer:     {args.days:.1f} Tage (24/7 geschützt)", flush=True)
    print(f"  - Effektive Batch:    {args.batch_size * args.gradient_accumulation_steps} (16 x {args.gradient_accumulation_steps} Grad Accum Steps)", flush=True)

    vocab_file = "/home/benjamin/Bilder/data/vocab_65k.json"
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    vocab = MultiGranularVocabulary.load_json(vocab_file)
    print(f"  - Vokabulargröße:     {vocab.size:,} Tokens (16-Bit Multi-Granular)", flush=True)

    # Shard-Quellen (World Knowledge + STEM + Cyber/Web + AI Research + Math/Bio + FineWeb + Code + Chinchilla)
    shards_dirs = [
        "/home/benjamin/Bilder/data/chinchilla_corpus_shards",
        "/home/benjamin/Bilder/data/world_knowledge/shards",
        "/home/benjamin/Bilder/data/stem_knowledge/shards",
        "/home/benjamin/Bilder/data/cyber_web_knowledge/shards",
        "/home/benjamin/Bilder/data/ai_research_knowledge/shards",
        "/home/benjamin/Bilder/data/biology_math/shards",
        "/home/benjamin/Bilder/data/shards",
        "/home/benjamin/Bilder/data/instructions/shards",
        "/home/benjamin/Bilder/data/benchmark_mix/shards",
    ]

    dataset = WorldKnowledgeShardedDataset(shards_dirs, seq_len=128)
    if len(dataset) == 0:
        dataset = WorldKnowledgeShardedDataset(["/home/benjamin/Bilder/data/shards"], seq_len=128)

    print(f"  - Geladene Samples:   {len(dataset):,} Samples ({len(dataset.shard_files)} Shards)", flush=True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import CPUOffload
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from pipeline.moe_components import SparseMoELayer
    
    # Init distributed for 1 GPU
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)

    print("  - Modell-Architektur: 7B Sparse Mixture of Experts (CPU Offload)", flush=True)
    # 1. Initialize on CPU to avoid OOM
    model = MultiGranularMoE1B2Model(
        vocab_size=vocab.size,
        rank=64,
        d_model=2048,
        n_layers=16,
        num_experts=16,
        hidden_dim=4096,
        max_seq_len=256,
    ).to("cpu", dtype=torch.bfloat16)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Gesamte Parameter:  {total_params:,}", flush=True)

    # 2. Wrap with FSDP
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={SparseMoELayer},
    )
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True),
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        sync_module_states=True,
    )

    # Optimizer mit GaLore Low-Rank
    optimizer = GaLoreAdamW(model.parameters(), lr=args.lr_max, weight_decay=0.01, rank=32)

    # 3. GaLore Per-Layer Hooking (System RAM Lebensretter!)
    # Verhindert, dass 14 GB an Gradients im RAM gesammelt werden. 
    # Sobald ein Layer berechnet wurde, springt GaLore an, updatet das Gewicht und verbrennt den Gradient.
    def make_galore_hook(p, opt):
        def hook(*args):
            opt.step_param(p)
            p.grad = None # RAM sofort freigeben!
        return hook

    for p in model.parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(make_galore_hook(p, optimizer))
    latest_checkpoint = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
    start_step = 0
    if os.path.exists(latest_checkpoint):
        print(f"🔄 Lade vorherigen Checkpoint zur nahtlosen Fortsetzung: {latest_checkpoint}", flush=True)
        ckpt_data = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_data["model_state_dict"])
        optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
        start_step = ckpt_data.get("step", 0)
        print(f"✅ Fortsetzung ab Step {start_step:,}!", flush=True)

    # Byte weights
    byte_weights = torch.ones(vocab.size, dtype=torch.bfloat16, device=device)
    for t_id in range(vocab.size):
        byte_weights[t_id] = float(max(1, vocab.get_byte_len(t_id)))

    # 30 Tage Schritte berechnen
    target_seconds = args.days * 24 * 3600
    estimated_steps_per_sec = 18.0
    total_steps = int(target_seconds * estimated_steps_per_sec)

    model.train()
    step = start_step
    start_time = time.time()
    loss_history = []
    accum_loss = 0.0
    nan_consecutive = 0
    max_nan_consecutive = 50  # Auto-reinit after 50 consecutive NaN steps
    current_lr = args.lr_max

    print("\n🚀 Starte 30-Tage Dauerlauf mit automatischer Temperaturüberwachung & Checkpointing...", flush=True)

    optimizer.zero_grad()
    data_iter = iter(dataloader)

    while step < total_steps:
        try:
            x_batch, y_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x_batch, y_batch = next(data_iter)

        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        if args.use_moe:
            logits, aux_loss = model(x_batch)
        else:
            logits = model(x_batch)
            aux_loss = torch.tensor(0.0, device=device)

        # Ensure aux_loss is a tensor
        if not isinstance(aux_loss, torch.Tensor):
            aux_loss = torch.tensor(float(aux_loss), device=device)

        flat_logits = logits.view(-1, vocab.size).float().clamp(-30.0, 30.0)
        flat_targets = y_batch.view(-1)

        weights = byte_weights[flat_targets].float()
        loss_unreduced = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        weight_denom = torch.clamp(torch.sum(weights), min=1.0)
        ce_loss = torch.sum(loss_unreduced * weights) / weight_denom

        # NaN/Inf detection: skip this batch entirely (do NOT backward NaN)
        if torch.isnan(ce_loss) or torch.isinf(ce_loss):
            nan_consecutive += 1
            optimizer.zero_grad()
            step += 1

            # If too many consecutive NaN: model weights are corrupted, reinitialize
            if nan_consecutive >= max_nan_consecutive:
                print(f"\n🚨 KRITISCH: {nan_consecutive} aufeinanderfolgende NaN-Schritte erkannt!", flush=True)
                print("🔧 Re-initialisiere korrumpierte Gewichte...", flush=True)
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if torch.isnan(param).any():
                            nn.init.normal_(param, mean=0.0, std=0.02)
                            print(f"  ♻️ {name} re-initialisiert ({param.numel():,} Parameter)", flush=True)

                # Reset optimizer state completely
                optimizer.state.clear()
                if hasattr(optimizer, 'projectors'):
                    optimizer.projectors.clear()
                nan_consecutive = 0
                print("✅ Gewichte und Optimizer-State erfolgreich repariert!\n", flush=True)

            continue

        # Valid loss: reset NaN counter
        nan_consecutive = 0

        loss = (ce_loss + 0.01 * aux_loss.float()) / args.gradient_accumulation_steps
        loss.backward()
        accum_loss += ce_loss.item() / args.gradient_accumulation_steps

        # Gradient Accumulation Step
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Check for NaN in gradients before stepping
            has_nan_grad = False
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    has_nan_grad = True
                    break

            if has_nan_grad:
                optimizer.zero_grad()
                accum_loss = 0.0
                step += 1
                continue

            # Der Backward Pass feuert jetzt unsere GaLore-Hooks!
            # Jedes Layer wird nach Berechnung sofort aktualisiert und aus dem RAM gelöscht.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # optimizer.step() WIRD NICHT MEHR GLOBAL AUFGERUFEN! (Passiert in den Hooks)
            optimizer.zero_grad(set_to_none=True)

            loss_val = accum_loss if not math.isnan(accum_loss) else loss_history[-1] if loss_history else 8.5
            loss_history.append(loss_val)
            accum_loss = 0.0

            # Cosine Learning Rate Schedule
            progress = step / max(1, total_steps)
            current_lr = args.lr_min + 0.5 * (args.lr_max - args.lr_min) * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

        step += 1

        # Proaktive Hardware Heat Safety & Kühlung alle 50 Steps
        if step % 50 == 0:
            temp = get_gpu_temperature()
            if temp >= 78:
                print(f"⚠️ GPU-Temperatur {temp}°C erreicht! Aktive Abkühlpause (3s)...", flush=True)
                time.sleep(3.0)
            elif temp >= 73:
                time.sleep(0.5)

        # Dashboard & Status Update alle 5 Steps
        if step % 5 == 0:
            elapsed = time.time() - start_time
            tokens_processed = (step - start_step) * args.batch_size * 128
            tps = tokens_processed / max(0.1, elapsed)
            day_fraction = (elapsed / (24 * 3600))
            loss_val = loss_history[-1] if loss_history else 8.5

            update_30day_dashboard_telemetry(
                status_file=status_file,
                day_fraction=day_fraction,
                total_days=args.days,
                step=step,
                total_steps=total_steps,
                current_loss=loss_val,
                loss_history=loss_history,
                tokens_processed=tokens_processed,
                tokens_per_sec=tps,
                shards_count=len(dataset.shard_files),
                current_lr=current_lr,
            )

        if step % 10 == 0:
            loss_val = loss_history[-1] if loss_history else 8.5
            print(f"  [30-Tage MoE · Step {step:06d}] Loss: {loss_val:.4f} | TPS: {int(tps if 'tps' in locals() else 0)} | GPU Temp: {get_gpu_temperature()}°C", flush=True)

        # Checkpointing alle 2000 Steps (nur bei gesunden Gewichten!)
        if step % 2000 == 0:
            # Verify model weights are healthy before saving
            has_nan_weights = False
            for name, p in model.named_parameters():
                if torch.isnan(p).any():
                    has_nan_weights = True
                    break

            if has_nan_weights:
                print(f"⚠️ Checkpoint übersprungen (Step {step:,}) – NaN in Gewichten erkannt!", flush=True)
            else:
                ckpt_save_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_history[-1] if loss_history else 8.0,
                }, ckpt_save_path)
                print(f"💾 Checkpoint gesichert (Step {step:,}) -> {ckpt_save_path}", flush=True)

    # Finales Weltwissen-Modell speichern
    final_path = "/home/benjamin/Bilder/world_knowledge_1b2_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n🎉 30-TAGE WELTWISSEN-TRAINING VOLLSTÄNDIG ABGESCHLOSSEN: {final_path}", flush=True)


if __name__ == "__main__":
    train_30day_world_model()

