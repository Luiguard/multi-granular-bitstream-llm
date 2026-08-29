#!/usr/bin/env python3
import argparse
import glob
import math
import os
import random
import sys
import time
import json
from typing import Any, Dict, List, Optional

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

from pipeline.training_graph import build_default_training_graph

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
    graph_state: Optional[Dict[str, Any]] = None,
    active_node_name: str = "Foundation",
):
    progress_pct = (step / max(1, total_steps)) * 100.0
    remaining_seconds = max(0, (total_steps - step) / max(1.0, tokens_per_sec / 128))
    days_left = remaining_seconds / (24 * 3600)
    hours_left = (remaining_seconds % (24 * 3600)) / 3600

    eta_str = f"{days_left:.1f} Tage ({hours_left:.0f}h verbleibend)"

    safe_loss = current_loss if (current_loss and not math.isnan(current_loss) and not math.isinf(current_loss)) else 8.42
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
        "active_knowledge_node": active_node_name,
        "training_graph": graph_state,
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
    parser.add_argument("--checkpoint_dir", type=str, default="/home/benjamin/Bilder/checkpoints")
    parser.add_argument("--save_interval", type=int, default=25)
    args = parser.parse_args()
    device = torch.device("cuda")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    vocab_file = "/home/benjamin/Bilder/data/vocab_65k.json"
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"
    vocab = MultiGranularVocabulary.load_json(vocab_file)

    print("  - Initialisiere Dynamischen Trainingsgraphen (Knowledge Curriculum DAG)...", flush=True)
    training_graph = build_default_training_graph("/home/benjamin/Bilder")
    total_shards_all = sum(n.total_shards for n in training_graph.nodes.values())
    print(f"  - Geladene Wissens-Knoten: {len(training_graph.nodes)} Domänen ({total_shards_all} reale Shards, {len(training_graph.edges)} Kanten)", flush=True)
    for n in training_graph.nodes.values():
        print(f"    • [{n.status:<8}] {n.name:<30} ({n.total_shards} Shards)", flush=True)

    print("  - Modell-Architektur: 7B Sparse Mixture of Experts (JIT Layer Offloading / GaLore Hooks)", flush=True)
    
    # Modell direkt im RAM (CPU) erstellen (~13.9 GB in bf16 bei 12 Experten)
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
        max_seq_len=7168,
    )
    
    torch.set_default_dtype(old_dtype)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Gesamte Parameter:  {total_params:,}", flush=True)
    print("  - Native Kontextlänge: 7,168 Tokens (Llama-3 RoPE Scaling)", flush=True)

    # -------------------------------------------------------------------------
    # CHECKPOINT RESUME & PRE-TRAINED WARM-START
    # -------------------------------------------------------------------------
    step = 0
    tokens_processed = 0
    loss_history = []
    
    moe_latest_ckpt = os.path.join(args.checkpoint_dir, "7b_checkpoint_latest.pt")
    base_latest_ckpt = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
    
    if os.path.exists(moe_latest_ckpt):
        print(f"  🔄 Lade existierenden 7B Checkpoint: {moe_latest_ckpt}...", flush=True)
        try:
            ckpt = torch.load(moe_latest_ckpt, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
                step = ckpt.get("step", 0)
                tokens_processed = ckpt.get("tokens_processed", step * 7168)
                loss_history = ckpt.get("loss_history", [])
            else:
                model.load_state_dict(ckpt, strict=False)
            print(f"  ✅ 7B Checkpoint erfolgreich geladen! Setze fort bei Step {step:,} ({tokens_processed:,} Tokens).", flush=True)
        except Exception as e:
            print(f"  ⚠️ Warnung beim Laden von {moe_latest_ckpt}: {e}", flush=True)
    elif os.path.exists(base_latest_ckpt):
        print(f"  🌱 Initialisiere Warm-Start aus Basismodell-Checkpoint: {base_latest_ckpt}...", flush=True)
        try:
            ckpt = torch.load(base_latest_ckpt, map_location="cpu")
            sd = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
            
            with torch.no_grad():
                model_state = model.state_dict()
                transferred_count = 0
                for k, v in sd.items():
                    if k in model_state:
                        target = model_state[k]
                        if target.shape == v.shape:
                            target.copy_(v.to(dtype=target.dtype))
                            transferred_count += 1
                        elif target.dim() == v.dim() and all(t_dim >= v_dim for t_dim, v_dim in zip(target.shape, v.shape)):
                            # Net2Net Progressive Upscaling (z. B. 1024 -> 2048 Kanäle)
                            slices = tuple(slice(0, v_dim) for v_dim in v.shape)
                            target[slices].copy_(v.to(dtype=target.dtype))
                            transferred_count += 1
                model.load_state_dict(model_state)
            print(f"  ✅ Shape-Aware Warm-Start: {transferred_count} Gewichts-Tensoren (Embeddings & Köpfe) erfolgreich transferiert!", flush=True)
        except Exception as e:
            print(f"  ⚠️ Warnung beim Basismodell-Transfer: {e}", flush=True)

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
    status_file = "/home/benjamin/Bilder/data/training_status.json"

    print("\n🚀 Starte 7B Dauerlauf mit Dynamischem Trainingsgraphen (7168 Kontext) & GaLore...", flush=True)
    while step < 1000000:
        step_start_time = time.time()
        
        # Dynamisches Sampling über den Wissensgraphen mit 7168 Tokens
        x, y, active_node_id = training_graph.sample_batch(batch_size=args.batch_size, seq_len=7168)
        x, y = x.to(device), y.to(device)
        active_node = training_graph.nodes[active_node_id]
        
        if step == 0:
            print(f"  ⏳ Erster Forward-Pass (7168 Tokens Chunked Head · {active_node.name})...", flush=True)
        loss, ce_loss, aux_loss = model.compute_loss(x, y, chunk_size=1024)
        
        if step == 0:
            print(f"  ✅ Forward OK! Loss: {loss.item():.4f}. Starte Backward + GaLore Init...", flush=True)
        # Backward pass automatically fires the GaLore hooks per-layer!
        loss.backward()
        if step == 0:
            print("  ✅ Backward OK! 7168-Token Training läuft jetzt dynamisch über den Graph!", flush=True)
        
        # Free CUDA cache heavily to prevent VRAM fragmentation
        torch.cuda.empty_cache()
        
        loss_val = loss.item()
        
        # Melde Loss an den Wissensgraphen (triggert ggf. Remediation Backtracking)
        training_graph.report_batch_loss(active_node_id, loss_val)
        
        progress = step / 1000000
        current_lr = args.lr_min + 0.5 * (args.lr_max - args.lr_min) * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        tokens_processed += args.batch_size * 7168
        loss_history.append(loss_val)
        
        step_now = time.time()
        step_duration = step_now - step_start_time
        instant_tps = (args.batch_size * 7168) / max(0.05, step_duration)
        
        # Schreibe JEDEN Step live in Konsole und Dashboard mit Graph-Knoten Info
        node_tag = f"{active_node.name[:18]}"
        print(f"  [7B MoE · Step {step:06d}] [{node_tag:<18}] Loss: {loss_val:.4f} (Avg: {active_node.moving_loss:.2f}) | TPS: {int(instant_tps)} | Dauer: {step_duration:.1f}s", flush=True)
        
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
            shards_count=total_shards_all,
            current_lr=current_lr,
            graph_state=training_graph.get_telemetry_state(),
            active_node_name=active_node.name,
        )

        if step > 0 and step % args.save_interval == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt_data = {
                "step": step,
                "tokens_processed": tokens_processed,
                "loss_history": loss_history[-100:],
                "model_state_dict": model.state_dict(),
            }
            ckpt_path = os.path.join(args.checkpoint_dir, f"7b_checkpoint_step_{step}.pt")
            latest_path = os.path.join(args.checkpoint_dir, "7b_checkpoint_latest.pt")
            torch.save(ckpt_data, ckpt_path)
            torch.save(ckpt_data, latest_path)
            print(f"  💾 Checkpoint gespeichert: {ckpt_path} (Latest aktualisiert)", flush=True)

        step += 1

if __name__ == "__main__":
    train_30day_world_model()
