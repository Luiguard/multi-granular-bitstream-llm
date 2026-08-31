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
import gc
import shutil
import functools
import numpy as np

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamDecoder
from pipeline.galore_optimizer import GaLoreAdamW
from pipeline.moe_7b_model import MultiGranularMoE7BModel
from pipeline.nemotron_components import RotaryEmbedding

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
from pipeline.model_evaluator import ModelEvaluator

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
    eval_metrics: Optional[Dict[str, Any]] = None,
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
        "eval_metrics": eval_metrics,
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
    parser.add_argument("--lr_max", type=float, default=2.5e-4)
    parser.add_argument("--lr_min", type=float, default=2.0e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="/home/benjamin/Bilder/checkpoints")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--enable_night_schedule", action="store_true", default=True, help="Pausiert GPU-Training zwischen 21:00 und 09:00 Uhr")
    parser.add_argument("--night_start_hour", type=int, default=21, help="Beginn der Nachtruhe (Stunde)")
    parser.add_argument("--night_end_hour", type=int, default=9, help="Ende der Nachtruhe (Stunde)")
    parser.add_argument("--test_quiet_minutes", type=int, default=0, help="Test-Ruhemodus für X Minuten aktivieren")
    args = parser.parse_args()
    device = torch.device("cuda")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    vocab_file = "/home/benjamin/Bilder/data/vocab_262k.json"
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/data/vocab_65k.json"
    vocab = MultiGranularVocabulary.load_json(vocab_file)

    print("  - Initialisiere Dynamischen Trainingsgraphen (Knowledge Curriculum DAG)...", flush=True)
    training_graph = build_default_training_graph("/home/benjamin/Bilder")
    total_shards_all = sum(n.total_shards for n in training_graph.nodes.values())
    print(f"  - Geladene Wissens-Knoten: {len(training_graph.nodes)} Domänen ({total_shards_all} reale Shards, {len(training_graph.edges)} Kanten)", flush=True)
    for n in training_graph.nodes.values():
        print(f"    • [{n.status:<8}] {n.name:<30} ({n.total_shards} Shards)", flush=True)

    print("  - Initialisiere Held-Out Validierungs-Evaluator...", flush=True)
    evaluator = ModelEvaluator()

    print("  - Modell-Architektur: 7B Sparse Mixture of Experts (JIT Layer Offloading / GaLore Hooks)", flush=True)
    
    # -------------------------------------------------------------------------
    # CHECKPOINT RESUME & PRE-TRAINED WARM-START (ZERO-RAM META INIT & MMAP)
    # -------------------------------------------------------------------------
    step = 0
    tokens_processed: int = 0
    loss_history: List[float] = []
    latest_eval_metrics = None
    current_lr: float = args.lr_min
    
    moe_latest_ckpt = os.path.join(args.checkpoint_dir, "7b_checkpoint_latest.pt")
    base_latest_ckpt = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
    
    if os.path.exists(moe_latest_ckpt):
        print(f"  🔄 Initialisiere 7B Meta-Modell & lade Checkpoint via Zero-RAM mmap: {moe_latest_ckpt}...", flush=True)
        with torch.device("meta"):
            model = MultiGranularMoE7BModel(
                vocab_size=vocab.size,
                rank=64,
                d_model=2048,
                n_layers=24,
                num_experts=12,
                hidden_dim=4096,
                max_seq_len=7168,
            )
        model.rope = RotaryEmbedding(dim=64, max_seq_len=7168)
        
        try:
            ckpt = torch.load(moe_latest_ckpt, map_location="cpu", mmap=True, weights_only=False)
            sd = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
            model.load_state_dict(sd, assign=True)
            if isinstance(ckpt, dict):
                step = ckpt.get("step", 0)
                tokens_processed = ckpt.get("tokens_processed", step * 7168)
                loss_history = ckpt.get("loss_history", [])
                latest_eval_metrics = ckpt.get("latest_eval_metrics")
                if "training_graph_state" in ckpt:
                    training_graph.load_telemetry_state(ckpt["training_graph_state"])
                    print("  🧠 Wissensgraph-Zustand (Knoten, Losses & Curricula) vollständig wiederhergestellt!", flush=True)
                
                # Restore RNG states if present
                if "rng_states" in ckpt and ckpt["rng_states"]:
                    try:
                        rng = ckpt["rng_states"]
                        if "torch" in rng and rng["torch"] is not None:
                            torch.set_rng_state(rng["torch"])
                        if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
                            torch.cuda.set_rng_state(rng["cuda"])
                    except Exception:
                        pass
                
                current_total_samples = sum(n.sample_count for n in training_graph.nodes.values())
                if step > current_total_samples:
                    missing_samples = step - current_total_samples
                    training_graph.nodes["node_0_foundation"].sample_count += missing_samples
                    training_graph.update_gating()
            del ckpt, sd
            gc.collect()
            print(f"  ✅ 7B Checkpoint via Zero-RAM mmap erfolgreich geladen! Setze fort bei Step {step:,} ({tokens_processed:,} Tokens).", flush=True)
        except Exception as e:
            print(f"  ⚠️ Warnung beim Laden von {moe_latest_ckpt}: {e}", flush=True)
    else:
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
        
        if os.path.exists(base_latest_ckpt):
            print(f"  🌱 Initialisiere Warm-Start aus Basismodell-Checkpoint: {base_latest_ckpt}...", flush=True)
            try:
                ckpt = torch.load(base_latest_ckpt, map_location="cpu", mmap=True, weights_only=False)
                sd = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
                with torch.no_grad():
                    model_state = model.state_dict()
                    for k, v in sd.items():
                        if k in model_state and model_state[k].shape == v.shape:
                            model_state[k].copy_(v.to(dtype=model_state[k].dtype))
                del ckpt, sd
            except Exception as e:
                print(f"  ⚠️ Warnung beim Warm-Start: {e}", flush=True)

    # JIT LAYER OFFLOADING
    class OffloadedLinear(torch.nn.Module):
        def __init__(self, linear_layer):
            super().__init__()
            self.in_features = linear_layer.in_features
            self.out_features = linear_layer.out_features
            self.weight = torch.nn.Parameter(linear_layer.weight.data.to(torch.bfloat16).pin_memory(), requires_grad=True)
            self.bias = torch.nn.Parameter(linear_layer.bias.data.to(torch.bfloat16).pin_memory(), requires_grad=True) if linear_layer.bias is not None else None
            
        def forward(self, x):
            w_gpu = self.weight.to(x.device, non_blocking=True)
            b_gpu = self.bias.to(x.device, non_blocking=True) if self.bias is not None else None
            return F.linear(x, w_gpu, b_gpu)

    class OffloadedEmbedding(torch.nn.Module):
        def __init__(self, emb_layer):
            super().__init__()
            self.num_embeddings = emb_layer.num_embeddings
            self.embedding_dim = emb_layer.embedding_dim
            self.weight = torch.nn.Parameter(emb_layer.weight.data.to(torch.bfloat16).pin_memory(), requires_grad=True)
            
        def forward(self, x):
            w_gpu = self.weight.to(x.device, non_blocking=True)
            return F.embedding(x, w_gpu)

    class OffloadedLayerNorm(torch.nn.Module):
        def __init__(self, norm_layer):
            super().__init__()
            self.normalized_shape = norm_layer.normalized_shape
            self.weight = torch.nn.Parameter(norm_layer.weight.data.to(torch.bfloat16).pin_memory(), requires_grad=True) if norm_layer.weight is not None else None
            self.bias = torch.nn.Parameter(norm_layer.bias.data.to(torch.bfloat16).pin_memory(), requires_grad=True) if norm_layer.bias is not None else None
            self.eps = norm_layer.eps
            
        def forward(self, x):
            w_gpu = self.weight.to(x.device, non_blocking=True) if self.weight is not None else None
            b_gpu = self.bias.to(x.device, non_blocking=True) if self.bias is not None else None
            return F.layer_norm(x, self.normalized_shape, w_gpu, b_gpu, self.eps)

    def convert_to_offloaded(module):
        for name, child in module.named_children():
            if name in ["head_proj", "head_out", "E_proj"]:
                child.to(device)
            elif isinstance(child, torch.nn.Linear):
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

    # PER-LAYER GALORE HOOKS MIT GRADIENT CLIPPING (MAX NORM = 1.0)
    def make_galore_hook(p, opt):
        def hook(*args):
            if p.grad is not None:
                torch.nn.utils.clip_grad_norm_([p], max_norm=1.0)
            opt.step_param(p)
            p.grad = None  # FREE RAM INSTANTLY
        return hook

    for p in model.parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(make_galore_hook(p, optimizer))

    model.train()
    start_time = time.time()
    status_file = "/home/benjamin/Bilder/data/training_status.json"
    test_quiet_until = (time.time() + args.test_quiet_minutes * 60) if args.test_quiet_minutes > 0 else None

    print("\n🚀 Starte 7B Dauerlauf mit Dynamischem Trainingsgraphen (7168 Kontext) & GaLore...", flush=True)
    if args.enable_night_schedule:
        print(f"🌙 Nachtruhe-Automatik aktiv: Täglich von {args.night_start_hour}:00 bis {args.night_end_hour}:00 Uhr (GPU-Standby & 0% GPU-Last).", flush=True)
    if test_quiet_until:
        print(f"🔇 20-Minuten Akustik-Test aktiviert: GPU pausiert sofort bis {time.strftime('%H:%M:%S', time.localtime(test_quiet_until))}.", flush=True)

    while step < 1000000:
        # Prüfe auf Nachtruhe (21:00 - 09:00 Uhr) oder temporären Akustik-Test
        now_ts = time.time()
        now_hour = time.localtime().tm_hour
        in_test = (test_quiet_until is not None and now_ts < test_quiet_until)
        in_regular_night = (now_hour >= args.night_start_hour or now_hour < args.night_end_hour) if args.enable_night_schedule else False

        if in_test or in_regular_night:
            pause_reason = "20-Minuten Akustik-Test" if in_test else f"Nachtruhe ({args.night_start_hour}:00 - {args.night_end_hour}:00 Uhr)"
            torch.cuda.empty_cache()
            gc.collect()
            print(f"\n🌙 [RUHEMODUS AKTIV] GPU-Training pausiert ({pause_reason}).", flush=True)
            print("   💤 GPU VRAM freigegeben, 0% GPU-Last, Lüfter kühlen ab. Nur Shard-Erstellung läuft auf Low-CPU.\n", flush=True)

            while True:
                now_ts = time.time()
                now_hour = time.localtime().tm_hour
                in_test = (test_quiet_until is not None and now_ts < test_quiet_until)
                in_regular_night = (now_hour >= args.night_start_hour or now_hour < args.night_end_hour) if args.enable_night_schedule else False
                if not in_test and not in_regular_night:
                    print(f"\n☀️ [AUFWACH-MODUS] Ruhephase beendet ({pause_reason}). Reaktiviere GPU-Training bei Step {step}...\n", flush=True)
                    break

                graph_telemetry = training_graph.get_telemetry_state()
                live_shards_count = sum(n["total_shards"] for n in graph_telemetry["nodes"])
                update_30day_dashboard_telemetry(
                    status_file=status_file,
                    day_fraction=step / 1000000,
                    total_days=30.0,
                    step=step,
                    total_steps=1000000,
                    tokens_per_sec=0.0,
                    current_loss=loss_history[-1] if loss_history else 8.0,
                    shards_count=live_shards_count,
                    loss_history=loss_history,
                    tokens_processed=int(tokens_processed) if tokens_processed is not None else 0,
                    current_lr=current_lr,
                    graph_state=graph_telemetry,
                    active_node_name=f"💤 Ruhemodus ({pause_reason})",
                    eval_metrics=latest_eval_metrics,
                )
                time.sleep(10)

        step_start_time = time.time()
        
        # Dynamisches Sampling über den Wissensgraphen mit 7168 Tokens
        x, y, active_node_id = training_graph.sample_batch(batch_size=args.batch_size, seq_len=7168)
        x, y = x.to(device), y.to(device)
        active_node = training_graph.nodes[active_node_id]
        
        torch.cuda.empty_cache()
        loss, ce_loss, aux_loss = model.compute_loss(x, y, chunk_size=128)
        
        # Backward pass automatically fires the GaLore hooks per-layer!
        loss.backward()
        
        # Free CUDA cache heavily to prevent VRAM fragmentation
        torch.cuda.empty_cache()
        
        loss_val = loss.item()
        
        # Melde Loss an den Wissensgraphen mit aktuellem Step (Anti-Thrashing Cooldown)
        training_graph.report_batch_loss(active_node_id, loss_val, current_step=step)
        
        # Linear Warmup (1.000 Steps) gefolgt von sanftem Cosine Decay
        if step < args.warmup_steps:
            current_lr = args.lr_min + (args.lr_max - args.lr_min) * (step / max(1, args.warmup_steps))
        else:
            progress = (step - args.warmup_steps) / max(1, 1000000 - args.warmup_steps)
            current_lr = args.lr_min + 0.5 * (args.lr_max - args.lr_min) * (1.0 + math.cos(math.pi * progress))

        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        tokens_processed += args.batch_size * 7168
        loss_history.append(loss_val)
        
        step_now = time.time()
        step_duration = step_now - step_start_time
        instant_tps = (args.batch_size * 7168) / max(0.05, step_duration)
        
        # Periodische Held-Out Validierung (alle 50 Steps)
        if step > 0 and step % args.eval_interval == 0:
            eval_res = evaluator.evaluate_model(model, device, num_batches=4, seq_len=512, step=step)
            latest_eval_metrics = eval_res
            print(f"  📊 [HELD-OUT EVALUATION · Step {step:06d}] Val-Loss: {eval_res['val_loss']:.4f} | Perplexität (PPL): {eval_res['perplexity']:.1f} | Acc: {eval_res['accuracy_pct']}%", flush=True)

        # Schreibe JEDEN Step live in Konsole und Dashboard mit Graph-Knoten Info
        node_tag = f"{active_node.name[:18]}"
        print(f"  [7B MoE · Step {step:06d}] [{node_tag:<18}] Loss: {loss_val:.4f} (Avg: {active_node.moving_loss:.2f}) | TPS: {int(instant_tps)} | Dauer: {step_duration:.1f}s", flush=True)
        
        graph_telemetry = training_graph.get_telemetry_state()
        live_shards_count = sum(n["total_shards"] for n in graph_telemetry["nodes"])
        
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
            shards_count=live_shards_count,
            current_lr=current_lr,
            graph_state=graph_telemetry,
            active_node_name=active_node.name,
            eval_metrics=latest_eval_metrics,
        )

        if step > 0 and step % args.save_interval == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt_data = {
                "step": step,
                "tokens_processed": tokens_processed,
                "loss_history": loss_history[-100:],
                "training_graph_state": training_graph.get_telemetry_state(),
                "latest_eval_metrics": latest_eval_metrics,
                "model_state_dict": model.state_dict(),
                "rng_states": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    "random": random.getstate(),
                    "numpy": np.random.get_state()
                }
            }
            latest_path = os.path.join(args.checkpoint_dir, "7b_checkpoint_latest.pt")
            temp_path = os.path.join(args.checkpoint_dir, "7b_checkpoint_temp.pt")
            
            # Atomarer Write: Verhindert beschädigte Dateien bei Unterbrechung
            torch.save(ckpt_data, temp_path)
            os.replace(temp_path, latest_path)

            # Nur alle 10 Schritte einen benannten Checkpoint anlegen und ältere aufräumen
            if step % (args.save_interval * 2) == 0 or step % 500 == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"7b_checkpoint_step_{step}.pt")
                try:
                    shutil.copyfile(latest_path, ckpt_path)
                except Exception:
                    pass

                # Checkpoint Retention: Behalte Meilensteine (alle 500 Steps) + letzte 3 Step-Checkpoints
                try:
                    all_step_ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, "7b_checkpoint_step_*.pt")),
                                            key=lambda p: int(p.split("_step_")[-1].replace(".pt", "")) if "_step_" in p else 0)
                    if len(all_step_ckpts) > 3:
                        for old_ckpt in all_step_ckpts[:-3]:
                            step_num = int(old_ckpt.split("_step_")[-1].replace(".pt", ""))
                            if step_num % 500 != 0:  # Meilensteine behalten
                                try:
                                    os.remove(old_ckpt)
                                except Exception:
                                    pass
                except Exception:
                    pass

            del ckpt_data
            gc.collect()
            print(f"  💾 Checkpoint atomar gesichert: Step {step} ({tokens_processed:,} Tokens · Wissensgraph & Gewichte gesichert)", flush=True)

        step += 1

if __name__ == "__main__":
    train_30day_world_model()
