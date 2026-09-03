#!/usr/bin/env python3
import argparse
import glob
import math
import os
import random
import sys
sys.path.insert(0, "/home/benjamin/Bilder")
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
import ctypes
import subprocess
try:
    libc = ctypes.CDLL("libc.so.6")
except Exception:
    libc = None

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
    parser = argparse.ArgumentParser(description="8.52B MoE Trainer (20L / 4E / 100.7M pro Experte / 20-Bit Golden Master / 5120 Kontext)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Effektive Batch-Größe: 2x 5120 = 10.240 Tokens pro Optimizer-Schritt")
    parser.add_argument("--seq_len", type=int, default=5120, help="5.120 Tokens Kontext/Batch (5x 1024)")
    parser.add_argument("--galore_rank", type=int, default=64, help="Verdoppelte GaLore-Projektionsgenauigkeit")
    parser.add_argument("--lr_max", type=float, default=2.5e-4)
    parser.add_argument("--lr_min", type=float, default=2.0e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="/home/benjamin/Bilder/checkpoints_8b")
    parser.add_argument("--save_interval", type=int, default=25, help="Speicher-Intervall (25 Steps = ~25 Min, minimiert SSD-I/O)")
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--enable_night_schedule", action="store_true", default=True, help="Pausiert GPU-Training zwischen 21:00 und 09:00 Uhr")
    parser.add_argument("--night_start_hour", type=int, default=21, help="Beginn der Nachtruhe (Stunde)")
    parser.add_argument("--night_end_hour", type=int, default=9, help="Ende der Nachtruhe (Stunde)")
    parser.add_argument("--test_quiet_minutes", type=int, default=0, help="Test-Ruhemodus für X Minuten aktivieren")
    parser.add_argument("--max_steps", type=int, default=0, help="Maximale Trainingsschritte (0 = unendlich)")
    args = parser.parse_args()
    device = torch.device("cuda")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if os.path.exists(MultiGranularVocabulary.CANONICAL_20BIT_BIN_PATH):
        vocab = MultiGranularVocabulary.load_canonical()
    elif os.path.exists("/home/benjamin/Bilder/data/vocab_262k.json"):
        vocab = MultiGranularVocabulary.load_json("/home/benjamin/Bilder/data/vocab_262k.json")
    else:
        vocab = MultiGranularVocabulary.load_json("/home/benjamin/Bilder/data/vocab_65k.json")

    print("  - Initialisiere Dynamischen Trainingsgraphen (Knowledge Curriculum DAG)...", flush=True)
    training_graph = build_default_training_graph("/home/benjamin/Bilder")
    total_shards_all = sum(n.total_shards for n in training_graph.nodes.values())
    print(f"  - Geladene Wissens-Knoten: {len(training_graph.nodes)} Domänen ({total_shards_all} reale Shards, {len(training_graph.edges)} Kanten)", flush=True)
    for n in training_graph.nodes.values():
        print(f"    • [{n.status:<8}] {n.name:<30} ({n.total_shards} Shards)", flush=True)

    print("  - Initialisiere Held-Out Validierungs-Evaluator...", flush=True)
    evaluator = ModelEvaluator()

    print(f"  - Modell-Architektur: 8.42B MoE (Option B: 20 Schichten, 4 Experten à 100.7M, {args.seq_len} Kontext, GaLore r={args.galore_rank})", flush=True)
    
    # -------------------------------------------------------------------------
    # CHECKPOINT RESUME & PRE-TRAINED WARM-START (ZERO-RAM META INIT & MMAP)
    # -------------------------------------------------------------------------
    step = 0
    tokens_processed: int = 0
    loss_history: List[float] = []
    latest_eval_metrics = None
    current_lr: float = args.lr_min
    
    moe_latest_ckpt = os.path.join(args.checkpoint_dir, "8b_checkpoint_latest.pt")
    base_latest_ckpt = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
    
    if os.path.exists(moe_latest_ckpt):
        print(f"  🔄 Initialisiere 8.42B Meta-Modell & lade Checkpoint via Zero-RAM mmap: {moe_latest_ckpt}...", flush=True)
        with torch.device("meta"):
            model = MultiGranularMoE7BModel(
                vocab_size=vocab.size,
                rank=64,
                d_model=2048,
                n_layers=20,
                num_experts=4,
                hidden_dim=16384,
                use_shared_expert=False,
                routing_k=2,
                max_seq_len=args.seq_len,
            )
        model.rope = RotaryEmbedding(dim=64, max_seq_len=args.seq_len)
        
        try:
            ckpt = torch.load(moe_latest_ckpt, map_location="cpu", mmap=True, weights_only=False)
            sd = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt
            model.load_state_dict(sd, assign=True)
            if isinstance(ckpt, dict):
                step = int(ckpt.get("step", 0))
                raw_tokens = ckpt.get("tokens_processed")
                tokens_processed = int(raw_tokens) if raw_tokens is not None else (step * args.seq_len * args.gradient_accumulation_steps)
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
            print(f"  ✅ 8.42B Checkpoint via Zero-RAM mmap erfolgreich geladen! Setze fort bei Step {step:,} ({tokens_processed:,} Tokens).", flush=True)
        except Exception as e:
            print(f"  ⚠️ Warnung beim Laden von {moe_latest_ckpt}: {e}", flush=True)
    else:
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        model = MultiGranularMoE7BModel(
            vocab_size=vocab.size,
            rank=64,
            d_model=2048,
            n_layers=20,
            num_experts=4,
            hidden_dim=16384,
            use_shared_expert=False,
            routing_k=2,
            max_seq_len=args.seq_len,
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

    # JIT LAYER OFFLOADING WRAPPER FUER EXPERTEN
    class OffloadedLinear(torch.nn.Module):
        def __init__(self, linear_layer):
            super().__init__()
            self.in_features = linear_layer.in_features
            self.out_features = linear_layer.out_features
            # Verwende unpinned Speicher: Erlaubt ZRAM-Swapping & Linux Page Cache (32GB Cache/Swap nutzbar)
            self.weight = torch.nn.Parameter(linear_layer.weight.data.to(torch.bfloat16), requires_grad=True)
            if linear_layer.bias is not None:
                self.bias = torch.nn.Parameter(linear_layer.bias.data.to(torch.bfloat16), requires_grad=True)
            else:
                self.bias = None
                
        def forward(self, x):
            w_gpu = self.weight.to(x.device, non_blocking=True)
            b_gpu = self.bias.to(x.device, non_blocking=True) if self.bias is not None else None
            return F.linear(x, w_gpu, b_gpu)

    # FEATURE 1: Attention, Norms, Router & RoPE DAUERHAFT im VRAM verankern (~318 MB)
    print("  - Verankere Attention, LayerNorms & Router dauerhaft im VRAM (~318 MB)...", flush=True)
    model.attn_layers.to(device)
    model.norms1.to(device)
    model.norms2.to(device)
    model.norm_final.to(device)
    model.rope.to(device)
    model.head_proj.to(device)
    model.head_out.to(device)
    model.E_proj.to(device)
    model.E_vocab.to(device)
    for ml in model.moe_layers:
        ml.router.to(device)
        if ml.shared_expert is not None:
            ml.shared_expert.to(device)
        for exp in ml.experts:
            for child_name, child in exp.named_children():
                if isinstance(child, torch.nn.Linear):
                    setattr(exp, child_name, OffloadedLinear(child))

    print(f"  - Initialisiere GaLore Optimizer mit selektivem Weight-Decay und Rang {args.galore_rank}...", flush=True)
    decay_params = []
    nodecay_params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.ndim >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": 0.01, "rank": args.galore_rank},
        {"params": nodecay_params, "weight_decay": 0.0, "rank": args.galore_rank},
    ]
    optimizer = GaLoreAdamW(param_groups, lr=args.lr_max)

    # PER-LAYER GALORE HOOKS MIT GRADIENT ACCUMULATION (MAX NORM = 1.0)
    accum_counter = 0
    def make_galore_hook(p, opt):
        def hook(*args_hook):
            if (accum_counter + 1) % args.gradient_accumulation_steps == 0:
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

    print(f"\n🚀 Starte 8.42B MoE Dauerlauf (Option B) mit {args.seq_len} Kontext & GaLore r={args.galore_rank}...", flush=True)
    print(f"📦 Gradient Accumulation: {args.gradient_accumulation_steps} Steps (effektiv {args.batch_size * args.seq_len * args.gradient_accumulation_steps:,} Tokens pro Gewichtsupdate)", flush=True)
    if args.enable_night_schedule:
        print(f"🌙 Nachtruhe-Automatik aktiv: Täglich von {args.night_start_hour}:00 bis {args.night_end_hour}:00 Uhr (GPU-Standby & 0% GPU-Last).", flush=True)
    if test_quiet_until:
        print(f"🔇 20-Minuten Akustik-Test aktiviert: GPU pausiert sofort bis {time.strftime('%H:%M:%S', time.localtime(test_quiet_until))}.", flush=True)

    step_start_time = time.time()
    accum_losses = []

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
                    tokens_processed=tokens_processed,
                    current_lr=current_lr,
                    graph_state=graph_telemetry,
                    active_node_name=f"💤 Ruhemodus ({pause_reason})",
                    eval_metrics=latest_eval_metrics,
                )
                time.sleep(10)
        
        # Dynamisches Sampling über den Wissensgraphen mit seq_len Tokens
        x, y, active_node_id = training_graph.sample_batch(batch_size=args.batch_size, seq_len=args.seq_len)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        active_node = training_graph.nodes[active_node_id]
        
        # Chunk-Size 128 (reduziert Schleifendurchläufe von 768 auf 48, ohne VRAM-Spikes)
        loss, ce_loss, aux_loss = model.compute_loss(x, y, chunk_size=128)
        
        # Gradient Accumulation: Skaliere Loss
        loss_scaled = loss / args.gradient_accumulation_steps
        loss_scaled.backward()
        
        loss_val = loss.item()
        accum_losses.append(loss_val)
        accum_counter += 1
        
        # Wenn Accumulation-Zyklus vollendet ist: Optimizer-Schritt & Telemetrie
        if accum_counter % args.gradient_accumulation_steps == 0:
            step += 1
            mean_loss = sum(accum_losses) / len(accum_losses)
            accum_losses.clear()
            
            # Melde Loss an den Wissensgraphen mit aktuellem Step (Anti-Thrashing Cooldown)
            training_graph.report_batch_loss(active_node_id, mean_loss, current_step=step)
            
            # Linear Warmup (1.000 Steps) gefolgt von sanftem Cosine Decay
            if step < args.warmup_steps:
                current_lr = args.lr_min + (args.lr_max - args.lr_min) * (step / max(1, args.warmup_steps))
            else:
                progress = (step - args.warmup_steps) / max(1, 1000000 - args.warmup_steps)
                current_lr = args.lr_min + 0.5 * (args.lr_max - args.lr_min) * (1.0 + math.cos(math.pi * progress))

            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            effective_tokens = args.batch_size * args.seq_len * args.gradient_accumulation_steps
            tokens_processed += effective_tokens
            loss_history.append(mean_loss)
            
            step_now = time.time()
            step_duration = step_now - step_start_time
            instant_tps = effective_tokens / max(0.05, step_duration)
            
            # Periodische Held-Out Validierung (alle 50 Steps)
            if step > 0 and step % args.eval_interval == 0:
                eval_res = evaluator.evaluate_model(model, device, num_batches=4, seq_len=512, step=step)
                latest_eval_metrics = eval_res
                print(f"  📊 [HELD-OUT EVALUATION · Step {step:06d}] Val-Loss: {eval_res['val_loss']:.4f} | Perplexität (PPL): {eval_res['perplexity']:.1f} | Acc: {eval_res['accuracy_pct']}%", flush=True)

            # Schreibe JEDEN Step live in Konsole und Dashboard mit Graph-Knoten Info
            node_tag = f"{active_node.name[:18]}"
            print(f"  [8.4B MoE · Step {step:06d}] [{node_tag:<18}] Loss: {mean_loss:.4f} (Avg: {active_node.moving_loss:.2f}) | TPS: {int(instant_tps)} | Dauer: {step_duration:.1f}s", flush=True)
            
            graph_telemetry = training_graph.get_telemetry_state()
            live_shards_count = sum(n["total_shards"] for n in graph_telemetry["nodes"])
            
            update_30day_dashboard_telemetry(
                status_file=status_file,
                day_fraction=step / 1000000,
                total_days=30.0,
                step=step,
                total_steps=1000000,
                current_loss=mean_loss,
                loss_history=loss_history,
                tokens_processed=tokens_processed,
                tokens_per_sec=instant_tps,
                shards_count=live_shards_count,
                current_lr=current_lr,
                graph_state=graph_telemetry,
                active_node_name=active_node.name,
                eval_metrics=latest_eval_metrics,
            )


            # Periodische Speicherhygiene & Thermal-Guard (alle 10 Steps)
            if step > 0 and step % 10 == 0:
                torch.cuda.empty_cache()
                if libc and hasattr(libc, "malloc_trim"):
                    libc.malloc_trim(0)
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                        encoding="utf-8"
                    ).strip()
                    gpu_temp = int(out) if out.isdigit() else 0
                    if gpu_temp >= 84:
                        print(f"  🌡️ [THERMAL-GUARD] GPU-Temperatur bei {gpu_temp}°C (Drosselgefahr). 1.5s Pacing zur Kühlung...", flush=True)
                        time.sleep(1.5)
                except Exception:
                    pass

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
                latest_path = os.path.join(args.checkpoint_dir, "8b_checkpoint_latest.pt")
                temp_path = os.path.join(args.checkpoint_dir, "8b_checkpoint_temp.pt")
                
                # Atomarer Write: Verhindert beschädigte Dateien bei Unterbrechung
                torch.save(ckpt_data, temp_path)
                os.replace(temp_path, latest_path)

                # Checkpoint benennen via 0-ms Hardlink (spart 16,6 GB SSD-Schreiblast!)
                ckpt_path = os.path.join(args.checkpoint_dir, f"8b_checkpoint_step_{step}.pt")
                try:
                    if os.path.exists(ckpt_path):
                        os.remove(ckpt_path)
                    os.link(latest_path, ckpt_path)
                except Exception:
                    try:
                        shutil.copyfile(latest_path, ckpt_path)
                    except Exception:
                        pass

                # Checkpoint Retention: Behalte Meilensteine (alle 500 Steps) + letzte 3 Step-Checkpoints
                try:
                    all_step_ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, "8b_checkpoint_step_*.pt")),
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

            # Regelmäßige Speicher-Defragmentierung & glibc Heap-Freigabe an Linux Kernel
            if step % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    import ctypes
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass

            if args.max_steps > 0 and step >= args.max_steps:
                print(f"🏁 Zielschritte erreicht ({step}/{args.max_steps}). Beende Trainer.", flush=True)
                break

            step_start_time = time.time()

if __name__ == "__main__":
    train_30day_world_model()
