import os
import sys
import time
import gc
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/benjamin/Bilder")
from pipeline.moe_7b_model import MultiGranularMoE7BModel
from pipeline.nemotron_components import RotaryEmbedding
from pipeline.galore_optimizer import GaLoreAdamW

def run_stress_test():
    print("=" * 80)
    print("🔥 ECHTER 30-SEKUNDEN STRESS-TEST: OPTION 1 (8.42B MoE, 20 SCHICHTEN)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("❌ FEHLER: Keine CUDA GPU gefunden!")
        return
        
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"  • Hardware: {gpu_name} ({total_vram_mb:.0f} MB VRAM)")
    
    vm = psutil.virtual_memory()
    print(f"  • System-RAM: {vm.total / (1024**3):.1f} GB Gesamt, {vm.available / (1024**3):.1f} GB Verfügbar")
    
    # Architektur-Parameter für Option 1
    vocab_size = 65536
    rank = 64
    d_model = 2048
    n_layers = 20
    num_experts = 4
    hidden_dim = 16384  # 100.7 Mio. pro Experte!
    max_seq_len = 2048
    routing_k = 2
    batch_size = 1
    
    print(f"\n[1] INITIALISIERUNG DER ARCHITEKTUR:")
    print(f"  • Schichten:        {n_layers}")
    print(f"  • Experten/Schicht: {num_experts}")
    print(f"  • hidden_dim:       {hidden_dim} (100.7 Mio. Parameter pro Experte)")
    print(f"  • Routing:          Top-{routing_k}")
    print(f"  • Kontextlänge:     {max_seq_len} Tokens")
    print(f"  • Batch-Size:       {batch_size}")
    
    # Modelinstanziierung in bfloat16
    print("  • Erstelle Modell in bfloat16 im CPU-Speicher...", flush=True)
    t0 = time.time()
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    
    model = MultiGranularMoE7BModel(
        vocab_size=vocab_size,
        rank=rank,
        d_model=d_model,
        n_layers=n_layers,
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        use_shared_expert=False,
        routing_k=routing_k,
        max_seq_len=max_seq_len,
    )
    torch.set_default_dtype(old_dtype)
    print(f"  ✅ Modell erstellt in {time.time() - t0:.2f}s!")
    
    # JIT Layer Offloading Wrappers
    class OffloadedLinear(torch.nn.Module):
        def __init__(self, linear_layer):
            super().__init__()
            self.in_features = linear_layer.in_features
            self.out_features = linear_layer.out_features
            # Versuch mit pin_memory, fallback ohne
            try:
                self.weight = torch.nn.Parameter(linear_layer.weight.data.to(torch.bfloat16).pin_memory(), requires_grad=True)
            except Exception:
                self.weight = torch.nn.Parameter(linear_layer.weight.data.to(torch.bfloat16), requires_grad=True)
            if linear_layer.bias is not None:
                try:
                    self.bias = torch.nn.Parameter(linear_layer.bias.data.to(torch.bfloat16).pin_memory(), requires_grad=True)
                except Exception:
                    self.bias = torch.nn.Parameter(linear_layer.bias.data.to(torch.bfloat16), requires_grad=True)
            else:
                self.bias = None
                
        def forward(self, x):
            w_gpu = self.weight.to(x.device, non_blocking=True)
            b_gpu = self.bias.to(x.device, non_blocking=True) if self.bias is not None else None
            return F.linear(x, w_gpu, b_gpu)

    class OffloadedEmbedding(torch.nn.Module):
        def __init__(self, emb_layer):
            super().__init__()
            self.num_embeddings = emb_layer.num_embeddings
            self.embedding_dim = emb_layer.embedding_dim
            try:
                self.weight = torch.nn.Parameter(emb_layer.weight.data.to(torch.bfloat16).pin_memory(), requires_grad=True)
            except Exception:
                self.weight = torch.nn.Parameter(emb_layer.weight.data.to(torch.bfloat16), requires_grad=True)
                
        def forward(self, x):
            w_gpu = self.weight.to(x.device, non_blocking=True)
            return F.embedding(x, w_gpu)

    class OffloadedLayerNorm(torch.nn.Module):
        def __init__(self, norm_layer):
            super().__init__()
            self.normalized_shape = norm_layer.normalized_shape
            try:
                self.weight = torch.nn.Parameter(norm_layer.weight.data.to(torch.bfloat16).pin_memory(), requires_grad=True) if norm_layer.weight is not None else None
            except Exception:
                self.weight = torch.nn.Parameter(norm_layer.weight.data.to(torch.bfloat16), requires_grad=True) if norm_layer.weight is not None else None
            try:
                self.bias = torch.nn.Parameter(norm_layer.bias.data.to(torch.bfloat16).pin_memory(), requires_grad=True) if norm_layer.bias is not None else None
            except Exception:
                self.bias = torch.nn.Parameter(norm_layer.bias.data.to(torch.bfloat16), requires_grad=True) if norm_layer.bias is not None else None
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

    print("  • Konvertiere Schichten in JIT Offloaded Module...", flush=True)
    convert_to_offloaded(model)
    
    # GaLore Optimizer mit Per-Layer Hook
    optimizer = GaLoreAdamW(model.parameters(), lr=2e-4, weight_decay=0.01, rank=32)
    def make_galore_hook(p, opt):
        def hook(*args):
            if p.grad is not None:
                torch.nn.utils.clip_grad_norm_([p], max_norm=1.0)
            opt.step_param(p)
            p.grad = None
        return hook

    for p in model.parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(make_galore_hook(p, optimizer))
            
    model.train()
    
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print("🚀 STARTE REALE 30-SEKUNDEN STRESS-MESSUNG AUF DER GPU...")
    print("=" * 80)
    print(f"{'Zeit':<6} | {'Step':<5} | {'VRAM Belegt':<12} | {'VRAM Peak':<11} | {'VRAM Freir':<11} | {'Sys-RAM':<10} | {'Step-Zeit':<10} | {'Status'}")
    print("-" * 80)
    
    test_duration = 30.0
    start_time = time.time()
    step = 0
    
    max_peak_vram_mb = 0.0
    min_free_vram_mb = total_vram_mb
    oom_occurred = False
    error_msg = ""
    
    while time.time() - start_time < test_duration:
        step_start = time.time()
        step += 1
        elapsed = time.time() - start_time
        
        # Erzeuge echten Batch von 2048 Tokens
        x = torch.randint(0, vocab_size, (batch_size, max_seq_len), dtype=torch.long, device=device)
        y = torch.randint(0, vocab_size, (batch_size, max_seq_len), dtype=torch.long, device=device)
        
        try:
            # Forward Pass & Chunked Cross-Entropy Loss
            loss, ce_loss, aux_loss = model.compute_loss(x, y, chunk_size=128)
            
            # Backward Pass (löst per-layer GaLore Hooks aus)
            loss.backward()
            
            # Cache leeren
            torch.cuda.empty_cache()
            
            # Messungen
            curr_vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
            peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
            reserved_vram_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
            free_vram_mb = total_vram_mb - reserved_vram_mb
            
            max_peak_vram_mb = max(max_peak_vram_mb, peak_vram_mb)
            min_free_vram_mb = min(min_free_vram_mb, free_vram_mb)
            
            sys_ram_gb = psutil.virtual_memory().used / (1024 ** 3)
            step_time = time.time() - step_start
            
            status = "🟢 SICHER"
            if free_vram_mb < 500:
                status = "🟡 ENG (<500MB)"
            if free_vram_mb < 200:
                status = "🔴 KRITISCH"
                
            print(f"{elapsed:>4.1f}s | {step:>4d} | {curr_vram_mb:>7.1f} MB   | {peak_vram_mb:>6.1f} MB  | {free_vram_mb:>6.1f} MB  | {sys_ram_gb:>5.1f} GB    | {step_time:>6.2f}s    | {status}")
            
        except torch.cuda.OutOfMemoryError as e:
            oom_occurred = True
            error_msg = str(e)
            print(f"❌ CUDA OUT OF MEMORY BEI STEP {step}!")
            break
        except Exception as e:
            oom_occurred = True
            error_msg = str(e)
            print(f"❌ FEHLER BEI STEP {step}: {e}")
            break
            
    total_elapsed = time.time() - start_time
    print("-" * 80)
    print("\n🏁 ERGEBNIS DES 30-SEKUNDEN TESTS:")
    print(f"  • Gesamtdauer:              {total_elapsed:.1f} Sekunden")
    print(f"  • Durchgeführte Steps:      {step} vollständige Forward+Backward Passes")
    print(f"  • Maximaler VRAM-Peak:      {max_peak_vram_mb:.1f} MB (von {total_vram_mb:.0f} MB)")
    print(f"  • Minimaler freier VRAM:    {min_free_vram_mb:.1f} MB")
    print(f"  • VRAM-Sicherheitsabstand:  {(min_free_vram_mb / total_vram_mb) * 100:.1f}% Puffer")
    print(f"  • OOM aufgetreten?:         {'❌ JA (ABSTURZ)' if oom_occurred else '✅ NEIN (100% STABIL)'}")
    
    if not oom_occurred:
        if min_free_vram_mb > 1000:
            verdict = "🟢 PERFEKT: Sehr großer Sicherheitsabstand (>1.000 MB frei). Absolut unkritisch für Dauerlauf!"
        elif min_free_vram_mb > 500:
            verdict = "🟢 GUT: Solider Puffer (>500 MB frei). Sicher für Dauerlauf."
        else:
            verdict = "🟡 KNAPP: Unter 500 MB Puffer. Könnte bei Fragmentierung riskant sein."
        print(f"\n👉 FAZIT: {verdict}")
    else:
        print(f"\n👉 FAZIT: 🔴 OOM aufgetreten! Details: {error_msg}")

if __name__ == "__main__":
    run_stress_test()
