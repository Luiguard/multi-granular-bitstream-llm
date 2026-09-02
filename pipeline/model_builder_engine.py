#!/usr/bin/env python3
"""
Bitstream AI Model Builder Engine.
Computes bit-exact parameter counts, active compute sizes, VRAM/RAM footprints,
hardware feasibility checks, and generates custom PyTorch MoE definitions.
Supports architectures from 100M Edge (1 Expert) up to 144B Frontier MoE (128 Experts).
"""

import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, "/home/benjamin/Bilder")


class ModelArchitectureSpecs:
    """Calculates and holds detailed specs for any MoE configuration."""

    def __init__(
        self,
        name: str = "Custom-MoE",
        d_model: int = 2048,
        n_layers: int = 24,
        n_heads: int = 16,
        num_experts: int = 12,
        top_k: int = 2,
        ffn_multiplier: float = 2.6875,
        vocab_size: int = 262144,
        rank_embedding: int = 64,
        max_seq_len: int = 7168,
        expert_domains: Optional[Dict[str, List[int]]] = None,
        guardrails: Optional[Dict[str, float]] = None,
    ):
        self.name = name
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.ffn_multiplier = ffn_multiplier
        self.vocab_size = vocab_size
        self.rank_embedding = rank_embedding
        self.max_seq_len = max_seq_len
        self.expert_domains = expert_domains or {}
        self.guardrails = guardrails or {
            "rlvr_strictness": 0.85,
            "hallucination_guard": 0.90,
            "epistemic_curiosity": 0.75,
            "sandbox_enforcement": 1.0,
        }

        # Derived dimensions
        self.hidden_dim = int(d_model * ffn_multiplier)
        # Ensure hidden_dim is a multiple of 64
        self.hidden_dim = ((self.hidden_dim + 63) // 64) * 64

    def compute_parameters(self) -> Dict[str, Any]:
        """Calculates exact total and active parameter count."""
        # 1. Embedding parameters (Factorized rank-64)
        if self.rank_embedding > 0:
            vocab_emb_params = self.vocab_size * self.rank_embedding  # E_vocab
            proj_in_params = self.rank_embedding * self.d_model       # E_proj
            head_proj_params = self.d_model * self.rank_embedding     # head_proj
            head_out_params = self.rank_embedding * self.vocab_size   # head_out
            total_embedding_params = vocab_emb_params + proj_in_params + head_proj_params + head_out_params
        else:
            total_embedding_params = 2 * (self.vocab_size * self.d_model)

        # 2. Per-Layer Shared Parameters (Attention + LayerNorms + Router)
        # Q, K, V projections + Output projection
        qkv_proj = 3 * (self.d_model * self.d_model)
        out_proj = self.d_model * self.d_model
        attn_norms = 2 * self.d_model  # pre_norm + post_norm
        router_params = self.d_model * self.num_experts
        shared_layer_params = qkv_proj + out_proj + attn_norms + router_params

        # 3. Single SwiGLU Expert Parameters (3 linear projections: w_gate, w_up, w_down)
        single_expert_params = 3 * (self.d_model * self.hidden_dim)

        # 4. Total MoE Layer Parameters
        all_experts_layer_params = self.num_experts * single_expert_params
        single_layer_total_params = shared_layer_params + all_experts_layer_params
        single_layer_active_params = shared_layer_params + (self.top_k * single_expert_params)

        # 5. Final Model Totals
        final_norm_params = self.d_model
        total_model_params = total_embedding_params + (self.n_layers * single_layer_total_params) + final_norm_params
        active_model_params = total_embedding_params + (self.n_layers * single_layer_active_params) + final_norm_params

        return {
            "total_params": total_model_params,
            "total_params_billion": round(total_model_params / 1e9, 2),
            "active_params": active_model_params,
            "active_params_million": round(active_model_params / 1e6, 1),
            "active_params_billion": round(active_model_params / 1e9, 2),
            "sparsity_ratio": round((total_model_params - active_model_params) / max(1, total_model_params) * 100, 1),
            "single_expert_params_million": round(single_expert_params / 1e6, 2),
            "embedding_params_million": round(total_embedding_params / 1e6, 2),
        }

    def compute_hardware_footprint(self) -> Dict[str, Any]:
        """Calculates precise VRAM and RAM requirements for training and inference."""
        params_info = self.compute_parameters()
        total_params = params_info["total_params"]
        active_params = params_info["active_params"]

        # In pure bytes
        bytes_16bit_total = total_params * 2
        bytes_8bit_total = total_params * 1
        bytes_4bit_total = int(total_params * 0.55)

        # KV-Cache for max_seq_len (2 * n_layers * n_heads * (d_model/n_heads) * seq_len * 2 bytes)
        kv_cache_bytes = 2 * self.n_layers * self.d_model * self.max_seq_len * 2

        # Training VRAM with JIT Layer Offloading + GaLore:
        # GPU only holds 1 layer active parameters + factorized embeddings + KV/Autograd activation peak
        active_layer_params = (4 * self.d_model * self.d_model) + (self.top_k * 3 * self.d_model * self.hidden_dim)
        active_layer_vram_bytes = active_layer_params * 2
        galore_optimizer_bytes = (active_layer_params * 64 // self.d_model) * 8  # rank-64 SVD optimizer state
        autograd_activations_bytes = self.max_seq_len * self.d_model * 2 * 12  # chunked activations peak
        
        training_gpu_vram_gb = (active_layer_vram_bytes + galore_optimizer_bytes + autograd_activations_bytes + 67 * 1024 * 1024 + 400 * 1024 * 1024) / 1e9
        training_sys_ram_gb = (bytes_16bit_total + 4 * 1024 * 1024 * 1024) / 1e9  # Full weights in RAM + OS buffer

        # Inference VRAM / RAM
        inference_vram_4bit_full_gpu_gb = (bytes_4bit_total + kv_cache_bytes + 350 * 1024 * 1024) / 1e9
        inference_vram_hybrid_gb = (active_params * 2 + kv_cache_bytes + 350 * 1024 * 1024) / 1e9
        inference_ram_hybrid_4bit_gb = (bytes_4bit_total + 3 * 1024 * 1024 * 1024) / 1e9

        # Feasibility check on RTX 3060 (6GB) & 32GB System RAM
        fits_rtx3060_train = training_gpu_vram_gb <= 5.8 and training_sys_ram_gb <= 30.0
        fits_rtx3060_infer_full = inference_vram_4bit_full_gpu_gb <= 5.8
        fits_rtx3060_infer_hybrid = inference_vram_hybrid_gb <= 5.8 and inference_ram_hybrid_4bit_gb <= 30.0

        if fits_rtx3060_train:
            tier_badge = "🟢 Läuft & trainiert auf Laptop (RTX 3060 6GB + 32GB RAM)"
            tier_color = "#10b981"
        elif fits_rtx3060_infer_hybrid:
            tier_badge = "🟡 Inferenz auf Laptop (Hybrid), Training benötigt NVMe-Streaming"
            tier_color = "#f59e0b"
        elif bytes_4bit_total / 1e9 <= 22.0:
            tier_badge = "🟠 Benötigt Workstation / RTX 4090 (24GB VRAM)"
            tier_color = "#f97316"
        else:
            tier_badge = "🔴 Mega-Frontier Modell: Benötigt NVMe PCIe 4.0 Streaming oder Multi-GPU"
            tier_color = "#ef4444"

        return {
            "params": params_info,
            "weights_size_16bit_gb": round(bytes_16bit_total / 1e9, 2),
            "weights_size_8bit_gb": round(bytes_8bit_total / 1e9, 2),
            "weights_size_4bit_gb": round(bytes_4bit_total / 1e9, 2),
            "kv_cache_gb": round(kv_cache_bytes / 1e9, 2),
            "training_gpu_vram_gb": round(training_gpu_vram_gb, 2),
            "training_sys_ram_gb": round(training_sys_ram_gb, 1),
            "inference_vram_4bit_full_gpu_gb": round(inference_vram_4bit_full_gpu_gb, 2),
            "inference_vram_hybrid_gb": round(inference_vram_hybrid_gb, 2),
            "inference_ram_hybrid_4bit_gb": round(inference_ram_hybrid_4bit_gb, 1),
            "fits_rtx3060_train": fits_rtx3060_train,
            "fits_rtx3060_infer_full": fits_rtx3060_infer_full,
            "fits_rtx3060_infer_hybrid": fits_rtx3060_infer_hybrid,
            "tier_badge": tier_badge,
            "tier_color": tier_color,
        }

    def generate_pytorch_code(self) -> str:
        """Generates a complete drop-in PyTorch architecture definition."""
        class_name = "".join(c if c.isalnum() else "_" for c in self.name)
        return f'''# Auto-generated by Bitstream AI Architecture Studio
# Model: {self.name}
# Parameters: {self.compute_parameters()["total_params_billion"]}B Total | {self.compute_parameters()["active_params_million"]}M Active per token
# Experts: {self.num_experts} (Top-{self.top_k} routing)

import torch
import torch.nn as nn
from pipeline.moe_components import SparseMoELayer, Top2GatingRouter
from pipeline.yarn_rope import YaRNScaledRotaryEmbedding

class {class_name}(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = {self.d_model}
        self.n_layers = {self.n_layers}
        self.n_heads = {self.n_heads}
        self.num_experts = {self.num_experts}
        self.top_k = {self.top_k}
        self.hidden_dim = {self.hidden_dim}
        self.vocab_size = {self.vocab_size}
        self.rank_emb = {self.rank_embedding}

        # 18-Bit Factorized Embedding
        self.E_vocab = nn.Embedding(self.vocab_size, self.rank_emb)
        self.E_proj = nn.Linear(self.rank_emb, self.d_model, bias=False)

        # Transformer Layers
        self.moe_layers = nn.ModuleList([
            SparseMoELayer(self.d_model, self.hidden_dim, num_experts=self.num_experts)
            for _ in range(self.n_layers)
        ])

        # Head Output
        self.head_proj = nn.Linear(self.d_model, self.rank_emb, bias=False)
        self.head_out = nn.Linear(self.rank_emb, self.vocab_size, bias=False)

    def forward(self, input_ids, domain_cluster=None):
        x = self.E_proj(self.E_vocab(input_ids))
        total_aux_loss = 0.0
        for layer in self.moe_layers:
            x, aux_loss = layer(x, domain_cluster=domain_cluster)
            total_aux_loss += aux_loss
        logits = self.head_out(self.head_proj(x))
        return logits, total_aux_loss
'''

    def generate_training_script(self) -> str:
        """Generates a standalone, plug-and-play training script for this custom architecture."""
        class_name = "".join(c if c.isalnum() else "_" for c in self.name)
        return f'''#!/usr/bin/env python3
"""
Auto-generated Standalone Training Script for {self.name}.
Architecture: {self.compute_parameters()["total_params_billion"]}B Total Parameters | {self.num_experts} Experts (Top-{self.top_k})
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import {class_name}

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
LEARNING_RATE = 3e-4
BATCH_SIZE = 2
SEQ_LEN = {self.max_seq_len}
MAX_STEPS = 100000
SAVE_INTERVAL = 50

def train():
    print(f"🚀 Initialisiere Training für {self.name} auf {{DEVICE}}...")
    model = {class_name}().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  • Modell-Parameter: {{total_params / 1e9:.2f}} Mrd.")
    print(f"  • Hardware: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}}")

    model.train()
    step = 0
    start_time = time.time()

    # Synthetic / Stream batch generator
    for step in range(1, MAX_STEPS + 1):
        input_ids = torch.randint(0, {self.vocab_size}, (BATCH_SIZE, min(SEQ_LEN, 512)), device=DEVICE)
        targets = torch.randint(0, {self.vocab_size}, (BATCH_SIZE, min(SEQ_LEN, 512)), device=DEVICE)

        optimizer.zero_grad()
        logits, aux_loss = model(input_ids)
        
        loss = F.cross_entropy(logits.view(-1, {self.vocab_size}), targets.view(-1)) + (0.01 * aux_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 10 == 0:
            elapsed = time.time() - start_time
            tps = (10 * BATCH_SIZE * min(SEQ_LEN, 512)) / max(1e-5, elapsed)
            print(f"  [Step {{step:05d}}] Loss: {{loss.item():.4f}} | TPS: {{tps:.1f}}")
            start_time = time.time()

        if step % SAVE_INTERVAL == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/checkpoint_step_{{step}}.pt"
            torch.save({{"step": step, "model_state_dict": model.state_dict()}}, ckpt_path)
            print(f"  💾 Checkpoint gespeichert -> {{ckpt_path}}")

if __name__ == "__main__":
    train()
'''

    def generate_api_server_script(self) -> str:
        """Generates a standalone OpenAI-compatible REST API Server (/v1/chat/completions)."""
        class_name = "".join(c if c.isalnum() else "_" for c in self.name)
        return f'''#!/usr/bin/env python3
"""
OpenAI-Compatible REST API Server for {self.name}.
Architecture: {self.compute_parameters()["total_params_billion"]}B Total | {self.num_experts} Experts (Top-{self.top_k})
Run with: python api_server.py
"""

import json
import time
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from model import {class_name}
from vocabulary import MultiGranularVocabulary
from tokenizer import ViterbiTokenizer

PORT = int(os.environ.get("PORT", 8000))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

print(f"🚀 Initialisiere {self.name} auf {{DEVICE}}...")
MODEL = {class_name}().to(DEVICE)
MODEL.eval()

VOCAB_FILE = "vocab.bin" if os.path.exists("vocab.bin") else "vocab.json"
print(f"📖 Lade Vokabular aus {{VOCAB_FILE}}...")
VOCAB = MultiGranularVocabulary.load_file(VOCAB_FILE)
TOKENIZER = ViterbiTokenizer(VOCAB)

class OpenAIAPIHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = {{
                "object": "list",
                "data": [{{
                    "id": "{self.name}",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "bitstream-ai",
                    "permission": []
                }}]
            }}
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {{}}
            messages = body.get("messages", [])
            max_tokens = int(body.get("max_tokens", 100))
            temperature = float(body.get("temperature", 0.7))
            
            prompt_text = "\\n".join([f"{{m.get('role', 'user')}}: {{m.get('content', '')}}" for m in messages]) + "\\nassistant:"
            encoded_tokens = TOKENIZER.encode(prompt_text)
            input_ids = list(encoded_tokens) if encoded_tokens else [1]

            generated_ids = []
            with torch.no_grad():
                for _ in range(max_tokens):
                    inp_t = torch.tensor([input_ids[-128:]], dtype=torch.long, device=DEVICE)
                    logits, _ = MODEL(inp_t)
                    next_id = int(torch.argmax(logits[0, -1, :]).item())
                    input_ids.append(next_id)
                    generated_ids.append(next_id)
                    if next_id == 0:
                        break

            output_text = TOKENIZER.decode(generated_ids)
            res = {{
                "id": f"chatcmpl-{{int(time.time())}}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "{self.name}",
                "choices": [{{
                    "index": 0,
                    "message": {{"role": "assistant", "content": output_text.strip()}},
                    "finish_reason": "stop"
                }}],
                "usage": {{
                    "prompt_tokens": len(encoded_tokens),
                    "completion_tokens": len(generated_ids),
                    "total_tokens": len(encoded_tokens) + len(generated_ids)
                }}
            }}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), OpenAIAPIHandler)
    print(f"✨ OpenAI API Server läuft auf http://0.0.0.0:{{PORT}}/v1")
    server.serve_forever()
'''

    def generate_web_chat_script(self) -> str:
        """Generates a self-contained Web Chat application with embedded dark glassmorphism UI."""
        class_name = "".join(c if c.isalnum() else "_" for c in self.name)
        return f'''#!/usr/bin/env python3
"""
Single-File Web Chat Studio for {self.name}.
Run with: python web_chat.py and open http://localhost:8080 in your browser!
"""

import json
import time
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from model import {class_name}
from vocabulary import MultiGranularVocabulary
from tokenizer import ViterbiTokenizer

PORT = int(os.environ.get("PORT", 8080))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

print(f"🚀 Lade {self.name} Web-Chat auf {{DEVICE}}...")
MODEL = {class_name}().to(DEVICE)
MODEL.eval()

VOCAB_FILE = "vocab.bin" if os.path.exists("vocab.bin") else "vocab.json"
print(f"📖 Lade Vokabular aus {{VOCAB_FILE}}...")
VOCAB = MultiGranularVocabulary.load_file(VOCAB_FILE)
TOKENIZER = ViterbiTokenizer(VOCAB)

HTML_PAGE = """<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <title>{self.name} · Web Chat</title>
  <style>
    body {{ background: #0b0f19; color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; }}
    header {{ background: rgba(15, 23, 42, 0.85); border-bottom: 1px solid rgba(51, 65, 85, 0.5); padding: 14px 24px; display: flex; justify-content: space-between; align-items: center; backdrop-filter: blur(12px); }}
    h1 {{ font-size: 1.1rem; margin: 0; color: #38bdf8; }}
    #chat-container {{ flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 14px; max-width: 860px; margin: 0 auto; width: 100%; }}
    .msg {{ padding: 12px 18px; border-radius: 12px; max-width: 80%; line-height: 1.5; font-size: 0.95rem; }}
    .user {{ background: #1e293b; align-self: flex-end; border-bottom-right-radius: 2px; }}
    .bot {{ background: #0f172a; border: 1px solid #334155; align-self: flex-start; border-bottom-left-radius: 2px; }}
    #input-form {{ padding: 16px 24px; background: rgba(15, 23, 42, 0.95); border-top: 1px solid #334155; display: flex; gap: 12px; max-width: 860px; margin: 0 auto; width: 100%; box-sizing: border-box; }}
    #msg-input {{ flex: 1; background: #020617; border: 1px solid #334155; color: #fff; padding: 12px 16px; border-radius: 8px; font-size: 0.95rem; outline: none; }}
    #msg-input:focus {{ border-color: #38bdf8; }}
    button {{ background: #38bdf8; color: #020617; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 700; cursor: pointer; }}
  </style>
</head>
<body>
  <header>
    <h1>🌌 {self.name}</h1>
    <span style="font-size: 0.8rem; background: #1e293b; padding: 4px 10px; border-radius: 6px;">{self.num_experts} Experten · Vokabular: {self.vocab_size}</span>
  </header>
  <div id="chat-container">
    <div class="msg bot">Hallo! Ich bin dein maßgeschneidertes <strong>{self.name}</strong> Modell. Wie kann ich dir helfen?</div>
  </div>
  <form id="input-form">
    <input type="text" id="msg-input" placeholder="Schreibe eine Nachricht..." autofocus autocomplete="off">
    <button type="submit">Senden</button>
  </form>
  <script>
    const form = document.getElementById('input-form');
    const input = document.getElementById('msg-input');
    const container = document.getElementById('chat-container');

    form.addEventListener('submit', async (e) => {{
      e.preventDefault();
      const txt = input.value.trim();
      if (!txt) return;
      input.value = '';

      container.innerHTML += `<div class="msg user">${{txt}}</div>`;
      container.scrollTop = container.scrollHeight;

      const botMsg = document.createElement('div');
      botMsg.className = 'msg bot';
      botMsg.textContent = '...';
      container.appendChild(botMsg);

      const res = await fetch('/api/chat', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ prompt: txt }})
      }});
      const data = await res.json();
      botMsg.textContent = data.reply;
      container.scrollTop = container.scrollHeight;
    }});
  </script>
</body>
</html>
"""

class WebChatHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path == "/api/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {{}}
            prompt = body.get("prompt", "")
            
            encoded_tokens = TOKENIZER.encode(prompt)
            input_ids = list(encoded_tokens) if encoded_tokens else [1]
            generated_ids = []
            with torch.no_grad():
                for _ in range(50):
                    inp_t = torch.tensor([input_ids[-128:]], dtype=torch.long, device=DEVICE)
                    logits, _ = MODEL(inp_t)
                    next_id = int(torch.argmax(logits[0, -1, :]).item())
                    input_ids.append(next_id)
                    generated_ids.append(next_id)
                    if next_id == 0:
                        break

            output_text = TOKENIZER.decode(generated_ids)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({{"reply": output_text.strip() or "Antwort generiert."}}).encode("utf-8"))

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), WebChatHandler)
    print(f"✨ Web Chat Studio läuft auf http://localhost:{{PORT}}")
    server.serve_forever()
'''

    def generate_cli_chat_script(self) -> str:
        """Generates an interactive terminal chat script with token streaming."""
        class_name = "".join(c if c.isalnum() else "_" for c in self.name)
        return f'''#!/usr/bin/env python3
"""
Interactive Terminal CLI Chat for {self.name}.
Run with: python cli_chat.py
"""

import sys
import os
import time
import torch
from model import {class_name}
from vocabulary import MultiGranularVocabulary
from tokenizer import ViterbiTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

print("=" * 65)
print(f"🌌 {self.name} · Terminal CLI Chat")
print(f"Hardware: {{DEVICE}} | Parameter: {self.compute_parameters()['total_params_billion']}B Total | Vokabular: {self.vocab_size}")
print("=" * 65)

VOCAB_FILE = "vocab.bin" if os.path.exists("vocab.bin") else "vocab.json"
print(f"📖 Lade Vokabular aus {{VOCAB_FILE}}...")
VOCAB = MultiGranularVocabulary.load_file(VOCAB_FILE)
TOKENIZER = ViterbiTokenizer(VOCAB)
print("Tippe 'exit' oder 'quit' zum Beenden.\\n")

model = {class_name}().to(DEVICE)
model.eval()

while True:
    try:
        user_input = input("\\033[1;36mDu:\\033[0m ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Auf Wiedersehen!")
            break

        print("\\033[1;35m{self.name}:\\033[0m ", end="", flush=True)
        encoded_tokens = TOKENIZER.encode(user_input)
        input_ids = list(encoded_tokens) if encoded_tokens else [1]

        with torch.no_grad():
            for _ in range(60):
                inp_t = torch.tensor([input_ids[-128:]], dtype=torch.long, device=DEVICE)
                logits, _ = model(inp_t)
                next_id = int(torch.argmax(logits[0, -1, :]).item())
                input_ids.append(next_id)
                token_str = TOKENIZER.decode([next_id])
                print(token_str, end="", flush=True)
                time.sleep(0.015)
                if next_id == 0:
                    break
        print("\\n")
    except (KeyboardInterrupt, EOFError):
        print("\\nBeendet.")
        break
'''

    def create_training_bundle_zip(self, output_zip_path: str) -> str:
        """Packages model.py, train.py, api_server.py, web_chat.py, cli_chat.py, config.json, and launchers into a zip file."""
        import zipfile

        class_name = "".join(c if c.isalnum() else "_" for c in self.name)
        model_code = self.generate_pytorch_code()
        train_code = self.generate_training_script()
        api_code = self.generate_api_server_script()
        web_chat_code = self.generate_web_chat_script()
        cli_chat_code = self.generate_cli_chat_script()

        config_data = json.dumps({
            "name": self.name,
            "specs": self.__dict__,
            "footprint": self.compute_hardware_footprint(),
        }, indent=2)

        start_linux = f'''#!/usr/bin/env bash
set -e
echo "=========================================================="
echo "🚀 {self.name} · All-in-One Linux Starter"
echo "=========================================================="
if [ ! -d ".venv" ]; then
    echo "📦 Erstelle virtuelles Python Environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt

echo "\\nWelcher Modus soll gestartet werden?"
echo "1) 💬 Web Chat Studio (Browser auf http://localhost:8080)"
echo "2) 🔌 OpenAI-kompatibler REST-API Server (Port 8000)"
echo "3) ⌨️ Terminal CLI Chat"
echo "4) 🚂 Training & Fine-Tuning starten"
read -p "Auswahl (1-4) [Standard: 1]: " choice
choice=${{choice:-1}}

if [ "$choice" = "1" ]; then
    python web_chat.py
elif [ "$choice" = "2" ]; then
    python api_server.py
elif [ "$choice" = "3" ]; then
    python cli_chat.py
elif [ "$choice" = "4" ]; then
    python train.py
fi
'''

        start_windows = f'''@echo off
echo ==========================================================
echo 🚀 {self.name} · All-in-One Windows Starter
echo ==========================================================
if not exist ".venv" (
    echo 📦 Erstelle virtuelles Python Environment...
    python -m venv .venv
)
call .venv\\Scripts\\activate
pip install -r requirements.txt

echo.
echo Welcher Modus soll gestartet werden?
echo 1) 💬 Web Chat Studio (Browser auf http://localhost:8080)
echo 2) 🔌 OpenAI-kompatibler REST-API Server (Port 8000)
echo 3) ⌨️ Terminal CLI Chat
echo 4) 🚂 Training starten
set /p choice="Auswahl (1-4) [Standard: 1]: "
if "%choice%"=="" set choice=1

if "%choice%"=="1" python web_chat.py
if "%choice%"=="2" python api_server.py
if "%choice%"=="3" python cli_chat.py
if "%choice%"=="4" python train.py
pause
'''

        start_mac = f'''#!/usr/bin/env bash
set -e
echo "=========================================================="
echo "🍎 {self.name} · Apple Silicon (MPS / Metal) Starter"
echo "=========================================================="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt
python web_chat.py
'''

        dockerfile = f'''FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8080
CMD ["python", "api_server.py"]
'''

        docker_compose = f'''version: '3.8'
services:
  {class_name.lower()}:
    build: .
    ports:
      - "8000:8000"
      - "8080:8080"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
'''

        reqs_txt = "torch>=2.0.0\nnumpy>=1.24.0\n"
        readme_md = f'''# {self.name} · Universelles Standalone KI-Paket

Automatisch generiert vom **Bitstream AI Architecture Studio**.

## 📊 Spezifikationen
- **Gesamt-Parameter**: {self.compute_parameters()["total_params_billion"]} Mrd.
- **Aktive Parameter**: {self.compute_parameters()["active_params_million"]} Mio. pro Token
- **Experten-Topologie**: {self.num_experts} Experten (Top-{self.top_k} Routing)
- **Vokabular**: {self.vocab_size} (18-Bit Viterbi)
- **Kontext-Länge**: {self.max_seq_len} Tokens

## 🚀 Schnellstart

### 🐧 Linux / WSL2:
```bash
chmod +x start_linux.sh
./start_linux.sh
```

### 🪟 Windows (10 / 11):
Doppelklick auf `start_windows.bat` oder in PowerShell ausführen:
```cmd
start_windows.bat
```

### 🍎 Mac (Apple Silicon M1/M2/M3/M4):
```bash
chmod +x start_mac.sh
./start_mac.sh
```

### 🐳 Docker:
```bash
docker compose up --build
```

## 🔌 OpenAI-kompatible API
Wenn `api_server.py` läuft (Port 8000), kannst du jedes Tool verbinden:
```bash
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{"messages": [{{"role": "user", "content": "Hallo KI!"}}]}}'
```
'''

        os.makedirs(os.path.dirname(os.path.abspath(output_zip_path)), exist_ok=True)
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.py", model_code)
            zf.writestr("train.py", train_code)
            zf.writestr("api_server.py", api_code)
            zf.writestr("web_chat.py", web_chat_code)
            zf.writestr("cli_chat.py", cli_chat_code)
            zf.writestr("config.json", config_data)
            zf.writestr("start_linux.sh", start_linux)
            zf.writestr("start_windows.bat", start_windows)
            zf.writestr("start_mac.sh", start_mac)
            zf.writestr("Dockerfile", dockerfile)
            zf.writestr("docker-compose.yml", docker_compose)
            zf.writestr("requirements.txt", reqs_txt)
            zf.writestr("README.md", readme_md)

            # Standalone Tokenizer & Vocabulary Module
            tokenizer_src = "/home/benjamin/Bilder/pipeline/tokenizer.py"
            vocab_src = "/home/benjamin/Bilder/pipeline/vocabulary.py"
            if os.path.exists(tokenizer_src):
                zf.write(tokenizer_src, arcname="tokenizer.py")
            if os.path.exists(vocab_src):
                zf.write(vocab_src, arcname="vocabulary.py")

            # Echte Vokabular-Dateien passend zur Vokabular-Breite (16-Bit, 18-Bit oder 20-Bit)
            if self.vocab_size >= 1000000:
                bin_1m = "/home/benjamin/Bilder/data/vocab_1m_20bit.bin"
                meta_1m = "/home/benjamin/Bilder/data/vocab_1m_metadata.json"
                if os.path.exists(bin_1m):
                    zf.write(bin_1m, arcname="vocab.bin")
                if os.path.exists(meta_1m):
                    zf.write(meta_1m, arcname="vocab_metadata.json")
            elif self.vocab_size == 262144:
                json_262k = "/home/benjamin/Bilder/data/vocab_262k.json"
                if os.path.exists(json_262k):
                    zf.write(json_262k, arcname="vocab.json")
            elif self.vocab_size == 65536:
                json_65k = "/home/benjamin/Bilder/data/vocab_65k.json"
                if os.path.exists(json_65k):
                    zf.write(json_65k, arcname="vocab.json")
            else:
                bin_1m = "/home/benjamin/Bilder/data/vocab_1m_20bit.bin"
                if os.path.exists(bin_1m):
                    zf.write(bin_1m, arcname="vocab.bin")

        print(f"  📦 Universelles All-in-One Paket geschnürt -> {output_zip_path}")
        return output_zip_path




# Presets Library (1 Exp 100M bis 1 Exp 144B & Multi-MoE)
MODEL_PRESETS = {
    "moe_golden_20bit_7b": {
        "name": "Bitstream-20Bit-MoE-7.45B",
        "d_model": 2048,
        "n_layers": 24,
        "n_heads": 16,
        "num_experts": 12,
        "top_k": 2,
        "ffn_multiplier": 2.0,
        "vocab_size": 1048576,
        "rank_embedding": 64,
        "max_seq_len": 7168,
        "description": "🌟 20-Bit Golden Master (1.048.576 Tokens · 175+ Sprachen) mit 12 Experten auf 6 GB VRAM.",
    },
    "exp1_100m": {
        "name": "Bitstream-1Exp-100M",
        "d_model": 512,
        "n_layers": 8,
        "n_heads": 8,
        "num_experts": 1,
        "top_k": 1,
        "ffn_multiplier": 2.0,
        "vocab_size": 262144,
        "rank_embedding": 32,
        "max_seq_len": 4096,
        "description": "1 Experte · 100M Single-Core für Handys, Raspberry Pi 5 und IoT.",
    },
    "exp1_300m": {
        "name": "Bitstream-1Exp-300M",
        "d_model": 768,
        "n_layers": 12,
        "n_heads": 12,
        "num_experts": 1,
        "top_k": 1,
        "ffn_multiplier": 2.5,
        "vocab_size": 262144,
        "rank_embedding": 64,
        "max_seq_len": 4096,
        "description": "1 Experte · 300M Single-Core für sparsame Edge-Geräte.",
    },
    "exp1_1b": {
        "name": "Bitstream-1Exp-1B",
        "d_model": 1024,
        "n_layers": 16,
        "n_heads": 16,
        "num_experts": 1,
        "top_k": 1,
        "ffn_multiplier": 2.5,
        "vocab_size": 262144,
        "rank_embedding": 64,
        "max_seq_len": 4096,
        "description": "1 Experte · 1B Single-Core für mobile Apps und lokale Assistenten.",
    },
    "exp1_3b": {
        "name": "Bitstream-1Exp-3B",
        "d_model": 1536,
        "n_layers": 20,
        "n_heads": 16,
        "num_experts": 1,
        "top_k": 1,
        "ffn_multiplier": 2.5,
        "vocab_size": 262144,
        "rank_embedding": 64,
        "max_seq_len": 7168,
        "description": "1 Experte · 3B Single-Core für Laptops und schnelle On-Device Inferenz.",
    },
    "exp1_7b": {
        "name": "Bitstream-1Exp-7.45B",
        "d_model": 2048,
        "n_layers": 24,
        "n_heads": 16,
        "num_experts": 1,
        "top_k": 1,
        "ffn_multiplier": 2.0,
        "vocab_size": 262144,
        "rank_embedding": 64,
        "max_seq_len": 7168,
        "description": "1 Experte · 7.45B Dense: Volle Rechenleistung für komplexe Aufgaben.",
    },
    "exp1_14b": {
        "name": "Bitstream-1Exp-14B",
        "d_model": 3072,
        "n_layers": 28,
        "n_heads": 24,
        "num_experts": 1,
        "top_k": 1,
        "ffn_multiplier": 2.5,
        "vocab_size": 262144,
        "rank_embedding": 128,
        "max_seq_len": 8192,
        "description": "1 Experte · 14B Dense für Workstations und Coding-Assistenten.",
    },
    "exp1_30b": {
        "name": "Bitstream-1Exp-30B",
        "d_model": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "num_experts": 1,
        "top_k": 1,
        "ffn_multiplier": 2.5,
        "vocab_size": 262144,
        "rank_embedding": 128,
        "max_seq_len": 16384,
        "description": "1 Experte · 30B Dense für anspruchsvolle Forschung und Server.",
    },
    "exp1_70b": {
        "name": "Bitstream-1Exp-70B",
        "d_model": 8192,
        "n_layers": 64,
        "n_heads": 64,
        "num_experts": 1,
        "top_k": 1,
        "ffn_multiplier": 3.0,
        "vocab_size": 262144,
        "rank_embedding": 256,
        "max_seq_len": 32768,
        "description": "1 Experte · 70B Dense Frontier Modell für Enterprise GPU-Cluster.",
    },
    "exp1_144b": {
        "name": "Bitstream-1Exp-144B",
        "d_model": 12288,
        "n_layers": 80,
        "n_heads": 96,
        "num_experts": 1,
        "top_k": 1,
        "ffn_multiplier": 3.5,
        "vocab_size": 262144,
        "rank_embedding": 256,
        "max_seq_len": 32768,
        "description": "1 Experte · 144B Monolithischer Titan: Maximaler Single-Core für Supercomputer.",
    },
    "moe_laptop_7b": {
        "name": "Bitstream-MoE-12x-7.45B",
        "d_model": 2048,
        "n_layers": 24,
        "n_heads": 16,
        "num_experts": 12,
        "top_k": 2,
        "ffn_multiplier": 2.0,
        "vocab_size": 262144,
        "rank_embedding": 64,
        "max_seq_len": 7168,
        "description": "Gesplittet auf 12 Experten (Top-2 Routing ~500M aktiv) für 6 GB Laptop VRAM.",
    },
    "moe_titan_144b": {
        "name": "Bitstream-MoE-128x-144B",
        "d_model": 8192,
        "n_layers": 48,
        "n_heads": 64,
        "num_experts": 128,
        "top_k": 8,
        "ffn_multiplier": 3.5,
        "vocab_size": 262144,
        "rank_embedding": 256,
        "max_seq_len": 32768,
        "description": "Gesplittet auf 128 Experten: Maximale Wissensspeicherung aller Weltbibliotheken.",
    },
    "moe_galaxy_1_7t": {
        "name": "Bitstream-Galaxy-12x144B (1.73T)",
        "d_model": 12288,
        "n_layers": 80,
        "n_heads": 96,
        "num_experts": 12,
        "top_k": 2,
        "ffn_multiplier": 3.5,
        "vocab_size": 262144,
        "rank_embedding": 256,
        "max_seq_len": 32768,
        "description": "12x 144B fusioniert: 1.73 Billionen Parameter Super-Intelligenz.",
    },
}
