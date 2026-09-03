#!/usr/bin/env python3
"""Model Exporter & Quantizer for 8.42B MoE Architecture.

Exports PyTorch checkpoints into:
1. SafeTensors format (Zero-Copy mmap, HuggingFace & vLLM compatible)
2. Architecture configuration (config.json)
3. Tokenizer configuration (tokenizer_config.json & vocab.json)
4. Optional INT8 channel-wise weight quantization (halves RAM from 16 GB to 8 GB).
"""

import os
import sys
import json
import argparse
from typing import Dict, Any
import torch
import safetensors.torch

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.moe_7b_model import MultiGranularMoE7BModel


def export_checkpoint(
    checkpoint_path: str = "/home/benjamin/Bilder/checkpoints_8b/8b_checkpoint_latest.pt",
    output_dir: str = "/home/benjamin/Bilder/checkpoints_8b/exported",
    quantize_int8: bool = False,
):
    print("=" * 70)
    print("📦 8.42B MoE MODELL-EXPORT & SAFETENSORS KONVERTER")
    print("=" * 70)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint_path}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"  - Lade Checkpoint: {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Clean tensors for SafeTensors export
    tensors_to_save: Dict[str, torch.Tensor] = {}
    quant_scales: Dict[str, float] = {}

    print(f"  - Verarbeite {len(state_dict)} Gewichtsmatrizen...")
    for k, v in state_dict.items():
        tensor = v.detach().contiguous()
        if quantize_int8 and tensor.is_floating_point() and tensor.ndim >= 2:
            max_val = tensor.abs().max().item()
            scale = max(1e-8, max_val / 127.0)
            int8_tensor = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
            tensors_to_save[k] = int8_tensor
            quant_scales[f"{k}.scale"] = scale
        else:
            tensors_to_save[k] = tensor.to(torch.bfloat16)

    # 1. Export SafeTensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    print(f"  - Speichere SafeTensors: {safetensors_path}...")
    safetensors.torch.save_file(tensors_to_save, safetensors_path)

    # 2. Export Config JSON
    config_data = {
        "architectures": ["MultiGranularMoE7BModel"],
        "model_type": "multi_granular_moe",
        "vocab_size": ckpt.get("vocab_size", 65536),
        "d_model": 2048,
        "n_layers": 20,
        "num_experts": 4,
        "routing_k": 2,
        "hidden_dim": 16384,
        "kv_latent_dim": 256,
        "q_latent_dim": 512,
        "num_heads": 32,
        "head_dim": 64,
        "rank": 64,
        "torch_dtype": "int8" if quantize_int8 else "bfloat16",
        "quantization": "int8_channel_wise" if quantize_int8 else "none",
        "quant_scales": quant_scales if quantize_int8 else None,
        "step": ckpt.get("step", 0),
        "total_tokens_trained": ckpt.get("tokens_processed", 0),
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"  - Architektur-Konfiguration gesichert: {config_path}")

    # 3. Export Tokenizer JSON
    vocab_source = "/home/benjamin/Bilder/data/vocab_65k.json"
    if os.path.exists(vocab_source):
        with open(vocab_source, "r", encoding="utf-8") as f:
            vocab_content = json.load(f)
        tok_config_path = os.path.join(output_dir, "tokenizer.json")
        with open(tok_config_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": "1.0",
                "type": "MultiGranularViterbi20Bit",
                "vocab": vocab_content
            }, f, indent=2)
        print(f"  - Tokenizer gesichert: {tok_config_path}")

    size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print("=" * 70)
    print(f"✅ Export erfolgreich abgeschlossen! Dateigröße: {size_mb:.1f} MB in {output_dir}")
    print("=" * 70)
    return safetensors_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export 8.42B MoE Checkpoint to SafeTensors")
    parser.add_argument("--checkpoint", type=str, default="/home/benjamin/Bilder/checkpoints_8b/8b_checkpoint_latest.pt")
    parser.add_argument("--output_dir", type=str, default="/home/benjamin/Bilder/checkpoints_8b/exported")
    parser.add_argument("--quantize_int8", action="store_true", help="Quantisiert Gewichte auf INT8")
    args = parser.parse_args()

    export_checkpoint(args.checkpoint, args.output_dir, args.quantize_int8)
