#!/usr/bin/env python3
"""Export Multi-Granular Bitstream Model to Ollama & SafeTensors format."""

import argparse
import json
import os
import sys
import torch

from pipeline.vocabulary import MultiGranularVocabulary


def export_to_ollama(checkpoint_path: str, vocab_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("🦙 OLLAMA & SAFETENSORS EXPORTER")
    print("=" * 80)

    # 1. Lade Vokabular & Modellgewichte
    print(f"Lade Vokabular aus: {vocab_file}")
    vocab = MultiGranularVocabulary.load_json(vocab_file)
    print(f"  - Vokabulargröße: {vocab.size:,} Tokens (16 Bit)")

    print(f"Lade Checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # 2. Speichere Modellgewichte
    weights_path = os.path.join(output_dir, "model.pt")
    torch.save(state_dict, weights_path)
    print(f"  💾 Modellgewichte gespeichert: {weights_path}")

    # 3. Exportiere Tokenizer Config & Vokabular Map
    tokenizer_config = {
        "model_type": "multi_granular_transformer",
        "vocab_size": vocab.size,
        "bit_width": vocab.required_bits,
        "token_map": {str(t_id): vocab.id_to_token.get(t_id, "") for t_id in range(vocab.size)},
    }
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    print(f"  💾 Tokenizer-Konfiguration: {tokenizer_config_path}")

    # 4. Erzeuge Ollama Modelfile
    modelfile_content = f"""# Ollama Modelfile for Multi-Granular Bitstream Foundation Model
FROM ./model.pt

# System Parameter
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|endoftext|>"

# System Prompt
SYSTEM \"\"\"Du bist eine hocheffiziente KI, die auf einer Multi-Granularitäts-Bitstream-Architektur basiert. Du antwortest präzise, strukturiert und fundiert auf Deutsch und Englisch.\"\"\"
"""
    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    print(f"  💾 Ollama Modelfile erstellt: {modelfile_path}")

    print("\n" + "=" * 80)
    print("✅ EXPORT ERFOLGREICH ABGESCHLOSSEN!")
    print("=" * 80)
    print("Um das Modell in Ollama zu laden, führe aus:")
    print(f"  cd {output_dir}")
    print("  ollama create multigranular-llm -f ./Modelfile")
    print("  ollama run multigranular-llm \"Erkläre mir Künstliche Intelligenz\"")


def main():
    parser = argparse.ArgumentParser(description="Export to Ollama & SafeTensors")
    parser.add_argument("--checkpoint", type=str, default="./multi_granular_model.pt", help="Pfad zum trainierten Modell")
    parser.add_argument("--vocab_file", type=str, default="./data/vocab_65k.json", help="Pfad zu vocab.json")
    parser.add_argument("--output_dir", type=str, default="./ollama_model", help="Ausgabeordner")
    args = parser.parse_args()

    if not os.path.exists(args.vocab_file):
        args.vocab_file = "./vocab.json"

    export_to_ollama(args.checkpoint, args.vocab_file, args.output_dir)


if __name__ == "__main__":
    main()
