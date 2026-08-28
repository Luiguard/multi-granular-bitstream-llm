#!/usr/bin/env python3
"""Ollama & GGUF Export Tool for Multi-Granular Bitstream Models with Constitutional Guardrails."""

import argparse
import os
import sys

from pipeline.alignment_guardrails import SafetyGuardrails


def generate_ollama_modelfile(model_path: str = "./multi_granular_instruct_model.pt", output_modelfile: str = "./Modelfile"):
    system_prompt = SafetyGuardrails.format_system_prompt()

    content = f"""# ==============================================================================
# OLLAMA MODELFILE FOR MULTI-GRANULAR BITSTREAM LLM
# ==============================================================================
FROM {model_path}

# Template für Frage-Antwort-Interaktionen
TEMPLATE \"\"\"### System:
{{{{ .System }}}}

### Benutzer:
{{{{ .Prompt }}}}

### Assistent:
{{{{ .Response }}}}\"\"\"

# Konstitutioneller System-Prompt (Echte Daten, Epistemische Ehrlichkeit & Sicherheit)
SYSTEM \"\"\"{system_prompt}\"\"\"

# Inferenz-Parameter
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 32768
PARAMETER stop "### Benutzer:"
PARAMETER stop "### Assistent:"
PARAMETER stop "### System:"
"""

    with open(output_modelfile, "w", encoding="utf-8") as f:
        f.write(content)

    print("=" * 80)
    print(f"📦 OLLAMA MODELFILE ERFOLGREICH GENERIERT: {output_modelfile}")
    print("=" * 80)
    print("Starte dein Modell mit:")
    print("  ollama create bitstream-llm -f ./Modelfile")
    print("  ollama run bitstream-llm")
    print("=" * 80)


if __name__ == "__main__":
    generate_ollama_modelfile()
