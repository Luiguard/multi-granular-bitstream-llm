#!/usr/bin/env python3
"""OpenAI & Ollama-compatible Local API Server for Multi-Granular Bitstream Model."""

import argparse
import json
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
import torch.nn.functional as F

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.tokenizer import ViterbiTokenizer
from scripts.train_distributed import MultiGranularCausalTransformer


GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
GLOBAL_DEVICE = None
GLOBAL_VOCAB = None


class OllamaBridgeHandler(BaseHTTPRequestHandler):

    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        if self.path == "/api/tags" or self.path == "/v1/models":
            # Return Ollama / OpenAI model list
            response = {
                "models": [
                    {
                        "name": "multigranular-llm:latest",
                        "model": "multigranular-llm:latest",
                        "details": {
                            "family": "multi-granular-bitstream",
                            "parameter_size": "110M",
                            "quantization_level": "FP16",
                        },
                    }
                ]
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not Found"}).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        data = json.loads(body) if body else {}

        # Extract prompt
        if "prompt" in data:
            prompt = data["prompt"]
        elif "messages" in data and len(data["messages"]) > 0:
            prompt = data["messages"][-1].get("content", "")
        else:
            prompt = "Hallo"

        # Generate response using Bitstream model
        tokens = GLOBAL_TOKENIZER.encode(prompt)
        input_ids = list(tokens)

        max_tokens = int(data.get("max_tokens", 32))
        temperature = float(data.get("temperature", 0.7))

        with torch.no_grad():
            for _ in range(max_tokens):
                inp_t = torch.tensor([input_ids[-512:]], dtype=torch.long, device=GLOBAL_DEVICE)
                logits = GLOBAL_MODEL(inp_t)
                last_logits = logits[0, -1, :] / max(0.1, temperature)
                probs = F.softmax(last_logits, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())
                input_ids.append(next_token)

        generated_text = GLOBAL_TOKENIZER.decode(input_ids[len(tokens):])

        # Return Ollama / OpenAI response format
        response = {
            "model": "multigranular-llm:latest",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": generated_text,
            "done": True,
            "choices": [
                {
                    "message": {"role": "assistant", "content": generated_text},
                    "finish_reason": "stop",
                }
            ],
        }

        self._set_headers(200)
        self.wfile.write(json.dumps(response).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Ollama Bridge Server")
    parser.add_argument("--checkpoint", type=str, default="./multi_granular_model.pt", help="Pfad zum Modell")
    parser.add_argument("--vocab_file", type=str, default="./data/vocab_65k.json", help="Pfad zu vocab.json")
    parser.add_argument("--port", type=int, default=11434, help="Port (Standard Ollama Port: 11434)")
    args = parser.parse_args()

    global GLOBAL_MODEL, GLOBAL_TOKENIZER, GLOBAL_DEVICE, GLOBAL_VOCAB

    if not os.path.exists(args.vocab_file):
        args.vocab_file = "./vocab.json"

    GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"🦙 OLLAMA & OPENAI COMPATIBLE BRIDGE SERVER")
    print("=" * 80)
    print(f"  - Device:     {GLOBAL_DEVICE}")
    print(f"  - Vokabular:  {args.vocab_file}")

    GLOBAL_VOCAB = MultiGranularVocabulary.load_json(args.vocab_file)
    GLOBAL_TOKENIZER = ViterbiTokenizer(GLOBAL_VOCAB)

    GLOBAL_MODEL = MultiGranularCausalTransformer(
        vocab_size=GLOBAL_VOCAB.size,
        rank=64,
        d_model=512,
        n_layers=6,
        n_heads=8,
    ).to(GLOBAL_DEVICE)

    if os.path.exists(args.checkpoint):
        print(f"  - Lade Gewichte: {args.checkpoint}")
        GLOBAL_MODEL.load_state_dict(torch.load(args.checkpoint, map_location=GLOBAL_DEVICE, weights_only=True), strict=False)

    GLOBAL_MODEL.eval()

    server = HTTPServer(("0.0.0.0", args.port), OllamaBridgeHandler)
    print(f"\n🚀 Server läuft auf http://localhost:{args.port}")
    print("Kompatibel mit Ollama, Open-WebUI und standard OpenAI APIs.")
    print("Drücke Strg+C zum Beenden.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer beendet.")


if __name__ == "__main__":
    main()
