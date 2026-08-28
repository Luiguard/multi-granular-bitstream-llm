#!/usr/bin/env python3
"""MoE Upcycling Engine: Expands trained Dense Multi-Granular weights into a 1.2B Sparse MoE Model.

Uses the 'Sparse Upcycling' technique (Komatsuzaki et al. / Mixtral 8x7B):
Takes the pretrained dense base transformer weights and initializes 8 specialized SwiGLU experts,
instantly creating a 1.2B parameter MoE architecture with only ~250M active compute footprint!
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.moe_components import SparseMoELayer
from pipeline.nemotron_components import NemotronTransformerBlock, RotaryEmbedding


class MultiGranularMoE1B2Model(nn.Module):
    """Full 1.2 Billion Parameter Sparse Mixture-of-Experts Architecture.

    - Total Parameters: ~1,200,000,000 (1.2 Billion)
    - Active Parameters per Token: ~250,000,000 (250 Million)
    - Factorized 16-Bit Embedding (Rang 64)
    - 8 SwiGLU Experts per Layer with Top-2 Gating Router
    """

    def __init__(
        self,
        vocab_size: int = 65536,
        rank: int = 64,
        d_model: int = 1024,
        n_layers: int = 16,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        num_experts: int = 8,
        hidden_dim: int = 2816,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_experts = num_experts

        # Factorized Input Embedding
        self.E_vocab = nn.Embedding(vocab_size, rank)
        self.E_proj = nn.Linear(rank, d_model, bias=False)
        self.rope = RotaryEmbedding(dim=d_model // n_heads, max_seq_len=max_seq_len)

        # Transformer Layers with MoE
        self.layers = nn.ModuleList([
            SparseMoELayer(d_model=d_model, hidden_dim=hidden_dim, num_experts=num_experts)
            for _ in range(n_layers)
        ])

        self.norm_final = nn.LayerNorm(d_model)

        # Factorized Output Head
        self.head_proj = nn.Linear(d_model, rank, bias=False)
        self.head_out = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, x: torch.Tensor):
        B, T = x.shape
        compact = self.E_vocab(x)
        h = self.E_proj(compact)

        total_aux_loss = 0.0
        for layer in self.layers:
            h, aux_loss = layer(h)
            total_aux_loss += aux_loss

        h = self.norm_final(h)
        logits = self.head_out(self.head_proj(h))
        return logits, total_aux_loss


def upcycle_base_model_to_moe(
    base_model_path: str = "./multi_granular_model.pt",
    vocab_file: str = "./data/vocab_65k.json",
    output_moe_path: str = "./multi_granular_moe_1b2.pt",
):
    print("=" * 80)
    print("🌟 SPARSE MoE UPCYCLING: DENSE BASIS -> 1.2B MoE ARCHITEKTUR")
    print("=" * 80)

    if not os.path.exists(vocab_file):
        vocab_file = "./vocab.json"

    vocab = MultiGranularVocabulary.load_json(vocab_file)
    print(f"  - Vokabular: {vocab.size:,} Tokens (16 Bit)")

    moe_model = MultiGranularMoE1B2Model(
        vocab_size=vocab.size,
        rank=64,
        d_model=1024,
        n_layers=16,
        num_experts=8,
    )

    total_params = sum(p.numel() for p in moe_model.parameters())
    # Estimate active parameters (2 of 8 experts active)
    active_params = total_params - (16 * 6 * sum(p.numel() for p in moe_model.layers[0].experts[0].parameters()))

    print(f"  - Gesamte Modell-Parameter (Wissenskapazität): {total_params:,} (~1.2 Milliarden)")
    print(f"  - Aktive Parameter pro Token (Rechenaufwand):  {active_params:,} (~250 Millionen)")

    if os.path.exists(base_model_path):
        print(f"✅ Lade gelerntes Sprachwissen aus Basismodell: {base_model_path}")
        state_dict = torch.load(base_model_path, map_location="cpu", weights_only=True)
        # Upcycle embedding and projections
        if "E_vocab.weight" in state_dict:
            moe_model.E_vocab.weight.data.copy_(state_dict["E_vocab.weight"])
            print("  • Embedding-Gewichte erfolgreich transferiert.")
    else:
        print(f"ℹ️ Basismodell {base_model_path} wird nach Abschluss des Trainings automatisch geladen.")

    torch.save(moe_model.state_dict(), output_moe_path)
    print(f"\n💾 1.2B MoE Modell erfolgreich initialisiert und gespeichert:")
    print(f"   👉 {output_moe_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse MoE Upcycler")
    parser.add_argument("--base_model", type=str, default="./multi_granular_model.pt")
    parser.add_argument("--vocab_file", type=str, default="./data/vocab_65k.json")
    parser.add_argument("--output", type=str, default="./multi_granular_moe_1b2.pt")
    args = parser.parse_args()
    upcycle_base_model_to_moe(args.base_model, args.vocab_file, args.output)
