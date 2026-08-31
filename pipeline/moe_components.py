"""Sparse Mixture of Experts (MoE) Architecture for Multi-Granular Bitstream LLMs.

Enables massive model capacity (e.g. 1.2 Billion total parameters) while only activating
a fraction (e.g. 250 Million parameters) per token, perfectly suited for running on consumer hardware!
"""

import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.nemotron_components import SwiGLUFeedForward


class Top2GatingRouter(nn.Module):
    """Top-2 Gating Router with Load Balancing Loss as used in Mixtral & DeepSeek."""

    def __init__(self, d_model: int, num_experts: int = 8):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        domain_cluster: Optional[int] = None,
        expert_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: (B * T, d_model)
        logits = self.gate(x).float().clamp(-30.0, 30.0)

        # Apply soft cluster bias if domain is specified (e.g. 12 experts -> 3 clusters of 4)
        if domain_cluster is not None and self.num_experts >= 3:
            cluster_size = max(1, self.num_experts // 3)
            start_idx = (domain_cluster % 3) * cluster_size
            end_idx = min(self.num_experts, start_idx + cluster_size)
            bias = torch.zeros(self.num_experts, dtype=logits.dtype, device=logits.device)
            bias[start_idx:end_idx] = 1.25  # Soft preference bias for domain experts
            logits = logits + bias

        if expert_bias is not None:
            logits = logits + expert_bias.to(device=logits.device, dtype=logits.dtype)

        weights = F.softmax(logits, dim=-1).type_as(x)

        # Select Top-2 experts
        top2_weights, top2_indices = torch.topk(weights, k=2, dim=-1)

        # Normalize weights safely
        weight_sum = torch.clamp(top2_weights.sum(dim=-1, keepdim=True), min=1e-6)
        top2_weights = top2_weights / weight_sum

        # Auxiliary Load Balancing Loss
        density = weights.float().mean(dim=0)
        fraction = (weights > 0.1).float().mean(dim=0)
        aux_loss = (self.num_experts * torch.sum(density * fraction)).type_as(x)

        return top2_indices, top2_weights, aux_loss


class SparseMoELayer(nn.Module):
    """Sparse Mixture-of-Experts Layer replacing dense FFN."""

    def __init__(self, d_model: int, hidden_dim: int, num_experts: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.router = Top2GatingRouter(d_model, num_experts=num_experts)

        # ModuleList of SwiGLU Experts
        self.experts = nn.ModuleList([
            SwiGLUFeedForward(d_model, hidden_dim)
            for _ in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        domain_cluster: Optional[int] = None,
        expert_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        flat_x = x.view(-1, C)

        top2_indices, top2_weights, aux_loss = self.router(flat_x, domain_cluster=domain_cluster, expert_bias=expert_bias)
        final_output = torch.zeros_like(flat_x)

        # Dispatch tokens to experts
        for expert_idx in range(self.num_experts):
            # Mask for tokens routed to this expert (either 1st or 2nd choice)
            mask1 = (top2_indices[:, 0] == expert_idx)
            mask2 = (top2_indices[:, 1] == expert_idx)

            if mask1.any():
                expert_input1 = flat_x[mask1]
                w1 = top2_weights[mask1, 0].unsqueeze(-1)
                final_output[mask1] += self.experts[expert_idx](expert_input1) * w1

            if mask2.any():
                expert_input2 = flat_x[mask2]
                w2 = top2_weights[mask2, 1].unsqueeze(-1)
                final_output[mask2] += self.experts[expert_idx](expert_input2) * w2

        return final_output.view(B, T, C), aux_loss
