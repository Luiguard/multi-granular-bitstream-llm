#!/usr/bin/env python3
"""
YaRN: Yet another RoPE extensioN for Long-Context Attention Scaling.
Enables expanding the 7.45B MoE context window from 7k to 32k-64k tokens without VRAM explosion.

Features:
1. Progressive Non-Uniform Frequency Interpolation (YaRN Ramp).
2. Attention Temperature Scaling (Entropy Compensation t_yarn = 0.1 * ln(s) + 1.0).
3. Complex RoPE Matrix Application over arbitrary sequence lengths.
"""

import math
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


class YaRNScaledRotaryEmbedding(nn.Module):
    """
    Computes YaRN-scaled Rotary Positional Embeddings for extended context windows (up to 65,536 tokens).
    """
    def __init__(
        self,
        dim: int = 64,
        max_position_embeddings: int = 65536,
        base: float = 10000.0,
        scale: float = 4.0,  # 4x scaling (7k -> 28k/32k tokens)
        original_max_position_embeddings: int = 7168,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor

        # 1. Compute Base Frequencies
        pos = torch.arange(0, dim, 2, dtype=torch.float32)
        inv_freq_base = 1.0 / (base ** (pos / dim))

        # 2. YaRN Progressive Non-Uniform Interpolation
        if scale > 1.0:
            low = math.floor(dim * math.log(original_max_position_embeddings / (beta_fast * 2 * math.pi)) / math.log(base))
            high = math.ceil(dim * math.log(original_max_position_embeddings / (beta_slow * 2 * math.pi)) / math.log(base))
            low = max(0, low)
            high = min(dim // 2 - 1, high)

            inv_freq_interpolated = inv_freq_base / scale
            ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / max(1, high - low), 0.0, 1.0)
            
            # Blend interpolated (high wavelengths) and original (low wavelengths)
            inv_freq = (1.0 - ramp) * inv_freq_interpolated + ramp * inv_freq_base
        else:
            inv_freq = inv_freq_base

        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 3. Attention Entropy Temperature Scaling (YaRN factor)
        if scale > 1.0:
            # Temperature formula: t = 0.1 * ln(s) + 1.0
            self.mscale = (0.1 * math.log(scale) + 1.0) * attn_factor
        else:
            self.mscale = 1.0 * attn_factor

        self._cached_cos: Optional[torch.Tensor] = None
        self._cached_sin: Optional[torch.Tensor] = None
        self._cached_seq_len = 0

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len <= self._cached_seq_len and self._cached_cos is not None and self._cached_cos.device == device:
            return

        self._cached_seq_len = max(seq_len, self.max_position_embeddings)
        t = torch.arange(self._cached_seq_len, device=device, dtype=torch.float32)
        inv_freq_tensor: torch.Tensor = self.inv_freq  # type: ignore
        freqs = torch.outer(t, inv_freq_tensor.to(device))
        
        # Duplicate for real/imag or interleaved pairs
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cached_cos = (emb.cos() * self.mscale).to(dtype)
        self._cached_sin = (emb.sin() * self.mscale).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies YaRN RoPE to query and key tensors: q, k: (B, H, T, D)
        """
        device = q.device
        dtype = q.dtype
        self._update_cos_sin_cache(seq_len, device, dtype)

        assert self._cached_cos is not None and self._cached_sin is not None
        cos = self._cached_cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
        sin = self._cached_sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

        # Rotate half: [-x2, x1]
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot


if __name__ == "__main__":
    yarn = YaRNScaledRotaryEmbedding(dim=64, scale=4.0, max_position_embeddings=32768)
    q = torch.randn(1, 8, 32768, 64)
    k = torch.randn(1, 8, 32768, 64)
    q_out, k_out = yarn(q, k, seq_len=32768)
    print("📏 YaRN Long-Context Scaler Test:")
    print(f"  • Skalierung: {yarn.scale}x (Kontext-Fenster: 32.768 Tokens)")
    print(f"  • Output Shape: {q_out.shape}")
    print(f"  • Temperature MScale: {yarn.mscale:.4f}")
