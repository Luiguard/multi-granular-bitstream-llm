import math
from typing import Dict, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class FactorizedEmbedding:
    """Factorized Token Embedding Layer (Pure Vectorized NumPy):

    Decomposes large V x d embedding matrix into:
      E_vocab: (V x r)  [Compact vocabulary representation]
      E_proj:  (r x d)  [Projection into model dimension]
    where r << d (e.g., r=32, d=256).
    """

    def __init__(self, vocab_size: int, rank: int, embedding_dim: int, seed: int = 42):
        self.vocab_size = vocab_size
        self.rank = rank
        self.embedding_dim = embedding_dim

        # Xavier / He weight initialization
        rng = np.random.default_rng(seed)
        self.E_vocab = rng.normal(0.0, 1.0 / math.sqrt(rank), size=(vocab_size, rank)).astype(np.float32)
        self.E_proj = rng.normal(0.0, 1.0 / math.sqrt(embedding_dim), size=(rank, embedding_dim)).astype(np.float32)

        # Gradients storage
        self.grad_E_vocab = np.zeros_like(self.E_vocab)
        self.grad_E_proj = np.zeros_like(self.E_proj)

        # Cache for backpropagation
        self._last_input: Optional[np.ndarray] = None
        self._last_compact: Optional[np.ndarray] = None

    @property
    def parameter_count(self) -> int:
        return (self.vocab_size * self.rank) + (self.rank * self.embedding_dim)

    @property
    def standard_parameter_count(self) -> int:
        return self.vocab_size * self.embedding_dim

    @property
    def parameter_reduction_ratio(self) -> float:
        return self.parameter_count / max(1, self.standard_parameter_count)

    def forward(self, token_indices: np.ndarray) -> np.ndarray:
        """Forward pass:

        token_indices: shape (batch_size, seq_len)
        Returns: shape (batch_size, seq_len, embedding_dim)
        """
        self._last_input = token_indices
        # Lookup in compact table: shape (batch_size, seq_len, rank)
        compact = self.E_vocab[token_indices]
        self._last_compact = compact

        # Project to full embedding dimension: (batch, seq, rank) @ (rank, dim) -> (batch, seq, dim)
        output = np.matmul(compact, self.E_proj)
        return output

    def backward(self, grad_output: np.ndarray) -> None:
        """Computes exact analytical gradients for E_proj and E_vocab.

        grad_output: shape (batch_size, seq_len, embedding_dim)
        """
        if self._last_compact is None or self._last_input is None:
            raise RuntimeError("Cannot call backward before calling forward.")
        B, T, r = self._last_compact.shape
        d = self.embedding_dim

        flat_compact = self._last_compact.reshape(-1, r)  # (B*T, r)
        flat_grad_out = grad_output.reshape(-1, d)         # (B*T, d)

        self.grad_E_proj = np.matmul(flat_compact.T, flat_grad_out)

        # dL / dCompact = grad_output @ E_proj.T -> (B*T, r)
        grad_compact = np.matmul(flat_grad_out, self.E_proj.T)

        # Accumulate gradients into E_vocab
        self.grad_E_vocab.fill(0.0)
        flat_input = self._last_input.reshape(-1)
        np.add.at(self.E_vocab, flat_input, -0.01 * grad_compact)  # direct SGD update


class ByteWeightedCrossEntropyLoss:
    """Cross-Entropy Loss weighted by token byte-lengths to balance information density."""

    def __init__(self, id_to_byte_len: Dict[int, int], vocab_size: int):
        self.vocab_size = vocab_size
        weights = np.ones(vocab_size, dtype=np.float32)
        for t_id, b_len in id_to_byte_len.items():
            if t_id < vocab_size:
                weights[t_id] = float(max(1, b_len))
        self.weights = weights

    def compute(
        self, logits: np.ndarray, targets: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Computes byte-weighted cross-entropy loss and softmax gradient.

        logits: shape (batch_size, seq_len, vocab_size)
        targets: shape (batch_size, seq_len)
        Returns: (loss_scalar, grad_logits)
        """
        B, T, V = logits.shape
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        target_weights = self.weights[targets]  # shape (B, T)
        total_weight = np.sum(target_weights)

        batch_idx = np.arange(B)[:, None]
        seq_idx = np.arange(T)[None, :]
        target_probs = probs[batch_idx, seq_idx, targets]  # shape (B, T)

        nll = -np.log(np.maximum(target_probs, 1e-15))
        weighted_loss = np.sum(nll * target_weights) / max(1e-8, total_weight)

        grad_logits = probs.copy()
        grad_logits[batch_idx, seq_idx, targets] -= 1.0
        grad_logits *= (target_weights[:, :, None] / max(1e-8, total_weight))

        return float(weighted_loss), grad_logits


class MiniTransformerBlock:
    """Lightweight single-layer Transformer feed-forward with causal mask for verification."""

    def __init__(self, embedding_dim: int, hidden_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W_q = rng.normal(0, 0.02, (embedding_dim, embedding_dim)).astype(np.float32)
        self.W_k = rng.normal(0, 0.02, (embedding_dim, embedding_dim)).astype(np.float32)
        self.W_v = rng.normal(0, 0.02, (embedding_dim, embedding_dim)).astype(np.float32)
        self.W_ff1 = rng.normal(0, 0.02, (embedding_dim, hidden_dim)).astype(np.float32)
        self.W_ff2 = rng.normal(0, 0.02, (hidden_dim, embedding_dim)).astype(np.float32)
        self.dim = embedding_dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: shape (batch_size, seq_len, dim)"""
        B, T, d = x.shape
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)

        # Scaled dot-product causal attention
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(d)
        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores[:, causal_mask] = -1e9

        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True)

        attn_out = np.matmul(attn_weights, V)
        x_norm = x + attn_out

        # Feed-forward
        ff = np.maximum(0, np.matmul(x_norm, self.W_ff1))  # ReLU
        out = x_norm + np.matmul(ff, self.W_ff2)
        return out


if HAS_TORCH:
    class FactorizedEmbeddingTorch(nn.Module):
        """PyTorch Module for Factorized Embedding Table."""

        def __init__(self, vocab_size: int, rank: int, embedding_dim: int):
            super().__init__()
            self.E_vocab = nn.Embedding(vocab_size, rank)
            self.E_proj = nn.Linear(rank, embedding_dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.E_proj(self.E_vocab(x))

    class ByteWeightedCrossEntropyLossTorch(nn.Module):
        """PyTorch Module for Byte-Weighted Cross-Entropy Loss."""

        weights: torch.Tensor

        def __init__(self, id_to_byte_len: Dict[int, int], vocab_size: int):
            super().__init__()
            weights = torch.ones(vocab_size, dtype=torch.float32)
            for t_id, b_len in id_to_byte_len.items():
                if t_id < vocab_size:
                    weights[t_id] = float(max(1, b_len))
            self.register_buffer("weights", weights)

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            # logits: (B, T, V), targets: (B, T)
            B, T, V = logits.shape
            flat_logits = logits.view(-1, V)
            flat_targets = targets.view(-1)
            token_weights = self.weights[flat_targets]
            loss_unreduced = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            weighted_loss = torch.sum(loss_unreduced * token_weights) / torch.sum(token_weights)
            return weighted_loss
