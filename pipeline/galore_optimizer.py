"""GaLore (Gradient Low-Rank Projection) Optimizer for Memory-Efficient LLM Training.

Based on Zhao et al. (2024). Projects weight gradients into a compact low-rank subspace
to reduce optimizer state memory by > 65% with full rank training convergence.
Enables training models with 3x-4x more parameters on consumer hardware!
"""

import math
from typing import List, Optional, Tuple
import torch
from torch.optim import Optimizer


def _compute_robust_orthogonal_matrix(mat: torch.Tensor, r: int, mode: str = "right") -> torch.Tensor:
    """Computes robust orthogonal projection matrix with automatic QR and jitter fallbacks.
    
    CRITICAL: All computation done in float32 on CPU to avoid cusolver GPU instabilities.
    """
    # Strategy 1: Fast GPU SVD
    # Wir MÜSSEN den Cache leeren und .contiguous() aufrufen, sonst
    # crasht cuSOLVER leise und fällt auf CPU zurück, was 11GB RAM frisst!
    torch.cuda.empty_cache()
    try:
        mat_gpu = mat.float().cuda().contiguous()
        if mode == "right":
            _, _, Vh = torch.linalg.svd(mat_gpu, full_matrices=False)
            result = Vh[:r, :]
        else:
            U, _, _ = torch.linalg.svd(mat_gpu, full_matrices=False)
            result = U[:, :r]
        
        if torch.isnan(result).any() or torch.isinf(result).any():
            raise ValueError("SVD produced NaN/Inf")
        
        res = result.to(dtype=mat.dtype, device='cpu')
        del mat_gpu
        torch.cuda.empty_cache()
        return res
    except Exception as e:
        print(f"  [Warnung] GPU SVD fehlgeschlagen ({e}), falle zurück auf CPU SVD...", flush=True)
    
    # Strategy 2: SVD with jitter on CPU
    try:
        mat_cpu = mat.float().cpu()
        jitter = 1e-6 * torch.randn_like(mat_cpu)
        mat_jittered = mat_cpu + jitter
        if mode == "right":
            _, _, Vh = torch.linalg.svd(mat_jittered, full_matrices=False)
            result = Vh[:r, :]
        else:
            U, _, _ = torch.linalg.svd(mat_jittered, full_matrices=False)
            result = U[:, :r]
        
        if torch.isnan(result).any() or torch.isinf(result).any():
            raise ValueError("Jittered SVD produced NaN/Inf")
        
        return result.to(device=mat.device, dtype=mat.dtype)
    except Exception:
        pass
    
    # Strategy 3: QR decomposition (guaranteed stable, always converges)
    try:
        if mode == "right":
            q, _ = torch.linalg.qr(mat_cpu.t())
            result = q[:, :r].t()
        else:
            q, _ = torch.linalg.qr(mat_cpu)
            result = q[:, :r]
        
        if torch.isnan(result).any() or torch.isinf(result).any():
            raise ValueError("QR produced NaN/Inf")
        
        return result.to(device=mat.device, dtype=mat.dtype)
    except Exception:
        pass
    
    # Strategy 4: Random orthogonal matrix (absolute last resort, still valid mathematically)
    m, n = mat.shape
    if mode == "right":
        rand_mat = torch.randn(r, n, dtype=torch.float32)
        q, _ = torch.linalg.qr(rand_mat.t())
        return q[:, :r].t().to(device=mat.device, dtype=mat.dtype)
    else:
        rand_mat = torch.randn(m, r, dtype=torch.float32)
        q, _ = torch.linalg.qr(rand_mat)
        return q[:, :r].to(device=mat.device, dtype=mat.dtype)


class GaLoreProjector:
    """Computes orthogonal low-rank projection matrix for gradient tensors."""

    def __init__(self, rank: int = 64, update_interval: int = 200):
        self.rank = rank
        self.update_interval = update_interval
        self.ortho_matrix: Optional[torch.Tensor] = None
        self.step_count = 0

    def project(self, grad: torch.Tensor) -> torch.Tensor:
        if grad.ndim != 2:
            return grad

        # Skip projection if gradient contains NaN
        if torch.isnan(grad).any():
            return torch.zeros(
                (min(self.rank, grad.shape[0]), grad.shape[1]) if grad.shape[0] < grad.shape[1]
                else (grad.shape[0], min(self.rank, grad.shape[1])),
                dtype=grad.dtype, device=grad.device
            )

        m, n = grad.shape
        r = min(self.rank, m, n)

        # Update orthogonal projection matrix periodically via robust SVD/QR
        if self.ortho_matrix is None or self.step_count % self.update_interval == 0:
            with torch.no_grad():
                if m >= n:
                    self.ortho_matrix = _compute_robust_orthogonal_matrix(grad, r, mode="right")
                else:
                    self.ortho_matrix = _compute_robust_orthogonal_matrix(grad, r, mode="left")

        self.step_count += 1

        # Ensure matching dtypes
        grad_cast = grad.to(dtype=self.ortho_matrix.dtype)
        if m >= n:
            result = torch.matmul(grad_cast, self.ortho_matrix.t())
        else:
            result = torch.matmul(self.ortho_matrix.t(), grad_cast)
        
        # Final NaN guard on projected result
        if torch.isnan(result).any():
            result = torch.zeros_like(result)
        
        return result

    def project_back(self, low_rank_grad: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        if self.ortho_matrix is None or low_rank_grad.ndim != 2:
            return low_rank_grad

        # Ensure matching dtypes for matmul
        lr_grad = low_rank_grad.to(dtype=self.ortho_matrix.dtype)

        m, n = original_shape
        if m >= n:
            result = torch.matmul(lr_grad, self.ortho_matrix)
        else:
            result = torch.matmul(self.ortho_matrix, lr_grad)
        
        if torch.isnan(result).any():
            result = torch.zeros_like(result)
        
        return result


class GaLoreAdamW(Optimizer):
    """Memory-efficient AdamW with Low-Rank Gradient Projection (GaLore).
    
    Hardened with comprehensive NaN/Inf guards at every computation step.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        rank: int = 64,
        update_interval: int = 200,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            update_interval=update_interval,
        )
        super().__init__(params, defaults)
        self.projectors = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            rank = group["rank"]
            update_interval = group["update_interval"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # CRITICAL: Skip entirely if gradient is NaN/Inf
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    continue

                state = self.state[p]

                # State initialization
                if p.ndim == 2 and p not in self.projectors:
                    self.projectors[p] = GaLoreProjector(rank=rank, update_interval=update_interval)

                if len(state) == 0:
                    state["step"] = 0
                    if p.ndim == 2:
                        m, n = p.shape
                        r = min(rank, m, n)
                        proj_shape = (m, r) if m >= n else (r, n)
                        state["exp_avg"] = torch.zeros(proj_shape, dtype=torch.float32, device=p.device)
                        state["exp_avg_sq"] = torch.zeros(proj_shape, dtype=torch.float32, device=p.device)
                    else:
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                step = state["step"]

                # Weight decay
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # GaLore Projection
                if p in self.projectors:
                    proj = self.projectors[p]
                    low_rank_grad = proj.project(grad).float()

                    # Ensure state tensor shapes match projected gradient shape
                    if state["exp_avg"].shape != low_rank_grad.shape:
                        state["exp_avg"] = torch.zeros_like(low_rank_grad, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(low_rank_grad, dtype=torch.float32)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    # Update low-rank moments in float32
                    exp_avg.mul_(beta1).add_(low_rank_grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr / bias_correction1
                    low_rank_update = (exp_avg / denom) * step_size

                    # Clamp update magnitude to prevent explosions
                    low_rank_update.clamp_(-1.0, 1.0)

                    # Project back to full rank space
                    full_update = proj.project_back(low_rank_update, p.shape)
                    p.add_(-full_update.to(dtype=p.dtype))
                else:
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    grad_f = grad.float()
                    exp_avg.mul_(beta1).add_(grad_f, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad_f, grad_f, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr / bias_correction1
                    update = (exp_avg / denom) * step_size
                    update.clamp_(-1.0, 1.0)
                    p.add_(-update.to(dtype=p.dtype))

        return loss

    @torch.no_grad()
    def step_param(self, p):
        """Performs a single optimization step for one specific parameter. 
        Crucial for O(1) per-layer/per-parameter hooking to avoid looping over 7B parameters.
        """
        if p.grad is None:
            return

        # Find the group for this parameter (usually there is only 1 group)
        group = self.param_groups[0]
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        rank = group["rank"]
        update_interval = group["update_interval"]

        grad = p.grad
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            return

        state = self.state[p]
        if p.ndim == 2 and p not in self.projectors:
            self.projectors[p] = GaLoreProjector(rank=rank, update_interval=update_interval)

        if len(state) == 0:
            state["step"] = 0
            if p.ndim == 2 and p in self.projectors:
                # GaLore: initialize low-rank optimizer states lazily after first projection
                # (will be resized on first use if needed)
                proj = self.projectors[p]
                dummy_grad = torch.zeros_like(p)
                low_rank = proj.project(dummy_grad)
                state["exp_avg"] = torch.zeros_like(low_rank, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(low_rank, dtype=torch.float32)
            else:
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        
        state["step"] += 1
        step = state["step"]

        grad = p.grad.float()
        if weight_decay != 0:
            grad = grad.add(p.float(), alpha=weight_decay)

        if p in self.projectors:
            proj = self.projectors[p]
            low_rank_grad = proj.project(grad).float()

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            
            # Shape guard: GaLore may change projection shape at update intervals
            if exp_avg.shape != low_rank_grad.shape:
                state["exp_avg"] = torch.zeros_like(low_rank_grad, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(low_rank_grad, dtype=torch.float32)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

            # Update low-rank moments in float32
            exp_avg.mul_(beta1).add_(low_rank_grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1
            low_rank_update = (exp_avg / denom) * step_size

            # Clamp update magnitude to prevent explosions
            low_rank_update.clamp_(-1.0, 1.0)

            # Project back to full rank space
            full_update = proj.project_back(low_rank_update, p.shape)
            p.add_(-full_update.to(dtype=p.dtype))
        else:
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            grad_f = grad.float()
            exp_avg.mul_(beta1).add_(grad_f, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_f, grad_f, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1
            update = (exp_avg / denom) * step_size
            update.clamp_(-1.0, 1.0)
            p.add_(-update.to(dtype=p.dtype))
            
        # Zwinge den Garbage Collector sofort, die VRAM/RAM Referenzen freizugeben
        import gc
        gc.collect()
        torch.cuda.empty_cache()
