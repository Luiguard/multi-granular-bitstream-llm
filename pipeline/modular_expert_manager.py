#!/usr/bin/env python3
"""
Modular Expert Management System for Sparse MoE Architectures.
Enables:
1. Exporting expert packs (e.g. STEM experts, Code experts) into standalone .pt files.
2. Importing / Hot-swapping expert packs into existing MoE checkpoints.
3. Splicing multiple expert packs into expanded models (e.g. 12 -> 24 experts for 14B).
4. Zero-Shot Router Alignment via Cosine Representation Centering (1-second analytical alignment).
"""

import copy
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, "/home/benjamin/Bilder")


class ModularExpertManager:
    """Manages modular saving, loading, branching, and splicing of MoE experts."""

    @staticmethod
    def inspect_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        """Inspects expert counts, layer counts, and dimensions in a checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} nicht gefunden!")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        sd = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt

        # Discover layers and experts
        expert_keys = [k for k in sd.keys() if ".experts." in k]
        layer_indices = set()
        expert_indices = set()

        for k in expert_keys:
            parts = k.split(".")
            for idx, p in enumerate(parts):
                if p == "moe_layers" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                    layer_indices.add(int(parts[idx + 1]))
                if p == "experts" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                    expert_indices.add(int(parts[idx + 1]))

        n_layers = len(layer_indices) if layer_indices else 24
        n_experts = len(expert_indices) if expert_indices else 12

        d_model = sd.get("E_proj.weight", torch.zeros(2048, 64)).shape[0]
        vocab_size = sd.get("E_vocab.weight", torch.zeros(262144, 64)).shape[0]

        step = ckpt.get("step", 0) if isinstance(ckpt, dict) else 0
        tokens = ckpt.get("tokens_processed", 0) if isinstance(ckpt, dict) else 0

        info = {
            "checkpoint_path": checkpoint_path,
            "step": step,
            "tokens_processed": tokens,
            "n_layers": n_layers,
            "num_experts_per_layer": n_experts,
            "total_experts": n_layers * n_experts,
            "d_model": d_model,
            "vocab_size": vocab_size,
            "total_parameter_keys": len(sd),
        }
        return info

    @staticmethod
    def export_expert_pack(
        checkpoint_path: str,
        expert_indices: List[int],
        output_pack_path: str,
        pack_name: str = "custom_expert_pack",
        description: str = "",
    ) -> Dict[str, Any]:
        """Extracts a specific subset of experts across all layers into a standalone pack file."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        sd = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt

        extracted_weights = {}
        for k, v in sd.items():
            if ".experts." in k:
                parts = k.split(".")
                for idx, p in enumerate(parts):
                    if p == "experts" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                        exp_idx = int(parts[idx + 1])
                        if exp_idx in expert_indices:
                            extracted_weights[k] = v.clone()

        pack_metadata = {
            "pack_name": pack_name,
            "description": description,
            "source_checkpoint": checkpoint_path,
            "expert_indices": expert_indices,
            "num_experts": len(expert_indices),
            "weights": extracted_weights,
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_pack_path)), exist_ok=True)
        torch.save(pack_metadata, output_pack_path)
        print(f"  💾 Experten-Paket '{pack_name}' ({len(expert_indices)} Experten) gespeichert -> {output_pack_path}")
        return pack_metadata

    @staticmethod
    def import_expert_pack(
        checkpoint_path: str,
        expert_pack_path: str,
        target_indices: List[int],
        output_checkpoint_path: Optional[str] = None,
    ) -> str:
        """Injects an exported expert pack into specific expert slots of a checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt

        pack = torch.load(expert_pack_path, map_location="cpu", weights_only=False)
        src_indices = pack["expert_indices"]
        pack_weights = pack["weights"]

        if len(src_indices) != len(target_indices):
            raise ValueError(f"Anzahl Quell-Experten ({len(src_indices)}) != Ziel-Experten ({len(target_indices)})!")

        index_mapping = dict(zip(src_indices, target_indices))

        for k, v in pack_weights.items():
            parts = k.split(".")
            new_parts = []
            for idx, p in enumerate(parts):
                if p == "experts" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                    old_exp = int(parts[idx + 1])
                    new_exp = index_mapping.get(old_exp, old_exp)
                    new_parts.append(p)
                    new_parts.append(str(new_exp))
                    # Skip next because we appended it
                    continue
                if idx > 0 and parts[idx - 1] == "experts" and p.isdigit():
                    continue
                new_parts.append(p)

            new_key = ".".join(new_parts)
            sd[new_key] = v.clone()

        if output_checkpoint_path is None:
            output_checkpoint_path = checkpoint_path

        ckpt["model_state_dict"] = sd
        torch.save(ckpt, output_checkpoint_path)
        print(f"  ✅ Experten-Paket erfolgreich in Slots {target_indices} importiert -> {output_checkpoint_path}")
        return output_checkpoint_path

    @staticmethod
    def splice_to_14b(
        checkpoint_a: str,
        checkpoint_b: str,
        output_14b_path: str,
    ) -> str:
        """Splices two 12-expert checkpoints into a single 24-expert 14.2B model checkpoint."""
        print(f"  🧬 Starte MoE-Splicing: {checkpoint_a} + {checkpoint_b} -> 14B...")
        ckpt_a = torch.load(checkpoint_a, map_location="cpu", weights_only=False)
        sd_a = ckpt_a["model_state_dict"] if (isinstance(ckpt_a, dict) and "model_state_dict" in ckpt_a) else ckpt_a

        ckpt_b = torch.load(checkpoint_b, map_location="cpu", weights_only=False)
        sd_b = ckpt_b["model_state_dict"] if (isinstance(ckpt_b, dict) and "model_state_dict" in ckpt_b) else ckpt_b

        sd_14b = copy.deepcopy(sd_a)

        # 1. Map all experts from ckpt_b into slots 12 to 23
        for k, v in sd_b.items():
            if ".experts." in k:
                parts = k.split(".")
                new_parts = []
                for idx, p in enumerate(parts):
                    if p == "experts" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                        old_exp = int(parts[idx + 1])
                        new_exp = old_exp + 12  # Slots 12-23
                        new_parts.append(p)
                        new_parts.append(str(new_exp))
                        continue
                    if idx > 0 and parts[idx - 1] == "experts" and p.isdigit():
                        continue
                    new_parts.append(p)

                new_key = ".".join(new_parts)
                sd_14b[new_key] = v.clone()

        # 2. Expand router gates from 12 to 24 outputs
        for k, v in list(sd_14b.items()):
            if ".router.gate.weight" in k:
                # Shape: [12, d_model] -> [24, d_model]
                w_a = v
                w_b = sd_b[k]
                w_24 = torch.cat([w_a, w_b], dim=0)
                sd_14b[k] = w_24

        ckpt_14b = {
            "step": max(ckpt_a.get("step", 0), ckpt_b.get("step", 0)),
            "tokens_processed": ckpt_a.get("tokens_processed", 0) + ckpt_b.get("tokens_processed", 0),
            "model_state_dict": sd_14b,
            "num_experts": 24,
            "architecture": "14B Sparse MoE (24 Experts / Top-2)",
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_14b_path)), exist_ok=True)
        torch.save(ckpt_14b, output_14b_path)
        print(f"  🎉 14.2B Modell mit 24 Experten erfolgreich erzeugt -> {output_14b_path}")
        return output_14b_path

    @staticmethod
    def align_router_zero_shot(state_dict: Dict[str, torch.Tensor], num_experts: int = 12) -> Dict[str, torch.Tensor]:
        """Aligns router gating matrices analytically via Cosine Centroid Centering in 1 second."""
        for k, v in state_dict.items():
            if ".router.gate.weight" in k:
                prefix = k.replace(".router.gate.weight", "")
                d_model = v.shape[1]
                
                # Compute weight centroid for each expert in this layer
                centroids = []
                for exp_i in range(num_experts):
                    w_gate_key = f"{prefix}.experts.{exp_i}.w_gate.weight"
                    if w_gate_key in state_dict:
                        w_expert = state_dict[w_gate_key]  # [hidden_dim, d_model]
                        centroid = w_expert.mean(dim=0)  # [d_model]
                        centroid = F.normalize(centroid, p=2, dim=0)
                        centroids.append(centroid)
                    else:
                        centroids.append(torch.randn(d_model))

                if centroids:
                    new_gate = torch.stack(centroids, dim=0).to(dtype=v.dtype, device=v.device)
                    state_dict[k] = new_gate

        return state_dict


if __name__ == "__main__":
    latest_ckpt = "/home/benjamin/Bilder/checkpoints/7b_checkpoint_latest.pt"
    if os.path.exists(latest_ckpt):
        info = ModularExpertManager.inspect_checkpoint(latest_ckpt)
        print("=" * 80)
        print("🔍 MODULAR MOE EXPERT INSPECTOR")
        print("=" * 80)
        for k, v in info.items():
            print(f"  • {k:<25}: {v}")
