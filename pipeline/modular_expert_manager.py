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
        return ModularExpertManager.splice_multi_models([checkpoint_a, checkpoint_b], output_14b_path)

    @staticmethod
    def splice_multi_models(
        checkpoint_paths: List[str],
        output_path: str,
    ) -> str:
        """Splices N arbitrary model checkpoints (e.g. 12x 144B models) into a unified Multi-MoE supermodel."""
        if not checkpoint_paths:
            raise ValueError("Mindestens 1 Checkpoint-Pfad muss angegeben werden!")

        print(f"  🧬 Starte Multi-Modell MoE-Splicing über {len(checkpoint_paths)} Checkpoints...")
        
        # Load base checkpoint 0
        base_ckpt = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
        base_sd = base_ckpt["model_state_dict"] if (isinstance(base_ckpt, dict) and "model_state_dict" in base_ckpt) else base_ckpt
        
        fused_sd = copy.deepcopy(base_sd)
        current_expert_offset = 0
        
        # Discover experts per layer in base
        base_exp_count = len(set(
            int(k.split(".")[k.split(".").index("experts") + 1])
            for k in base_sd.keys() if ".experts." in k
        )) or 1
        current_expert_offset = base_exp_count

        for ckpt_idx in range(1, len(checkpoint_paths)):
            src_path = checkpoint_paths[ckpt_idx]
            ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
            sd = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt

            src_exp_count = len(set(
                int(k.split(".")[k.split(".").index("experts") + 1])
                for k in sd.keys() if ".experts." in k
            )) or 1

            for k, v in sd.items():
                if ".experts." in k:
                    parts = k.split(".")
                    new_parts = []
                    for idx, p in enumerate(parts):
                        if p == "experts" and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                            old_exp = int(parts[idx + 1])
                            new_exp = old_exp + current_expert_offset
                            new_parts.append(p)
                            new_parts.append(str(new_exp))
                            continue
                        if idx > 0 and parts[idx - 1] == "experts" and p.isdigit():
                            continue
                        new_parts.append(p)

                    new_key = ".".join(new_parts)
                    fused_sd[new_key] = v.clone()

            # Expand router gates
            for k, v in list(fused_sd.items()):
                if ".router.gate.weight" in k and k in sd:
                    fused_sd[k] = torch.cat([fused_sd[k], sd[k]], dim=0)

            current_expert_offset += src_exp_count

        total_experts = current_expert_offset
        fused_ckpt = {
            "step": max(base_ckpt.get("step", 0) if isinstance(base_ckpt, dict) else 0, 0),
            "tokens_processed": sum(
                (torch.load(p, map_location="cpu", weights_only=False).get("tokens_processed", 0) if isinstance(torch.load(p, map_location="cpu", weights_only=False), dict) else 0)
                for p in checkpoint_paths
            ),
            "model_state_dict": fused_sd,
            "num_experts": total_experts,
            "architecture": f"Multi-MoE Unified Supermodel ({len(checkpoint_paths)} Modul-Packs / {total_experts} Gesamt-Experten)",
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(fused_ckpt, output_path)
        print(f"  🎉 Multi-MoE Modell mit {total_experts} Experten erfolgreich fusioniert -> {output_path}")
        return output_path

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
