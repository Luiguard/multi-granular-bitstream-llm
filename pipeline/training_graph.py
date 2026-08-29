#!/usr/bin/env python3
"""Knowledge Curriculum DAG & Dynamic Training Graph for Multi-Granular Bitstream LLM.

Provides:
- Strict DAG-based Knowledge Node hierarchy across 450+ real dataset shards.
- Dynamic Prerequisite-Gating: Complex reasoning nodes unlock based on foundational convergence.
- Loss-Aware Error Remediation Routing: Automatically backtracks sampling to prerequisite parents upon loss spikes.
- Live Graph Telemetry for real-time dashboard visualization.
"""

import os
import glob
import math
import random
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

from pipeline.bitstream import BitstreamDecoder


class KnowledgeNode:
    """Represents a discrete domain in the Training Knowledge Graph."""

    def __init__(
        self,
        node_id: str,
        name: str,
        description: str,
        shard_dirs: List[str],
        prerequisites: List[str],
        mastery_threshold: float = 4.5,
        base_weight: float = 1.0,
    ):
        self.node_id = node_id
        self.name = name
        self.description = description
        self.shard_dirs = shard_dirs
        self.prerequisites = prerequisites
        self.mastery_threshold = mastery_threshold
        self.base_weight = base_weight

        self.shard_files: List[str] = []
        self._discover_shards()

        self.current_loss: float = 10.0
        self.moving_loss: float = 10.0
        self.loss_history: List[float] = []
        self.sample_count: int = 0
        self.status: str = "LOCKED" if prerequisites else "ACTIVE"
        self.remediation_boost: float = 1.0

        self._cached_tokens: Optional[np.ndarray] = None
        self._current_shard_idx: int = 0
        self._token_cursor: int = 0

    def _discover_shards(self):
        self.shard_files = []
        for d in self.shard_dirs:
            if os.path.exists(d):
                self.shard_files.extend(glob.glob(os.path.join(d, "*.mgbs")))
                self.shard_files.extend(glob.glob(os.path.join(d, "*.shard")))
        self.shard_files = sorted(list(set(self.shard_files)))
        if self.shard_files:
            random.shuffle(self.shard_files)

    @property
    def total_shards(self) -> int:
        return len(self.shard_files)

    def _load_next_shard(self) -> bool:
        if not self.shard_files:
            return False

        attempts = 0
        while attempts < len(self.shard_files):
            shard_path = self.shard_files[self._current_shard_idx]
            self._current_shard_idx = (self._current_shard_idx + 1) % len(self.shard_files)
            attempts += 1

            try:
                if shard_path.endswith(".mgbs"):
                    _, tokens = BitstreamDecoder.load_from_file(shard_path)
                else:
                    tokens = np.fromfile(shard_path, dtype=np.uint16).tolist()

                if len(tokens) >= 512:
                    self._cached_tokens = np.array(tokens, dtype=np.int64)
                    self._token_cursor = 0
                    return True
            except Exception:
                continue

        return False

    def get_batch(self, batch_size: int = 1, seq_len: int = 7168) -> Tuple[torch.Tensor, torch.Tensor]:
        needed = batch_size * (seq_len + 1)
        if self._cached_tokens is None or self._token_cursor + needed >= len(self._cached_tokens):
            if not self._load_next_shard():
                # Fallback: synthetic pseudo-token pattern if shard loading exhausted
                self._cached_tokens = np.random.randint(0, 65536, size=needed * 4, dtype=np.int64)
                self._token_cursor = 0

        assert self._cached_tokens is not None
        tokens_arr: np.ndarray = self._cached_tokens

        # Extract sequential batch slices
        x_list = []
        y_list = []
        for _ in range(batch_size):
            start = self._token_cursor
            end = start + seq_len
            if end + 1 > len(tokens_arr):
                self._load_next_shard()
                if self._cached_tokens is not None:
                    tokens_arr = self._cached_tokens
                start = 0
                end = seq_len

            x_seq = tokens_arr[start:end]
            y_seq = tokens_arr[start + 1 : end + 1]
            self._token_cursor += seq_len

            x_list.append(x_seq)
            y_list.append(y_seq)

        x_tensor = torch.tensor(np.stack(x_list), dtype=torch.long)
        y_tensor = torch.tensor(np.stack(y_list), dtype=torch.long)
        return x_tensor, y_tensor

    def update_loss(self, loss_val: float):
        if math.isnan(loss_val) or math.isinf(loss_val):
            return

        self.current_loss = loss_val
        self.sample_count += 1
        if len(self.loss_history) == 0:
            self.moving_loss = loss_val
        else:
            self.moving_loss = 0.92 * self.moving_loss + 0.08 * loss_val

        self.loss_history.append(round(loss_val, 4))
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)

        # Update mastery status
        if self.sample_count >= 20 and self.moving_loss <= self.mastery_threshold:
            self.status = "MASTERED"
        elif self.status != "LOCKED":
            self.status = "ACTIVE"


class TrainingKnowledgeGraph:
    """Manages the DAG of Knowledge Nodes, curriculum gating, and remediation routing."""

    def __init__(self, base_dir: str = "/home/benjamin/Bilder"):
        self.base_dir = base_dir
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[Tuple[str, str]] = []
        self.remediation_log: List[str] = []
        self._build_default_graph()

    def _build_default_graph(self):
        # 1. Node 0: Foundation Bitstream & Syntax (Root)
        self.add_node(KnowledgeNode(
            node_id="node_0_foundation",
            name="Foundation Bitstream & Syntax",
            description="Elementare 16-Bit Viterbi-Tokens & Grundsyntax",
            shard_dirs=[os.path.join(self.base_dir, "data/shards")],
            prerequisites=[],
            mastery_threshold=6.5,
            base_weight=2.0,
        ))

        # 2. Node 1: Cyber & Web Knowledge (Requires Foundation)
        self.add_node(KnowledgeNode(
            node_id="node_1_cyber_web",
            name="Cyber & Web Knowledge",
            description="Netzwerkprotokolle, Web-Strukturen & Code-Tokens",
            shard_dirs=[
                os.path.join(self.base_dir, "data/cyber_web_knowledge/shards"),
                os.path.join(self.base_dir, "data/cyber_web_knowledge"),
            ],
            prerequisites=["node_0_foundation"],
            mastery_threshold=5.8,
            base_weight=1.5,
        ))

        # 3. Node 2: STEM, Math & Biology (Requires Foundation)
        self.add_node(KnowledgeNode(
            node_id="node_2_stem_math",
            name="STEM, Math & Science",
            description="Formale Mathematik, Naturwissenschaften & Algorithmen",
            shard_dirs=[
                os.path.join(self.base_dir, "data/stem_knowledge/shards"),
                os.path.join(self.base_dir, "data/biology_math/shards"),
                os.path.join(self.base_dir, "data/chinchilla_corpus/arxiv_shards"),
            ],
            prerequisites=["node_0_foundation"],
            mastery_threshold=5.5,
            base_weight=1.8,
        ))

        # 4. Node 3: World & History Corpus (Requires Cyber/Web + STEM)
        self.add_node(KnowledgeNode(
            node_id="node_3_world_corpus",
            name="World & History Corpus",
            description="Umfassendes Weltwissen (FineWeb-Edu 100BT, Wiki-DE, Gutenberg)",
            shard_dirs=[
                os.path.join(self.base_dir, "data/world_knowledge/shards"),
                os.path.join(self.base_dir, "data/world_knowledge/fineweb_edu_100BT_shards"),
                os.path.join(self.base_dir, "data/chinchilla_corpus/wiki_de_shards"),
                os.path.join(self.base_dir, "data/chinchilla_corpus/gutenberg_shards"),
                os.path.join(self.base_dir, "data/chinchilla_corpus/phil_shards"),
            ],
            prerequisites=["node_1_cyber_web", "node_2_stem_math"],
            mastery_threshold=5.0,
            base_weight=2.5,
        ))

        # 5. Node 4: AI & Deep Reasoning (Requires STEM + World)
        self.add_node(KnowledgeNode(
            node_id="node_4_ai_reasoning",
            name="AI & Deep Reasoning",
            description="Deep Learning Forschung, Beweise & StackExchange Logik",
            shard_dirs=[
                os.path.join(self.base_dir, "data/ai_research_knowledge/shards"),
                os.path.join(self.base_dir, "data/chinchilla_corpus/stackexchange_shards"),
            ],
            prerequisites=["node_2_stem_math", "node_3_world_corpus"],
            mastery_threshold=4.6,
            base_weight=2.0,
        ))

        # 6. Node 5: Instruction & Reflexive Alignment (Requires AI Reasoning)
        self.add_node(KnowledgeNode(
            node_id="node_5_instruction_alignment",
            name="Instruction & Alignment",
            description="Multi-Step Dialoge, Think-Tags & RLVR Guardrails",
            shard_dirs=[
                os.path.join(self.base_dir, "data/instructions/shards"),
                os.path.join(self.base_dir, "data/instructions"),
            ],
            prerequisites=["node_4_ai_reasoning"],
            mastery_threshold=4.0,
            base_weight=2.0,
        ))

        # Build Edges based on prerequisites
        self.edges = []
        for node in self.nodes.values():
            for prereq in node.prerequisites:
                self.edges.append((prereq, node.node_id))

    def add_node(self, node: KnowledgeNode):
        self.nodes[node.node_id] = node

    def update_gating(self):
        """Unlocks downstream nodes when prerequisite parent nodes achieve acceptable loss."""
        for node in self.nodes.values():
            if not node.prerequisites:
                if node.status == "LOCKED":
                    node.status = "ACTIVE"
                continue

            prereqs_met = True
            for p_id in node.prerequisites:
                p_node = self.nodes.get(p_id)
                # Prerequisite is considered met if it is MASTERED or has moving_loss <= threshold * 1.35
                if p_node is None or (p_node.status != "MASTERED" and p_node.moving_loss > p_node.mastery_threshold * 1.35):
                    prereqs_met = False
                    break

            if prereqs_met and node.status == "LOCKED":
                node.status = "ACTIVE"
            elif not prereqs_met and node.status != "MASTERED":
                node.status = "LOCKED"

    def sample_batch(self, batch_size: int = 1, seq_len: int = 7168) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Samples a training batch dynamically according to curriculum weights and remediation boost."""
        self.update_gating()

        eligible_nodes = [n for n in self.nodes.values() if n.status in ("ACTIVE", "MASTERED") and n.total_shards > 0]
        if not eligible_nodes:
            # Fallback to root node if everything is locked
            eligible_nodes = [self.nodes["node_0_foundation"]]

        # Calculate sampling weights: higher loss + higher remediation boost = higher sampling probability
        weights = []
        for node in eligible_nodes:
            # Nodes with higher loss get prioritized, scaled by their base weight and remediation boost
            w = node.base_weight * node.remediation_boost * (1.0 + math.log(max(1.0, node.moving_loss)))
            if node.status == "MASTERED":
                w *= 0.35  # Maintain mastery with lower sampling frequency
            weights.append(max(0.01, w))

        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        chosen_node = random.choices(eligible_nodes, weights=probs, k=1)[0]
        x, y = chosen_node.get_batch(batch_size=batch_size, seq_len=seq_len)
        return x, y, chosen_node.node_id

    def report_batch_loss(self, node_id: str, loss_val: float):
        """Updates moving loss and triggers Error Remediation Backtracking upon loss spikes."""
        node = self.nodes.get(node_id)
        if node is None:
            return

        # Check for loss spike (> 35% higher than moving average) to trigger remediation
        if node.sample_count > 5 and loss_val > node.moving_loss * 1.35:
            # Backtracking: boost sampling weight of all prerequisite parent nodes!
            for prereq_id in node.prerequisites:
                parent = self.nodes.get(prereq_id)
                if parent:
                    parent.remediation_boost = min(3.0, parent.remediation_boost + 0.4)
                    event_str = f"⚠️ Loss-Spike auf {node.name} ({loss_val:.2f}) -> Verstärke Eltern-Knoten {parent.name} (+40%)"
                    self.remediation_log.append(event_str)
                    if len(self.remediation_log) > 20:
                        self.remediation_log.pop(0)

        # Decay remediation boost slowly back to 1.0
        node.remediation_boost = max(1.0, node.remediation_boost * 0.98)
        node.update_loss(loss_val)

    def get_telemetry_state(self) -> Dict[str, Any]:
        """Serializes current graph topology, nodes, edges, and mastery for dashboard telemetry."""
        nodes_data = []
        for n in self.nodes.values():
            nodes_data.append({
                "id": n.node_id,
                "name": n.name,
                "description": n.description,
                "status": n.status,
                "current_loss": round(n.current_loss, 4),
                "moving_loss": round(n.moving_loss, 4),
                "sample_count": n.sample_count,
                "total_shards": n.total_shards,
                "mastery_threshold": n.mastery_threshold,
                "remediation_boost": round(n.remediation_boost, 2),
            })

        edges_data = [{"from": u, "to": v} for u, v in self.edges]

        mastered_count = sum(1 for n in self.nodes.values() if n.status == "MASTERED")
        active_count = sum(1 for n in self.nodes.values() if n.status == "ACTIVE")

        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "mastered_count": mastered_count,
            "active_count": active_count,
            "total_nodes": len(self.nodes),
            "recent_remediations": self.remediation_log[-5:],
        }


def build_default_training_graph(base_dir: str = "/home/benjamin/Bilder") -> TrainingKnowledgeGraph:
    return TrainingKnowledgeGraph(base_dir=base_dir)


if __name__ == "__main__":
    graph = build_default_training_graph()
    print("=" * 80)
    print("🧠 MULTI-GRANULAR BITSTREAM TRAINING KNOWLEDGE GRAPH")
    print("=" * 80)
    for n in graph.nodes.values():
        print(f"[{n.status:<8}] {n.name:<32} | Shards: {n.total_shards:>3} | Prereqs: {n.prerequisites}")
    print("=" * 80)
    print(f"Kanten: {len(graph.edges)} gerichtete Abhängigkeiten")
