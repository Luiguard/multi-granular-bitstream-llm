#!/usr/bin/env python3
"""
Native 16-Bit Bitstream Knowledge Graph Memory Engine.
Direct token-space memory graph operating on Viterbi uint16 token IDs.
Zero translation overhead, sub-millisecond retrieval, and multi-hop graph traversal.
"""

import json
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, "/home/benjamin/Bilder")

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any

from pipeline.tokenizer import ViterbiTokenizer
from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.native_bitstream_tgat import NativeBitstreamTGAT, TemporalBochnerEncoder


@dataclass
class BitstreamEdge:
    target_id: str
    relation_text: str
    relation_tokens: List[int]
    weight: float = 1.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "relation_text": self.relation_text,
            "relation_tokens": self.relation_tokens,
            "weight": self.weight,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BitstreamEdge':
        return cls(
            target_id=d["target_id"],
            relation_text=d["relation_text"],
            relation_tokens=d["relation_tokens"],
            weight=d.get("weight", 1.0),
            created_at=d.get("created_at", time.time())
        )


@dataclass
class BitstreamNode:
    id: str
    label: str
    tokens: List[int]
    category: str = "concept"  # user, project, system, tech, math, minecraft
    edges: List[BitstreamEdge] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "tokens": self.tokens,
            "category": self.category,
            "edges": [e.to_dict() for e in self.edges],
            "attributes": self.attributes,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BitstreamNode':
        node = cls(
            id=d["id"],
            label=d["label"],
            tokens=d["tokens"],
            category=d.get("category", "concept"),
            attributes=d.get("attributes", {}),
            created_at=d.get("created_at", time.time())
        )
        node.edges = [BitstreamEdge.from_dict(e) for e in d.get("edges", [])]
        return node


class BitstreamGraphMemory:
    """
    High-Speed In-Memory & Persistent Knowledge Graph in 16-Bit Token Space.
    """

    def __init__(self, tokenizer: Optional[ViterbiTokenizer] = None, storage_path: str = "/home/benjamin/Bilder/data/bitstream_graph_memory.json"):
        if tokenizer is None:
            vocab_path = "/home/benjamin/Bilder/data/vocab_65k.json"
            if not os.path.exists(vocab_path):
                vocab_path = "/home/benjamin/Bilder/vocab.json"
            if os.path.exists(vocab_path):
                vocab = MultiGranularVocabulary.load_json(vocab_path)
            else:
                vocab = MultiGranularVocabulary()
            self.tokenizer = ViterbiTokenizer(vocab)
        else:
            self.tokenizer = tokenizer

        self.storage_path = storage_path
        self.tgat_binary_path = storage_path.replace(".json", ".tgat")
        self.nodes: Dict[str, BitstreamNode] = {}
        self.token_to_nodes: Dict[int, Set[str]] = {}
        self.tgat = NativeBitstreamTGAT(bochner_dim=64)

        # Load or initialize default knowledge graph
        if os.path.exists(self.storage_path):
            self.load()
            if not os.path.exists(self.tgat_binary_path):
                self.save()
        else:
            self._seed_default_memory()
            self.save()

    def _clean_id(self, label: str) -> str:
        return label.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")

    def get_or_create_node(self, label: str, category: str = "concept", attributes: Optional[Dict[str, Any]] = None) -> BitstreamNode:
        node_id = self._clean_id(label)
        if node_id in self.nodes:
            return self.nodes[node_id]

        tokens = list(self.tokenizer.encode(label))
        node = BitstreamNode(
            id=node_id,
            label=label.strip(),
            tokens=tokens,
            category=category,
            attributes=attributes or {}
        )
        self.nodes[node_id] = node

        # Sync to Native TGAT CSR engine
        self.tgat.add_node(label.strip(), tokens)

        # Index tokens
        for t in tokens:
            if t not in self.token_to_nodes:
                self.token_to_nodes[t] = set()
            self.token_to_nodes[t].add(node_id)

        return node

    def add_triplet(self, source_label: str, relation_text: str, target_label: str, category: str = "concept", weight: float = 1.0, timestamp: Optional[float] = None) -> Tuple[BitstreamNode, BitstreamNode]:
        """Adds a directional (or bidirectional) fact into the bitstream graph and Native TGAT engine."""
        src_node = self.get_or_create_node(source_label, category)
        tgt_node = self.get_or_create_node(target_label, category)

        rel_tokens = list(self.tokenizer.encode(relation_text))
        ts = timestamp if timestamp is not None else time.time()

        # Check if edge already exists
        for e in src_node.edges:
            if e.target_id == tgt_node.id and e.relation_text == relation_text:
                e.weight = weight
                e.created_at = ts
                return src_node, tgt_node

        edge = BitstreamEdge(
            target_id=tgt_node.id,
            relation_text=relation_text.strip(),
            relation_tokens=rel_tokens,
            weight=weight,
            created_at=ts
        )
        src_node.edges.append(edge)

        # Sync into Native TGAT CSR Engine
        src_idx = self.tgat.node_label_to_idx.get(source_label.strip(), 0)
        tgt_idx = self.tgat.node_label_to_idx.get(target_label.strip(), 0)
        self.tgat.add_edge(src_idx, tgt_idx, rel_tokens, timestamp=ts, weight=weight)

        return src_node, tgt_node

    def recall_context(self, prompt: str, max_triplets: int = 8, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Directly intersects query token-IDs against the Bitstream Graph Index.
        Returns the most relevant multi-hop subgraphs in under 1 millisecond.
        """
        query_tokens = list(self.tokenizer.encode(prompt))
        matched_node_ids: Dict[str, int] = {}

        # 1. Direct Token Matching
        for t in query_tokens:
            if t in self.token_to_nodes:
                for n_id in self.token_to_nodes[t]:
                    matched_node_ids[n_id] = matched_node_ids.get(n_id, 0) + 1

        if not matched_node_ids:
            return []

        # Sort seed nodes by token match count
        sorted_seeds = sorted(matched_node_ids.items(), key=lambda x: x[1], reverse=True)[:5]
        visited_triplets: List[Dict[str, Any]] = []
        seen_triplets_set: Set[Tuple[str, str, str]] = set()

        # 2. Multi-Hop Graph Traversal
        for seed_id, match_score in sorted_seeds:
            current_level = [seed_id]
            for depth in range(max_depth):
                next_level = []
                for n_id in current_level:
                    if n_id not in self.nodes:
                        continue
                    node = self.nodes[n_id]
                    for edge in node.edges:
                        tgt_id = edge.target_id
                        tgt_node = self.nodes.get(tgt_id)
                        if not tgt_node:
                            continue

                        triplet_key = (node.id, edge.relation_text, tgt_id)
                        if triplet_key not in seen_triplets_set:
                            seen_triplets_set.add(triplet_key)
                            visited_triplets.append({
                                "source": node.label,
                                "source_category": node.category,
                                "relation": edge.relation_text,
                                "target": tgt_node.label,
                                "target_category": tgt_node.category,
                                "weight": edge.weight,
                                "relevance_score": match_score / (depth + 1)
                            })
                            if len(visited_triplets) >= max_triplets:
                                return visited_triplets
                            next_level.append(tgt_id)
                current_level = next_level

        return visited_triplets

    def format_memory_prompt(self, prompt: str) -> str:
        """Formats recalled graph memories for direct injection into the model context."""
        triplets = self.recall_context(prompt, max_triplets=6)
        if not triplets:
            return ""

        lines = ["### Langzeit-Gedächtnis & Fakten (Bitstream Graph):"]
        for t in triplets:
            lines.append(f"- **{t['source']}** --[{t['relation']}]--> **{t['target']}**")
        lines.append("")
        return "\n".join(lines) + "\n"

    def export_graph_json(self) -> Dict[str, Any]:
        """Exports the graph nodes and links for interactive visualization."""
        nodes_list = []
        links_list = []

        for n_id, n in self.nodes.items():
            nodes_list.append({
                "id": n.id,
                "label": n.label,
                "category": n.category,
                "edge_count": len(n.edges)
            })
            for e in n.edges:
                links_list.append({
                    "source": n.id,
                    "target": e.target_id,
                    "relation": e.relation_text,
                    "weight": e.weight
                })

        return {
            "total_nodes": len(nodes_list),
            "total_links": len(links_list),
            "nodes": nodes_list,
            "links": links_list
        }

    def _seed_default_memory(self):
        """Pre-populates the graph memory with real system and project facts."""
        facts = [
            ("Benjamin", "entwickelt", "7.45B Multi-Granular Bitstream MoE", "user"),
            ("7.45B Multi-Granular Bitstream MoE", "nutzt", "16-Bit Viterbi Bitstream Tokenizer", "tech"),
            ("7.45B Multi-Granular Bitstream MoE", "optimiert_durch", "GaLore Per-Layer SVD Optimizer", "tech"),
            ("7.45B Multi-Granular Bitstream MoE", "besitzt", "24 Schichten und 288 Sparse MoE Experten", "tech"),
            ("7.45B Multi-Granular Bitstream MoE", "läuft_auf", "NVIDIA GeForce RTX 3060 (6 GB VRAM)", "system"),
            ("Wissensbasis", "umfasst", "1.393 Shards und 696 Millionen Tokens", "tech"),
            ("Minecraft Experte", "beherrscht", "Minecraft 1.8 bis 1.21+ Voxel-Physik", "minecraft"),
            ("Minecraft Experte", "berechnet", "Instant-Mining und 3D DDA-Raycasts", "minecraft"),
            ("STEM & Mathematik", "enthält", "Alle 7 Clay Millennium-Probleme", "math"),
            ("STEM & Mathematik", "beinhaltet", "P vs NP und Cook-Levin Komplexitätstheorie", "math"),
            ("Web Studio", "verfügt_über", "Universal AI Chat Studio mit Canvas", "tech"),
            ("Universal AI Chat Studio", "bietet", "4 Modi: Reasoning, Code, Dialog, Polymath", "tech"),
            ("Gedächtnis-Architektur", "basiert_auf", "Nativem 16-Bit Bitstream Graph & MCP", "tech"),
            ("MCP Memory Server", "implementiert", "JSON-RPC 2.0 Schnittstelle", "tech")
        ]

        for src, rel, tgt, cat in facts:
            self.add_triplet(src, rel, tgt, category=cat)

    def save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            "version": "1.0-bitstream",
            "nodes": {n_id: n.to_dict() for n_id, n in self.nodes.items()}
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Save Native TGAT Binary
        try:
            self.tgat.save_binary(self.tgat_binary_path)
        except Exception:
            pass

    def load(self):
        if not os.path.exists(self.storage_path):
            return
        with open(self.storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.nodes = {}
        self.token_to_nodes = {}
        self.tgat = NativeBitstreamTGAT(bochner_dim=64)

        for n_id, n_data in data.get("nodes", {}).items():
            node = BitstreamNode.from_dict(n_data)
            self.nodes[n_id] = node
            self.tgat.add_node(node.label, node.tokens)
            for t in node.tokens:
                if t not in self.token_to_nodes:
                    self.token_to_nodes[t] = set()
                self.token_to_nodes[t].add(n_id)

        # Sync edges into TGAT
        for n_id, node in self.nodes.items():
            src_idx = self.tgat.node_label_to_idx.get(node.label, 0)
            for e in node.edges:
                tgt_node = self.nodes.get(e.target_id)
                if tgt_node:
                    tgt_idx = self.tgat.node_label_to_idx.get(tgt_node.label, 0)
                    self.tgat.add_edge(src_idx, tgt_idx, e.relation_tokens, timestamp=e.created_at, weight=e.weight)


if __name__ == "__main__":
    mem = BitstreamGraphMemory()
    print("🧠 Native Bitstream Graph Memory initialisiert!")
    print(f"📊 Geladene Knoten: {len(mem.nodes)}")
    
    test_query = "Erzähle mir etwas über Minecraft und GaLore"
    recalled = mem.recall_context(test_query)
    print(f"\n🔍 Abfrage: '{test_query}'")
    print(f"⚡ Gefundene Fakten im Bitstream-Graph ({len(recalled)}):")
    for r in recalled:
        print(f"  • {r['source']} --[{r['relation']}]--> {r['target']} (Score: {r['relevance_score']:.2f})")
    
    print("\n📝 Formatierter System-Kontext:")
    print(mem.format_memory_prompt(test_query))
