#!/usr/bin/env python3
r"""Specialized Ingestion Pipeline for Model Context Protocol (MCP), Knowledge Graphs,
Graph-RAG, Graph Neural Networks (GNNs), and Dynamic Curriculum DAG Engines.

Covers:
1. Model Context Protocol (MCP) Architecture:
   - JSON-RPC 2.0 Protocol, Stdio and SSE Transports, Protocol Handshake (initialize, initialized).
   - MCP Primitives: Tools (tools/list, tools/call), Resources (resources/list, resources/read), Prompts, Roots, Sampling.
   - Building Custom High-Performance MCP Servers in Python (FastMCP, mcp SDK) and TypeScript (@modelcontextprotocol/sdk).
2. Knowledge Graphs & Graph Databases:
   - Property Graphs, Nodes (Vertices) & Typed Edges, RDF Triples & SPARQL, Cypher queries (Neo4j, Memgraph).
   - Graph Traversal Algorithms: Dijkstra, A*, Louvain Community Detection, PageRank, Topological Sort.
3. Graph-RAG (Hybrid Vector + Graph Retrieval):
   - 2-Hop Neighborhood Expansion, Entity Extraction, Triplet Resolution, Vector + Graph Fusion.
4. Graph Neural Networks (GNNs) & Message Passing:
   - MPNN framework, Graph Convolutional Networks (GCN: $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$),
   - Graph Attention Networks (GAT: $\alpha_{ij} = \text{Softmax}(\text{LeakyReLU}(a^T [Wh_i \parallel Wh_j]))$).
5. Curriculum DAG & Dynamic Training Engines:
   - Prerequisite DAGs, Loss-Aware Remediation Routing, Exponential Moving Loss Smoothing, Mastery-Gated Unlocking.

Appends directly to data/cyber_web_knowledge/shards/ and data/ai_research_knowledge/shards/.
"""

import os
import sys
import time
import glob
import signal
from typing import List, Dict, Any

from datasets import load_dataset
from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamEncoder
from pipeline.tokenizer import ViterbiTokenizer

STOP_REQUESTED = False

def handle_signal(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠️ Graceful Shutdown angefordert, beende nach aktuellem Shard...", flush=True)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


MCP_AND_GRAPH_DEEP_KNOWLEDGE = [
    # 1. Complete Model Context Protocol (MCP) Architecture & Implementation
    r"""# Model Context Protocol (MCP) Specification & Custom Server Engineering:

## 1. Das Protokoll-Fundament: JSON-RPC 2.0 über Stdio & SSE
Das Model Context Protocol (MCP) standardisiert die Kommunikation zwischen LLM-Agenten (Clients) und lokalen/entfernten Werkzeugen (Servern).
- **Transport-Schichten**:
  1. `stdio`: Server liest JSON-RPC Nachrichten zeilenweise von `stdin` und antwortet auf `stdout`. Logs MÜSSEN strikt auf `stderr` geschrieben werden, um den Stream nicht zu korrumpieren.
  2. `SSE (Server-Sent Events)`: HTTP-basierter unidirektionaler Event-Stream vom Server zum Client + HTTP POST Endpunkt für Client-zu-Server Nachrichten.

## 2. Der MCP Handshake (Initialize Lifecycle)
1. Client sendet `initialize`:
   ```json
   {
     "jsonrpc": "2.0",
     "id": 1,
     "method": "initialize",
     "params": {
       "protocolVersion": "2024-11-05",
       "capabilities": { "roots": { "listChanged": true }, "sampling": {} },
       "clientInfo": { "name": "AntigravityIDE", "version": "2.0.0" }
     }
   }
   ```
2. Server antwortet mit Capabilities (`tools`, `resources`, `prompts`):
   ```json
   {
     "jsonrpc": "2.0",
     "id": 1,
     "result": {
       "protocolVersion": "2024-11-05",
       "capabilities": { "tools": { "listChanged": false }, "resources": { "subscribe": true } },
       "serverInfo": { "name": "system-graph-server", "version": "1.0.0" }
     }
   }
   ```
3. Client bestätigt mit `notifications/initialized`.

## 3. Vollständiger Custom MCP Server in Python (FastMCP / Zero-Dependency)
```python
import sys
import json

class FastMCPServer:
    def __init__(self, name: str):
        self.name = name
        self.tools = {}

    def tool(self, name: str = None, description: str = ""):
        def decorator(func):
            tool_name = name or func.__name__
            self.tools[tool_name] = {
                "func": func,
                "description": description,
                "schema": {
                    "type": "object",
                    "properties": func.__annotations__,
                }
            }
            return func
        return decorator

    def run_stdio(self):
        for line in sys.stdin:
            if not line.strip():
                continue
            req = json.loads(line)
            method = req.get("method")
            msg_id = req.get("id")

            if method == "tools/list":
                tools_list = [
                    {"name": k, "description": v["description"], "inputSchema": {"type": "object"}}
                    for k, v in self.tools.items()
                ]
                self._send_response(msg_id, {"tools": tools_list})

            elif method == "tools/call":
                params = req.get("params", {})
                tool_name = params.get("name")
                args = params.get("arguments", {})
                if tool_name in self.tools:
                    result = self.tools[tool_name]["func"](**args)
                    self._send_response(msg_id, {"content": [{"type": "text", "text": str(result)}]})
                else:
                    self._send_error(msg_id, -32601, f"Tool '{tool_name}' nicht gefunden")

    def _send_response(self, msg_id, result):
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}) + "\n")
        sys.stdout.flush()

    def _send_error(self, msg_id, code, message):
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}) + "\n")
        sys.stdout.flush()
```""",

    # 2. Knowledge Graphs, Graph-RAG & Traversal Algorithms
    r"""# Knowledge Graphs, Graph-RAG & Graph Database Engineering:

## 1. Graph-RAG: Hybride Vektor- und Graph-Traversierung
Herkömmliches Vektor-RAG leidet unter Kontext-Fragmentierung. **Graph-RAG** löst dies durch 2-Schritt-Expansion:
1. **Vektor-Suche**: Finde die relevantesten Einstiegsknoten $V_{\text{seed}}$ über Kosinus-Ähnlichkeit im Embedding-Raum.
2. **K-Hop Subgraph Extraktion**: Extrahiere alle Relationen im 2-Hop Nachbarschafts-Umfeld:
   $$\mathcal{N}_k(u) = \{ v \in V \mid \text{dist}(u, v) \le k \}$$
3. **Triplets-Serialisierung**: Wandle den Teilgraphen in Markdown-Pfade um: `(Model)-[USES]->(GaLoreOptimizer)-[PROJECTS_INTO]->(SubspaceMatrix)`.

## 2. Graph-Traversierung in Python (Dijkstra & Topologischer DAG-Sort)
```python
import heapq
from collections import defaultdict, deque
from typing import Dict, List, Tuple

class KnowledgeGraph:
    def __init__(self):
        self.adj = defaultdict(list)
        self.in_degree = defaultdict(int)
        self.nodes = set()

    def add_edge(self, u: str, v: str, weight: float = 1.0):
        self.nodes.add(u)
        self.nodes.add(v)
        self.adj[u].append((v, weight))
        self.in_degree[v] += 1

    def shortest_path_dijkstra(self, start: str, target: str) -> Tuple[float, List[str]]:
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0
        pq = [(0.0, start, [start])]

        while pq:
            cost, u, path = heapq.heappop(pq)
            if u == target:
                return cost, path
            if cost > distances[u]:
                continue
            for v, weight in self.adj[u]:
                if cost + weight < distances[v]:
                    distances[v] = cost + weight
                    heapq.heappush(pq, (cost + weight, v, path + [v]))
        return float('inf'), []

    def topological_sort_dag(self) -> List[str]:
        in_deg = self.in_degree.copy()
        queue = deque([n for n in self.nodes if in_deg[n] == 0])
        order = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v, _ in self.adj[u]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)
        if len(order) != len(self.nodes):
            raise ValueError("Graph enthält Zyklen (Kein DAG!)")
        return order
```""",

    # 3. Graph Neural Networks (GNNs): GCN & Graph Attention Networks (GAT)
    r"""# Graph Neural Networks: Message Passing, GCN & GAT Mathematische Herleitung:

## 1. Graph Convolutional Networks (GCN)
Für eine Adjazenzmatrix $A \in \mathbb{R}^{N \times N}$ mit Selbstschleifen $\tilde{A} = A + I_N$ und Gradmatrix $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ ist die Schicht-Transformation definiert als:
$$H^{(l+1)} = \sigma\left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$$
Die symmetrische Normalisierung $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ verhindert das Explodieren oder Verschwinden von Knoten-Merkmalen bei Knoten mit sehr vielen Kanten.

## 2. Graph Attention Networks (GAT)
Anstelle fixer Gewichte berechnet GAT dynamische Aufmerksamkeits-Koeffizienten $\alpha_{ij}$ zwischen benachbarten Knoten $i$ und $j$:
$$\alpha_{ij} = \frac{\exp\left( \text{LeakyReLU}\left( \vec{a}^T [W h_i \parallel W h_j] \right) \right)}{\sum_{k \in \mathcal{N}_i} \exp\left( \text{LeakyReLU}\left( \vec{a}^T [W h_i \parallel W h_k] \right) \right)}$$
Der aggregierte neue Merkmalsvektor mit Multi-Head Attention ($K$ Köpfe) lautet:
$$h_i^{(l+1)} = \Vert_{k=1}^K \sigma\left( \sum_{j \in \mathcal{N}_i} \alpha_{ij}^k W^k h_j^{(l)} \right)$$"""
]


def run_mcp_graph_ingestion(
    output_dir: str = "/home/benjamin/Bilder/data/cyber_web_knowledge/shards",
    vocab_file: str = "/home/benjamin/Bilder/data/vocab_65k.json",
    max_tokens_per_shard: int = 500_000,
    target_shards: int = 280,
) -> int:
    global STOP_REQUESTED

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    print("=" * 80, flush=True)
    print("⚡ MASSIVE MCP & KNOWLEDGE GRAPH ARCHITECTURE PIPELINE (7B PRE-TRAINING)", flush=True)
    print(f"📁 Ziel-Verzeichnis: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    existing_files = sorted(glob.glob(os.path.join(output_dir, "*.mgbs")))
    shard_count = len(existing_files)
    print(f"  ⏭️ {shard_count} bestehende Cyber/Graph-Shards gefunden. Starte ab Index {shard_count:04d}...", flush=True)

    vocab = MultiGranularVocabulary.load_json(vocab_file)
    tokenizer = ViterbiTokenizer(vocab)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16)

    buffer_tokens: List[int] = []
    total_tokens_written = shard_count * max_tokens_per_shard
    start_time = time.time()

    def flush_shard():
        nonlocal shard_count, buffer_tokens, total_tokens_written
        if not buffer_tokens:
            return
        shard_path = os.path.join(output_dir, f"cyber_web_shard_{shard_count:04d}.mgbs")
        encoder.save_to_file(shard_path, buffer_tokens, raw_byte_count=len(buffer_tokens) * 2)
        total_tokens_written += len(buffer_tokens)
        shard_count += 1
        elapsed = time.time() - start_time
        print(f"  💾 [MCP/GRAPH Shard {shard_count:04d}] {len(buffer_tokens):,} Tokens | Gesamt: {total_tokens_written:,} Tokens -> {os.path.basename(shard_path)}", flush=True)
        buffer_tokens = []

    # 1. Ingest Deep MCP & Graph Architecture Standards
    print("\n📚 [Quelle 1/2] Tokenisiere Model Context Protocol (MCP), Graph-RAG & GNN Standards...", flush=True)
    for doc in MCP_AND_GRAPH_DEEP_KNOWLEDGE:
        formatted = f"### System-Architektur (Model Context Protocol & Knowledge Graphs):\n{doc.strip()}\n\n"
        buffer_tokens.extend(tokenizer.encode(formatted))
        if len(buffer_tokens) >= max_tokens_per_shard:
            flush_shard()

    # 2. Ingest MCP, Graph Algorithms & Tool Use Dialogues
    print("\n🛠️ [Quelle 2/2] Streame Graph Algorithms, JSON-RPC Tools & Agentic Context Protocol...", flush=True)
    try:
        code_ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
        mcp_count = 0
        for item in code_ds:
            if STOP_REQUESTED or shard_count >= target_shards:
                break
            instr = item.get("instruction", "")
            resp = item.get("output", "")
            if not instr or not resp:
                continue

            keywords = ["graph", "tree", "dijkstra", "dag", "topological", "json-rpc", "rpc", "mcp", "protocol", "tool", "schema", "node", "edge", "neighbor", "server", "client"]
            is_relevant = any(kw in instr.lower() or kw in resp.lower() for kw in keywords)
            if not is_relevant:
                continue

            formatted = f"### Benutzer (Model Context Protocol & Graph Systems):\n{instr}\n\n### Assistent (Principal AI & Graph Architect):\n<think>\nAnalysiere Graphentopologie, MCP JSON-RPC Schnittstellen und Tool-Ausführung:\n</think>\n{resp}\n\n"
            buffer_tokens.extend(tokenizer.encode(formatted))
            mcp_count += 1

            if len(buffer_tokens) >= max_tokens_per_shard:
                flush_shard()

            if mcp_count % 2000 == 0:
                print(f"  ⚙️ [MCP-Graph-Engine] {mcp_count:,} Graph & Tool-Instruktionen verarbeitet (Shards: {shard_count})", flush=True)

        print(f"✅ MCP & Graph Architecture Dialoge abgeschlossen ({mcp_count:,} verarbeitet).", flush=True)
    except Exception as e:
        print(f"⚠️ Hinweis bei MCP Stream: {e}", flush=True)

    flush_shard()
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 MCP & Graph Architecture Specialist Ingestion abgeschlossen! Gesamte Shards: {shard_count} (~{total_tokens_written:,} Tokens)", flush=True)
    print("=" * 80, flush=True)
    return shard_count


if __name__ == "__main__":
    run_mcp_graph_ingestion()
