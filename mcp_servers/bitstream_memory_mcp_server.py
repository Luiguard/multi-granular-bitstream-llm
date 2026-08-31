#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Server for Bitstream Graph Memory.
Standard JSON-RPC 2.0 interface exposing memory graph tools over stdio/HTTP.
"""

import json
import os
import sys

sys.path.insert(0, "/home/benjamin/Bilder")

from pipeline.bitstream_graph_memory import BitstreamGraphMemory


class BitstreamMemoryMCPServer:
    def __init__(self):
        self.memory = BitstreamGraphMemory()

    def handle_request(self, request_json: str) -> str:
        try:
            req = json.loads(request_json)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method == "tools/list":
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "graph_memorize",
                                "description": "Speichert ein neues Fakten-Tripel direkt als 16-Bit Bitstream im Wissensgraphen.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "source": {"type": "string", "description": "Subjekt / Start-Knoten"},
                                        "relation": {"type": "string", "description": "Prädikat / Beziehung"},
                                        "target": {"type": "string", "description": "Objekt / Ziel-Knoten"},
                                        "category": {"type": "string", "default": "concept", "description": "Kategorie (user, project, tech, math, minecraft)"}
                                    },
                                    "required": ["source", "relation", "target"]
                                }
                            },
                            {
                                "name": "graph_query",
                                "description": "Sucht verknüpfte Wissens-Knoten direkt über die Viterbi Token-Schnittmenge in unter 1 ms.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "Suchtext oder Frage"},
                                        "max_triplets": {"type": "integer", "default": 8}
                                    },
                                    "required": ["query"]
                                }
                            },
                            {
                                "name": "graph_get_summary",
                                "description": "Liefert die Gesamtstatistik der Knoten und Kanten im Gedächtnis.",
                                "inputSchema": {"type": "object", "properties": {}}
                            },
                            {
                                "name": "graph_export_json",
                                "description": "Exportiert den gesamten Wissensgraphen für visuelle Canvas-Darstellung.",
                                "inputSchema": {"type": "object", "properties": {}}
                            }
                        ]
                    }
                })

            elif method == "tools/call":
                tool_name = params.get("name")
                args = params.get("arguments", {})

                if tool_name == "graph_memorize":
                    src = args["source"]
                    rel = args["relation"]
                    tgt = args["target"]
                    cat = args.get("category", "concept")
                    self.memory.add_triplet(src, rel, tgt, category=cat)
                    self.memory.save()
                    return json.dumps({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": f"✅ Fakt memorisiert: ({src}) --[{rel}]--> ({tgt})"}]
                        }
                    })

                elif tool_name == "graph_query":
                    query = args["query"]
                    max_t = int(args.get("max_triplets", 8))
                    recalled = self.memory.recall_context(query, max_triplets=max_t)
                    formatted_txt = self.memory.format_memory_prompt(query)
                    return json.dumps({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "triplets": recalled,
                            "formatted_context": formatted_txt,
                            "content": [{"type": "text", "text": formatted_txt or "Keine direkten Erinnerungen gefunden."}]
                        }
                    })

                elif tool_name == "graph_get_summary":
                    summary = {
                        "total_nodes": len(self.memory.nodes),
                        "total_edges": sum(len(n.edges) for n in self.memory.nodes.values()),
                        "categories": list(set(n.category for n in self.memory.nodes.values()))
                    }
                    return json.dumps({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": summary
                    })

                elif tool_name == "graph_export_json":
                    return json.dumps({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": self.memory.export_graph_json()
                    })

                else:
                    return json.dumps({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": f"Tool '{tool_name}' nicht gefunden."}
                    })

            else:
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32600, "message": f"Methode '{method}' nicht unterstützt."}
                })

        except Exception as e:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": f"Interner MCP Fehler: {str(e)}"}
            })

    def run_stdio(self):
        """Standard MCP Stdio Loop."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            response = self.handle_request(line)
            sys.stdout.write(response + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    server = BitstreamMemoryMCPServer()
    server.run_stdio()
