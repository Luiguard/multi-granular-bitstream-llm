#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Server for Live Web-Search & Browsing.
Exposes web_search and web_browse_page tools over JSON-RPC 2.0.
"""

import json
import os
import sys

sys.path.insert(0, "/home/benjamin/Bilder")

from pipeline.web_surfer import WebSurfer


class WebSearchMCPServer:
    def __init__(self):
        self.surfer = WebSurfer()

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
                                "name": "web_search",
                                "description": "Führt eine echte Live-Websuche im Internet durch und liefert aktuelle URLs, Titel und Zusammenfassungen.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "Suchbegriff oder Frage"},
                                        "max_results": {"type": "integer", "default": 5, "description": "Maximale Anzahl an Suchergebnissen"}
                                    },
                                    "required": ["query"]
                                }
                            },
                            {
                                "name": "web_browse_page",
                                "description": "Lädt eine spezifische Website-URL herunter und extrahiert den sauberen Artikeltext (Reader Mode).",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "url": {"type": "string", "description": "Vollständige HTTP/HTTPS URL"}
                                    },
                                    "required": ["url"]
                                }
                            }
                        ]
                    }
                })

            elif method == "tools/call":
                tool_name = params.get("name")
                args = params.get("arguments", {})

                if tool_name == "web_search":
                    query = args["query"]
                    max_results = int(args.get("max_results", 5))
                    results = self.surfer.live_search(query, max_results=max_results)
                    formatted_ctx = self.surfer.format_web_context(query, max_results=max_results)
                    return json.dumps({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "results": results,
                            "formatted_context": formatted_ctx,
                            "content": [{"type": "text", "text": formatted_ctx or "Keine Suchergebnisse gefunden."}]
                        }
                    })

                elif tool_name == "web_browse_page":
                    url = args["url"]
                    page_data = self.surfer.browse_and_extract_page(url)
                    return json.dumps({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": page_data
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
                "error": {"code": -32603, "message": f"Interner Web-Search MCP Fehler: {str(e)}"}
            })

    def run_stdio(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            response = self.handle_request(line)
            sys.stdout.write(response + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    server = WebSearchMCPServer()
    server.run_stdio()
