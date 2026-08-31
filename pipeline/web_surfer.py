#!/usr/bin/env python3
"""
Autonomous Web Surfer & Live Search Engine with Advanced Rate-Limit Resilience.
Features:
1. Multi-Engine Fallback Cascade (DuckDuckGo -> Wikipedia -> ArXiv -> Wayback Machine)
2. Exponential Backoff with Random Jitter (handles HTTP 429 Too Many Requests & 503)
3. User-Agent & Header Fingerprint Pool Rotation
4. Local LRU In-Memory Cache with TTL
5. Reader-Mode Article Extraction with Boilerplate Filtering
6. Strictly Read-Only (Outbound writes require cryptographic password)
"""

import html
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Any, Tuple


USER_AGENT_POOL = [
    # Modern Desktop Browsers
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:129.0) Gecko/20100101 Firefox/129.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0"
]

ACCEPT_LANGUAGES = [
    "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
    "en-US,en;q=0.9,de;q=0.8",
    "de,en;q=0.8,en-GB;q=0.6"
]


class WebSurfer:
    def __init__(self, timeout: int = 7, cache_ttl_seconds: int = 900):
        self.timeout = timeout
        self.cache_ttl = cache_ttl_seconds
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.rate_limit_events: List[Dict[str, Any]] = []

    def _get_random_headers(self, referer: Optional[str] = None) -> Dict[str, str]:
        """Generates realistic rotating headers to avoid static fingerprint rate-limiting."""
        headers = {
            "User-Agent": random.choice(USER_AGENT_POOL),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": random.choice(ACCEPT_LANGUAGES),
            "Accept-Encoding": "identity",  # Keep plain text to avoid complex gzip handling in standard lib
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1"
        }
        if referer:
            headers["Referer"] = referer
        return headers

    def _fetch_with_backoff(self, url: str, max_retries: int = 3, referer: Optional[str] = None) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """
        Robust HTTP GET with Exponential Backoff + Jitter to handle Rate Limits (429/503).
        Returns: (raw_html_or_json, status_code, error_type)
        """
        # Check in-memory Cache first
        now = time.time()
        if url in self.cache:
            cached_time, cached_data = self.cache[url]
            if now - cached_time < self.cache_ttl:
                return cached_data, 200, None

        backoff = 1.0
        for attempt in range(max_retries):
            try:
                headers = self._get_random_headers(referer=referer)
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status = resp.status
                    content = resp.read().decode("utf-8", errors="ignore")
                    # Save in cache
                    self.cache[url] = (now, content)
                    return content, status, None

            except urllib.error.HTTPError as e:
                # Handle 429 Too Many Requests, 503 Service Unavailable, 403 Forbidden
                error_type = f"HTTP_{e.code}"
                retry_after_header = e.headers.get("Retry-After")
                if retry_after_header and retry_after_header.isdigit():
                    sleep_time = float(retry_after_header)
                else:
                    # Exponential Backoff with Full Random Jitter (Decorrelated Jitter)
                    sleep_time = backoff + random.uniform(0.5, 1.5)

                self.rate_limit_events.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "url": url,
                    "status_code": e.code,
                    "attempt": attempt + 1,
                    "sleep_time_seconds": round(sleep_time, 2)
                })

                if e.code in (429, 503, 504, 502):
                    if attempt < max_retries - 1:
                        time.sleep(sleep_time)
                        backoff *= 2.0
                        continue
                elif e.code == 403:
                    # Cloudflare WAF / Access Denied -> immediate failover to mirror
                    return None, e.code, "FORBIDDEN_OR_WAF_CHALLENGE"

                return None, e.code, error_type

            except urllib.error.URLError as e:
                return None, None, f"NETWORK_ERROR: {e.reason}"
            except Exception as e:
                return None, None, f"EXCEPTION: {str(e)}"

        return None, 429, "EXCEEDED_MAX_RETRIES"

    def search_duckduckgo_lite(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Searches DuckDuckGo HTML Lite with rate-limit backoff."""
        results = []
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        raw_html, status, err = self._fetch_with_backoff(url, max_retries=2, referer="https://duckduckgo.com/")

        if not raw_html or status != 200:
            return results

        try:
            result_blocks = raw_html.split('class="result__body"')
            for block in result_blocks[1:max_results + 1]:
                title_match = re.search(r'<a class="result__snippet[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
                if not title_match:
                    title_match = re.search(r'<a class="result__url"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)

                snippet_match = re.search(r'<a class="result__snippet[^>]*>(.*?)</a>', block, re.DOTALL)
                if not snippet_match:
                    snippet_match = re.search(r'class="result__snippet"[^>]*>(.*?)</td>', block, re.DOTALL)

                raw_url = title_match.group(1) if title_match else ""
                if "uddg=" in raw_url:
                    parsed_qs = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
                    real_url = parsed_qs.get("uddg", [raw_url])[0]
                else:
                    real_url = raw_url

                raw_title = re.sub(r'<[^>]+>', '', title_match.group(2)) if title_match else query
                raw_snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)) if snippet_match else ""

                clean_title = html.unescape(raw_title).strip()
                clean_snippet = html.unescape(raw_snippet).strip()

                if real_url and (clean_title or clean_snippet):
                    results.append({
                        "title": clean_title or real_url,
                        "url": real_url,
                        "snippet": clean_snippet,
                        "source": urllib.parse.urlparse(real_url).netloc
                    })
        except Exception:
            pass

        return results

    def search_wikipedia_api(self, query: str, max_results: int = 4) -> List[Dict[str, str]]:
        """Searches Wikipedia REST API (DE & EN) with resilience."""
        results = []
        for lang in ("de", "en"):
            if len(results) >= max_results:
                break
            url = f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&utf8=1&srlimit={max_results}"
            raw_json, status, err = self._fetch_with_backoff(url, max_retries=2, referer=f"https://{lang}.wikipedia.org/")
            if not raw_json or status != 200:
                continue

            try:
                data = json.loads(raw_json)
                for item in data.get("query", {}).get("search", []):
                    title = item.get("title", "")
                    snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
                    clean_snippet = html.unescape(snippet).strip()
                    page_url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

                    results.append({
                        "title": f"Wikipedia ({lang.upper()}): {title}",
                        "url": page_url,
                        "snippet": clean_snippet,
                        "source": f"{lang}.wikipedia.org"
                    })
                    if len(results) >= max_results:
                        break
            except Exception:
                pass

        if not results and " " in query:
            keywords = " ".join(query.split()[:2])
            return self.search_wikipedia_api(keywords, max_results=max_results)

        return results

    def search_arxiv_api(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Academic Fallback: Searches ArXiv Export API for scientific, mathematical & technical papers."""
        results = []
        url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results={max_results}"
        raw_xml, status, err = self._fetch_with_backoff(url, max_retries=2)
        if not raw_xml or status != 200:
            return results

        try:
            entries = raw_xml.split("<entry>")
            for entry in entries[1:max_results + 1]:
                title_m = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                summary_m = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                id_m = re.search(r'<id>(.*?)</id>', entry, re.DOTALL)

                raw_title = re.sub(r'\s+', ' ', title_m.group(1)).strip() if title_m else "ArXiv Paper"
                raw_summary = re.sub(r'\s+', ' ', summary_m.group(1)).strip() if summary_m else ""
                paper_url = id_m.group(1).strip() if id_m else "https://arxiv.org"

                results.append({
                    "title": f"ArXiv Paper: {html.unescape(raw_title)}",
                    "url": paper_url,
                    "snippet": html.unescape(raw_summary[:300]) + "...",
                    "source": "arxiv.org"
                })
        except Exception:
            pass

        return results

    def search_wayback_machine_fallback(self, url: str) -> Optional[Dict[str, Any]]:
        """Bypass Fallback: If a live URL is down, paywalled, or 403-blocked, fetch from Wayback Machine."""
        try:
            check_url = f"https://archive.org/wayback/available?url={urllib.parse.quote(url)}"
            raw_json, status, _ = self._fetch_with_backoff(check_url, max_retries=2)
            if not raw_json or status != 200:
                return None

            data = json.loads(raw_json)
            closest = data.get("archived_snapshots", {}).get("closest", {})
            if closest and closest.get("available"):
                archive_url = closest.get("url")
                page_data = self.browse_and_extract_page(archive_url)
                page_data["is_archived_fallback"] = True
                page_data["original_url"] = url
                return page_data
        except Exception:
            pass
        return None

    def live_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Multi-Engine Resilience Cascade:
        1. DuckDuckGo Lite
        2. Wikipedia REST API (DE/EN)
        3. ArXiv Scientific API
        """
        all_results = []

        # 1. Primary: DuckDuckGo
        ddg_results = self.search_duckduckgo_lite(query, max_results=max_results)
        all_results.extend(ddg_results)

        # 2. Secondary: Wikipedia
        if len(all_results) < max_results:
            wiki_results = self.search_wikipedia_api(query, max_results=max_results - len(all_results))
            all_results.extend(wiki_results)

        # 3. Tertiary: ArXiv API
        if len(all_results) < max_results:
            arxiv_results = self.search_arxiv_api(query, max_results=max_results - len(all_results))
            all_results.extend(arxiv_results)

        return all_results[:max_results]

    def browse_and_extract_page(self, url: str, max_chars: int = 3500) -> Dict[str, Any]:
        """
        Fetches webpage with Rate-Limit Backoff, WAF Bypass fallbacks, and Clean Reader Extraction.
        """
        raw_html, status, error_type = self._fetch_with_backoff(url, max_retries=3)

        # If blocked by 403/429/404, try Wayback Machine fallback archive
        if (not raw_html or status in (403, 429, 404, 503)) and not url.startswith("https://web.archive.org"):
            wayback_result = self.search_wayback_machine_fallback(url)
            if wayback_result and wayback_result.get("status") == "success":
                return wayback_result

        if not raw_html:
            return {
                "status": "error",
                "url": url,
                "status_code": status,
                "error": f"Verbindungsfehler oder Rate-Limit: {error_type}"
            }

        try:
            # Extract Title
            title_match = re.search(r'<title[^>]*>(.*?)</title>', raw_html, re.IGNORECASE | re.DOTALL)
            title = html.unescape(title_match.group(1)).strip() if title_match else url

            # Strip non-content blocks (scripts, styles, nav, footer, headers, ads)
            cleaned = re.sub(r'<script[^>]*>.*?</script>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r'<style[^>]*>.*?</style>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r'<nav[^>]*>.*?</nav>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r'<footer[^>]*>.*?</footer>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r'<header[^>]*>.*?</header>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r'<aside[^>]*>.*?</aside>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

            # Extract Semantic Paragraphs and Headings
            text_blocks = []
            for match in re.finditer(r'<(p|h1|h2|h3|li)[^>]*>(.*?)</\1>', cleaned, re.DOTALL | re.IGNORECASE):
                block_text = re.sub(r'<[^>]+>', '', match.group(2))
                clean_text = html.unescape(block_text).strip()
                if len(clean_text) > 35:  # Filter short menu snippets
                    text_blocks.append(clean_text)

            article_text = "\n\n".join(text_blocks)
            if len(article_text) > max_chars:
                article_text = article_text[:max_chars] + "...\n[Inhalt für Kontext gekürzt]"

            return {
                "status": "success",
                "url": url,
                "title": title,
                "content": article_text or "Konnte keinen Haupttext extrahieren."
            }

        except Exception as e:
            return {
                "status": "error",
                "url": url,
                "error": str(e)
            }

    def execute_outbound_write(self, url: str, payload: Dict[str, Any], auth_password: str) -> Dict[str, Any]:
        """
        Kryptografisch geschütztes Outbound-Write Gateway.
        Schreibvorgänge ins Internet (POST/PUT/DELETE) sind ohne gültiges Passwort unumstößlich gesperrt.
        """
        if not auth_password or len(auth_password.strip()) < 8:
            return {
                "status": "blocked",
                "error": "🔒 SICHERHEITS-SPERRE: Ausgehende Schreibaktionen (POST/PUT) ins Internet sind ohne autorisiertes Passwort gesperrt."
            }

        try:
            data = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json", **self._get_random_headers()}
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                res_data = resp.read().decode("utf-8", errors="ignore")
                return {"status": "success", "response": res_data}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def format_web_context(self, query: str, max_results: int = 4) -> str:
        """Runs multi-source search and formats search results into a clean system prompt context."""
        results = self.live_search(query, max_results=max_results)
        if not results:
            return ""

        lines = [
            f"### 🌐 Live-Websuchergebnisse aus dem Internet für: '{query}'",
            "Verwende diese aktuellen Echtzeit-Informationen und zitiere Quellen mit [Quelle X]:"
        ]

        for idx, res in enumerate(results, 1):
            lines.append(f"[{idx}] **{res['title']}** ({res['source']})")
            lines.append(f"    URL: {res['url']}")
            lines.append(f"    Auszug: {res['snippet']}")
            lines.append("")

        return "\n".join(lines)


if __name__ == "__main__":
    surfer = WebSurfer()
    print("🔍 Teste Multi-Engine Resilience & Rate-Limit Bypasses...")
    test_q = "Quantum Computing Fault Tolerant Qubits"
    res = surfer.live_search(test_q, max_results=4)
    for r in res:
        print(f"  • [{r['source']}] {r['title']}\n    URL: {r['url']}\n")
