"""Constitutional Safety Guardrails & Alignment Engine for Multi-Granular Bitstream LLMs.

Enforces:
1. Epistemic Honesty (Strictly forbids hallucinations, fake citations, and mock data).
2. Ethical Safety Boundaries (Blocks malware, malicious exploits, and dangerous instructions).
3. Clean Formatting (Clean GitHub markdown, plain-text math, no broken artifacts).
"""

import re
from typing import Tuple, List

# Banned dangerous intents
BLOCKED_PATTERNS = [
    r"\b(create|write|generate|build|code|make)\b(?:\s+\w+){0,3}\s+\b(malware|ransomware|keylogger|rootkit|ddos|exploit|virus)\b",
    r"\b(how\s+to\s+make|synthesize|build)\b(?:\s+\w+){0,3}\s+\b(explosives|weapons|nerve\s+agent|bomb)\b",
]

# Epistemic Honesty Principles
EPISTEMIC_SYSTEM_PROMPT = """Du bist ein hochentwickelter, epistemisch ehrlicher und präziser KI-Assistent, angetrieben von einer verlustfreien Multi-Granularitäts-Bitstream-Engine.

### Deine Verhaltensregeln & Leitplanken:
1. **Wahrhaftigkeit & Echte Daten:** Erfinde niemals Fakten, Pseudo-Quellen, gefälschte Zahlen oder Scheinergebnisse. Wenn du eine Information nicht sicher weißt, sage klar und ehrlich: "Dazu liegen mir keine verifizierten Daten vor."
2. **Keine Platzhalter & Mocks:** Verwende in Code- und Datenantworten immer reale, funktionierende und vollständige Umsetzungen (keine `// mock data` oder `TODO`-Lücken).
3. **Sicherheit & Integrität:** Verweigere die Erstellung von bösartiger Schadsoftware, Exploits oder gefährlichen Anleitungen. Biete stattdessen konstruktive Sicherheitsanalysen und Best Practices an.
4. **Klare Formatierung:** Nutze übersichtliches Markdown, saubere Code-Blöcke und lesbare, natürliche Formatierung ohne störende Syntax-Artefakte.
"""


class SafetyGuardrails:
    """Enforces prompt filtering and response sanitization for inference."""

    @staticmethod
    def inspect_prompt(prompt: str) -> Tuple[bool, str]:
        """Checks if the user prompt violates safety boundaries."""
        prompt_lower = prompt.lower()
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, prompt_lower):
                return False, "Ich kann diese Anfrage nicht bearbeiten, da sie gegen Sicherheitsrichtlinien (Erstellung von Schadsoftware oder gefährlichen Substanzen) verstößt."
        return True, ""

    @staticmethod
    def format_system_prompt() -> str:
        """Returns the constitutional system prompt for Ollama and API serving."""
        return EPISTEMIC_SYSTEM_PROMPT.strip()

    @staticmethod
    def sanitize_output(response_text: str) -> str:
        """Ensures formatting consistency (e.g. cleans raw LaTeX dollar signs into clean text)."""
        # Normalizes any residual LaTeX dollar signs to clean plain text
        cleaned = re.sub(r"\$([^\$]+)\$", r"\1", response_text)
        return cleaned.strip()
