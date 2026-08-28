"""System 2 Reasoning & Thinking Engine (DeepSeek-R1 / OpenAI o1 Style).

Enables deep multi-step cognitive reasoning chains within <think>...</think> tags,
self-reflection loops, and mathematical verification before outputting final answers.
"""

import re
from typing import Dict, List, Tuple
import torch


class System2ReasoningChain:
    """Manages multi-turn thinking, intermediate verification, and final answer extraction."""

    THINK_START = "<think>"
    THINK_END = "</think>"

    @staticmethod
    def format_reasoning_prompt(user_prompt: str) -> str:
        """Formats the input prompt to trigger deep cognitive reasoning."""
        return (
            f"### Benutzer:\n{user_prompt}\n\n"
            f"### Assistent:\n"
            f"{System2ReasoningChain.THINK_START}\n"
            f"1. Problem-Zerlegung: Was ist das genaue Ziel?\n"
            f"2. Schritt-für-Schritt Herleitung:\n"
        )

    @staticmethod
    def parse_thinking_output(full_text: str) -> Dict[str, str]:
        """Separates the internal thought process from the final user-facing response."""
        think_pattern = r"<think>(.*?)</think>"
        match = re.search(think_pattern, full_text, flags=re.DOTALL)

        if match:
            thinking_trace = match.group(1).strip()
            # Everything after </think> is the final clean answer
            final_answer = full_text.split("</think>")[-1].strip()
        else:
            thinking_trace = ""
            final_answer = full_text.strip()

        return {
            "thinking_trace": thinking_trace,
            "final_answer": final_answer,
            "has_thinking_trace": len(thinking_trace) > 0,
        }

    @staticmethod
    def verify_mathematical_step(expression: str) -> Tuple[bool, str]:
        """Real Python evaluation of mathematical expressions in thought chains (Zero Mocks)."""
        clean_expr = expression.replace("^", "**").strip()
        # Whitelist safe arithmetic chars
        if not re.match(r"^[0-9+\-*/().\s**eE]+$", clean_expr):
            return False, "Ungültiger Rechenausdruck"

        try:
            result = eval(clean_expr, {"__builtins__": None}, {})
            return True, str(result)
        except Exception as e:
            return False, str(e)
