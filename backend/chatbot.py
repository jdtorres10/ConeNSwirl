"""LLM helpers for the Cone N' Swirl API (order recap only — no free-text RAG chat)."""

import json
import os

from langchain_anthropic import ChatAnthropic

_ORDER_SUMMARY_PROMPT = """You are Swirly, the Cone N' Swirl assistant. A customer just finished \
picking their build using the on-screen buttons (not free-typed chat). Their exact choices are \
in the JSON below — do not invent add-ons they did not pick.

Turn this into a friendly plain-English recap they can read at the window: what they're getting, \
in a natural order. The JSON uses order_type (cone_n_swirl, cup_n_swirl, or cone_only) and fields \
cone_type, base, filling, standard_blend, extra_blend, premium_blend, stick_em, drizzle — use \
only the keys that are non-null. If they chose no filling, no extra blend, no premium blend, \
no Stick'em, or no drizzle, say so briefly using the exact meaning of those choices. Mention that \
pricing is confirmed at the truck if you are not listing dollar totals.

Output rules: plain text only — no Markdown (no asterisks, no hash headings, no backticks). \
Short paragraphs; warm San Antonio ice-cream-truck energy.

Customer order JSON:
{order_json}"""


def summarize_build_order(order: dict) -> str:
    """Turn a validated structured order into conversational plain text."""
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.35,
        max_tokens=512,
    )
    msg = _ORDER_SUMMARY_PROMPT.format(order_json=json.dumps(order, indent=2))
    out = llm.invoke(msg)
    text = getattr(out, "content", None)
    if text is None:
        text = str(out)
    elif isinstance(text, list):
        parts = []
        for block in text:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(getattr(block, "text", str(block)))
        text = "".join(parts)
    else:
        text = str(text)
    text = text.strip()
    if not text:
        raise ValueError("empty summary from model")
    return text
