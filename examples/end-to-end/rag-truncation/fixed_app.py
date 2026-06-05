"""Fixed RAG packer: carries stable citation labels into the prompt."""

from __future__ import annotations


def render_chunk(text: str, citation: str) -> str:
    return f"<doc cite=\"{citation}\">{text}</doc>"


def build_prompt(question: str, chunk_text: str, citation: str) -> str:
    return f"Answer with citations.\n{render_chunk(chunk_text, citation)}\nQ: {question}"

