"""Buggy RAG packer: drops citation metadata before prompt assembly."""

from __future__ import annotations


def render_chunk(text: str) -> str:
    return f"<doc>{text}</doc>"


def build_prompt(question: str, chunk_text: str) -> str:
    return f"Answer with citations.\n{render_chunk(chunk_text)}\nQ: {question}"

