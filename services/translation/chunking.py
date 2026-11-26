"""Split long text so every NMT input stays within the model's token budget (NLLB: 512)."""

from __future__ import annotations

import re
from typing import Callable

# Sentence boundary: terminator (Latin or CJK) followed by whitespace, or a CJK terminator by itself.
_SENT = re.compile(r"(?<=[.!?…])\s+|(?<=[。！？])")


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENT.split(text.strip()) if p and p.strip()]


def pack_chunks(text: str, count_tokens: Callable[[str], int], max_tokens: int) -> list[str]:
    """Greedily pack whole sentences into chunks of at most ``max_tokens`` tokens.

    A single sentence that is still too long is split on whitespace (words), and as a last resort
    left whole - the backend then truncates it at ``max_input_length`` (documented limitation).
    """
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]
    pieces: list[str] = []
    for sent in split_sentences(text):
        if count_tokens(sent) <= max_tokens:
            pieces.append(sent)
        else:
            pieces.extend(_split_words(sent, count_tokens, max_tokens))
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_words(sentence: str, count_tokens: Callable[[str], int], max_tokens: int) -> list[str]:
    words = sentence.split()
    if len(words) <= 1:
        return [sentence]
    out: list[str] = []
    current: list[str] = []
    for w in words:
        trial = " ".join(current + [w])
        if current and count_tokens(trial) > max_tokens:
            out.append(" ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        out.append(" ".join(current))
    return out
