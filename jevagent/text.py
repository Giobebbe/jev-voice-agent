"""Pure text helpers: sentence split, candidate request boundaries, candidate spans.

Jev does not generate text, so free-text arguments (a note title, a search query) are
extracted the TypeSafe way: code enumerates candidate spans of the utterance and Jev
picks one with a Choice (see docs/typesafe/cookbooks_pre_parsed.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Sentence ends: . ? ! followed by space or end, but not inside domains like x.com
_SENT_END = re.compile(r"(?<=[.?!])\s+(?=\S)")
# Places where a second request may start inside one sentence.
_BOUNDARY = re.compile(r",?\s+(?:and then|and also|and after that|and|then|after that)\s+", re.I)
_WORD = re.compile(r"[\w'’@#&+\-]+(?:\.[\w\-]+)*", re.U)
_EDGE_PUNCT = " \t\n.,!?;:\"'“”‘’()[]"

FILLERS = {
    "um", "uh", "erm", "hmm", "okay", "ok", "so", "alright", "right", "please", "like",
}


def normalize(text: str) -> str:
    """Collapse whitespace and drop double quotes: Scribe quotes names ('"Mission Plan."'),
    and an unbalanced quote makes a clause look unfinished."""
    text = re.sub(r"[\"“”]", "", text or "")
    return re.sub(r"\s+", " ", text).strip()


def sentences(text: str) -> list[str]:
    text = normalize(text)
    if not text:
        return []
    return [s.strip() for s in _SENT_END.split(text) if s.strip(_EDGE_PUNCT)]


@dataclass(frozen=True)
class Boundary:
    index: int  # char offset where the right side starts
    left: str
    right: str


def boundaries(sentence: str, max_boundaries: int = 4) -> list[Boundary]:
    out: list[Boundary] = []
    for m in _BOUNDARY.finditer(sentence):
        left = sentence[: m.start()].strip(_EDGE_PUNCT)
        right = sentence[m.end():].strip(_EDGE_PUNCT)
        if len(left.split()) < 2 or not right:
            continue
        out.append(Boundary(m.end(), left, right))
        if len(out) >= max_boundaries:
            break
    return out


def split_at(sentence: str, cuts: list[Boundary]) -> list[str]:
    """Split a sentence at the accepted boundaries (cut points = start of right side)."""
    if not cuts:
        return [sentence.strip(_EDGE_PUNCT + " ")]
    pieces: list[str] = []
    start = 0
    for b in sorted(cuts, key=lambda b: b.index):
        m = None
        # find the connector that ends at b.index to cut before it
        for mm in _BOUNDARY.finditer(sentence, start):
            if mm.end() == b.index:
                m = mm
                break
        end = m.start() if m else b.index
        piece = sentence[start:end].strip(_EDGE_PUNCT + " ")
        if piece:
            pieces.append(piece)
        start = b.index
    tail = sentence[start:].strip(_EDGE_PUNCT + " ")
    if tail:
        pieces.append(tail)
    return pieces


def key(text: str) -> str:
    """Cache key: the same words with different punctuation or case are the same clause."""
    return " ".join(w.lower().strip(_EDGE_PUNCT) for w in text.split() if w.strip(_EDGE_PUNCT))


def all_pieces(sentence: str, bs: list[Boundary], max_boundaries: int = 3) -> list[str]:
    """Every clause any split of the sentence could produce (contiguous runs between boundaries)."""
    bs = sorted(bs, key=lambda b: b.index)[:max_boundaries]
    out: list[str] = []
    for i in range(len(bs) + 1):
        for j in range(i, len(bs) + 1):
            keep = [b for k, b in enumerate(bs) if k < i or k >= j]
            pieces = split_at(sentence, keep)
            # the run i..j is the piece that starts at boundary i (or the start)
            out.append(pieces[i] if i < len(pieces) else pieces[-1])
    return list(dict.fromkeys(p for p in out if p))


def words(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def candidate_spans(text: str, max_len: int = 8, cap: int = 240) -> list[str]:
    """Every contiguous run of up to max_len words, as it appears in the text.

    Spans starting or ending with a filler word are dropped, duplicates are removed
    case-insensitively, and the list is capped (Choice allows at most 255 options).
    """
    ws = words(text)
    seen: set[str] = set()
    out: list[str] = []
    for n in range(1, max_len + 1):
        for i in range(0, len(ws) - n + 1):
            s, e = ws[i][0], ws[i + n - 1][1]
            span = text[s:e].strip(_EDGE_PUNCT)
            if not span:
                continue
            first, last = span.split()[0].lower(), span.split()[-1].lower()
            if first in FILLERS or last in FILLERS:
                continue
            key = span.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(span)
    if len(out) > cap:
        # keep shorter spans first: titles and queries are rarely very long
        out = out[:cap]
    return out


_DOMAIN = re.compile(r"\b([a-z0-9-]+(?:\s?(?:\.|dot)\s?[a-z0-9-]+)*\s?(?:\.|dot)\s?(?:com|org|net|io|ai|it|dev|app|co|tv|me))\b", re.I)


def domains(text: str) -> list[str]:
    """Domains said or transcribed in the text: 'X.com', 'x dot com' -> 'x.com'."""
    out = []
    for m in _DOMAIN.finditer(text):
        d = re.sub(r"\s*(?:\.|\bdot\b)\s*", ".", m.group(1), flags=re.I).lower().replace(" ", "")
        if d not in out:
            out.append(d)
    return out


def names_item(clause: str, item: str | None) -> bool:
    """Does the clause literally name this file or folder (every word of its stem)?"""
    if not item:
        return False
    stem = item.rstrip("/").split("/")[-1].rsplit(".", 1)[0]
    words = {w.lower() for w in re.findall(r"[\w']+", stem)}
    said = {w.lower() for w in re.findall(r"[\w']+", clause)}
    return bool(words) and words <= said
