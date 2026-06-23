"""Shared player-name normalization and fuzzy matching (esports-style)."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

COMBO_MARKERS = (" + ", " / ", " and ", " & ")


def is_combo_player(name: str) -> bool:
    lowered = str(name or "").strip().lower()
    if not lowered:
        return True
    return any(marker in lowered for marker in COMBO_MARKERS)


def normalize_lookup_name(value: str) -> str:
    """PrizePicks-facing key: lowercase, strip accents, collapse spaces."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_compact_key(value: str) -> str:
    """Aggressive key for fuzzy index (no spaces/punctuation)."""
    text = normalize_lookup_name(value)
    return re.sub(r"[^a-z0-9]", "", text)


def names_match(target: str, candidate: str, *, min_fuzzy: float = 0.82) -> bool:
    t = normalize_lookup_name(target)
    c = normalize_lookup_name(candidate)
    if not t or not c:
        return False
    if t == c:
        return True
    if t in c or c in t:
        return True
    t_parts = t.replace("-", " ").split()
    c_parts = c.replace("-", " ").split()
    if t_parts and c_parts and t_parts[-1] == c_parts[-1] and t_parts[0][:1] == c_parts[0][:1]:
        return True
    return SequenceMatcher(None, normalize_compact_key(t), normalize_compact_key(c)).ratio() >= min_fuzzy


def fuzzy_best_match(
    query: str,
    candidates: list[str],
    *,
    min_score: float = 0.82,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    q = normalize_lookup_name(query)
    q_compact = normalize_compact_key(q)
    scored: list[tuple[str, float]] = []
    for cand in candidates:
        c = normalize_lookup_name(cand)
        if not c:
            continue
        score = SequenceMatcher(None, q_compact, normalize_compact_key(c)).ratio()
        if q and (q in c or c in q):
            score = max(score, 0.88 if len(q) >= 4 else score)
        if score >= min_score:
            scored.append((cand, round(score, 3)))
    scored.sort(key=lambda x: x[1], reverse=True)
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for name, score in scored:
        key = normalize_lookup_name(name)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, score))
        if len(out) >= top_n:
            break
    return out
