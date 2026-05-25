"""App name normalization and fuzzy resolution.

Platform code builds candidate maps (Start Menu shortcuts, registry App
Paths, Steam library titles); this module holds the shared matching logic
so tests can run without Windows installed.
"""

from __future__ import annotations

import difflib
import re
from typing import TypeVar

_T = TypeVar("_T")

# Minimum difflib SequenceMatcher ratio for a character-similarity match
# to count on its own. Below this, only token-overlap / substring /
# prefix signals can clear the resolve cutoff. Tuned so real typos
# ("photoshpo" -> "photoshop" ~0.89) pass while near-anagrams
# ("teams" vs "steam" = 0.80) and unrelated short names ("vscode" vs
# "discord" ~0.5) do not.
_DIFFLIB_TYPO_FLOOR: float = 0.85

# Strip conversational filler before matching.
_QUERY_FILLERS = re.compile(
    r"^(?:my|the|a|an)\s+",
    re.IGNORECASE,
)


# Peel "open chrome", "launch notepad", etc. down to the app token.
_OPEN_LEADING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^hey\s+jarvis,?\s+", re.IGNORECASE),
    re.compile(r"^(?:(?:can|could|would)\s+you\s+(?:please\s+)?)", re.IGNORECASE),
    re.compile(r"^please\s+", re.IGNORECASE),
    re.compile(r"^(?:i\s+)?(?:want\s+(?:you\s+)?to\s+)", re.IGNORECASE),
    re.compile(r"^(?:open|launch|start|run)\s+", re.IGNORECASE),
    re.compile(r"^(?:the\s+)?(?:app(?:lication)?|programme?|program)\s+", re.IGNORECASE),
    re.compile(r"^(?:my|the|a|an)\s+", re.IGNORECASE),
)
_OPEN_TRAILING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\s+for\s+me$", re.IGNORECASE),
    re.compile(r"\s+please$", re.IGNORECASE),
    re.compile(r"\s+sir$", re.IGNORECASE),
    re.compile(r"\s+up$", re.IGNORECASE),
)


def normalize_open_query(text: str) -> str:
    """Distill an utterance or tool arg to an app name token."""
    s = text.strip()
    if not s:
        return ""
    s = re.sub(r"[\s.!?,]+$", "", s)
    s = _peel_patterns(s, _OPEN_LEADING_PATTERNS)
    s = _peel_patterns(s, _OPEN_TRAILING_PATTERNS)
    s = re.sub(r"[\s.!?,]+$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _peel_patterns(s: str, patterns: tuple[re.Pattern[str], ...]) -> str:
    while True:
        new = s
        for pat in patterns:
            new = pat.sub("", new, count=1)
        if new == s:
            break
        s = new.strip()
    return s


def normalize_query(text: str) -> str:
    """Lowercase, collapse non-alphanumerics to spaces, strip fillers."""
    s = text.strip().lower()
    while True:
        new = _QUERY_FILLERS.sub("", s, count=1)
        if new == s:
            break
        s = new
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _match_score(needle: str, needle_tokens: set[str], key: str) -> float:
    """Score a normalized query against one normalized candidate key in
    [0, 1]. Combines four strategies and takes the best:

    1. Exact equality (1.0).
    2. Token overlap — how much of the query's words appear in the key.
       This is what makes single-word queries find multi-word app names
       ("word" -> "microsoft word", "teams" -> "microsoft teams").
    3. Substring prominence — a query that appears whole inside the key
       ("code" -> "visual studio code") or vice versa.
    4. difflib character-ratio — catches typos ("photoshpo").
    """
    if needle == key:
        return 1.0

    # difflib character-ratio is the least precise signal — it produces
    # false positives between unrelated short names ("vscode" vs
    # "discord", "calculator" vs "capcut" both score ~0.5). Only let it
    # count when it's high enough to mean a genuine typo ("photoshpo" ->
    # "photoshop"); otherwise the semantic signals below must carry the
    # match.
    ratio = difflib.SequenceMatcher(None, needle, key).ratio()
    score = ratio if ratio >= _DIFFLIB_TYPO_FLOOR else 0.0

    # Token overlap: fraction of query words found in the key. Full
    # coverage (every query word present) is a strong signal even when
    # the key carries extra words like a vendor prefix.
    key_tokens = set(key.split())
    if needle_tokens and key_tokens:
        common = needle_tokens & key_tokens
        if common:
            coverage = len(common) / len(needle_tokens)
            token_score = 0.95 * coverage if coverage >= 1.0 else 0.8 * coverage
            score = max(score, token_score)

    # Substring prominence. Guard tiny needles (len >= 3) so "go" doesn't
    # match every key containing those letters.
    shorter, longer = (needle, key) if len(needle) <= len(key) else (key, needle)
    if len(shorter) >= 3 and shorter in longer:
        score = max(score, 0.6 + 0.4 * (len(shorter) / len(longer)))

    # Prefix bonus: "spot" -> "spotify".
    if len(needle) >= 3 and key.startswith(needle):
        score = max(score, 0.7 + 0.3 * (len(needle) / len(key)))

    return score


def fuzzy_resolve(
    query: str,
    choices: dict[str, _T],
    *,
    cutoff: float = 0.55,
) -> tuple[str, _T] | None:
    """Return (matched_key, value) for the best fuzzy match, or None.

    Scores every candidate via _match_score and returns the highest if it
    clears `cutoff`. `choices` keys must already be normalized via
    normalize_query. Ties resolve to the first key in iteration order."""
    needle = normalize_query(query)
    if not needle or not choices:
        return None
    if needle in choices:
        return needle, choices[needle]
    needle_tokens = set(needle.split())
    best_key: str | None = None
    best_score = 0.0
    for key in choices:
        score = _match_score(needle, needle_tokens, key)
        if score > best_score:
            best_score = score
            best_key = key
    if best_key is not None and best_score >= cutoff:
        return best_key, choices[best_key]
    return None
