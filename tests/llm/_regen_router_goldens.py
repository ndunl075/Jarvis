"""Regenerate the router pattern-layer goldens.

    python3 tests/llm/_regen_router_goldens.py

Run this only when you intend to change routing behaviour. Review the
resulting diff: a changed line in ``router_pattern_order_golden.json``
is a first-match-wins precedence change, and a changed value in
``router_pattern_golden.json`` is an utterance that now routes somewhere
else. Both need justifying.

Underscore-prefixed so pytest does not collect it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jarvis.llm.intent_router import _build_patterns  # noqa: E402
from tests.llm.test_router_pattern_equivalence import (  # noqa: E402
    _INTENT_GOLDEN,
    _ORDER_GOLDEN,
    _router_without_registry,
    fixed_time_provider,
    resolve_corpus,
)


def main() -> None:
    order = [p.pattern for p, _b in _build_patterns(fixed_time_provider)]
    _ORDER_GOLDEN.write_text(
        json.dumps(order, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    intents = resolve_corpus(_router_without_registry())
    _INTENT_GOLDEN.write_text(
        json.dumps(intents, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(order)} patterns -> {_ORDER_GOLDEN.name}")
    print(f"wrote {len(intents)} utterances -> {_INTENT_GOLDEN.name}")


if __name__ == "__main__":
    main()
