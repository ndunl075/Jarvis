"""Shared typing seams for the test suite.

Nothing here fakes behaviour. These are the few places where a test
legitimately knows more than the production type does, and where saying
so once beats repeating the same cast at three dozen call sites. None of
them changes what any test does at runtime: `as_mock` and `output_text`
are pure `cast()`, and `fake_module` writes the same module `__dict__`
that attribute assignment writes.

If you are reaching for another helper, prefer typing the fake properly
instead: a fake that structurally satisfies the protocol it stands in
for breaks at check time when the protocol changes, which is the whole
point of running pyright over `tests/`.
"""

from __future__ import annotations

import types
from typing import cast
from unittest.mock import AsyncMock

from jarvis.tools.registry import ToolResult


def as_mock(attr: object) -> AsyncMock:
    """Narrow an attribute a test replaced with a mock back to a mock.

    Tests here patch mocks onto *concretely typed* objects — `app.qt_app`
    is declared `QApplication`, `lm.load_all` is a bound method — so the
    checker still sees the declared type and rejects `.assert_called_*`,
    `.call_args`, `.call_count`, `.side_effect` and friends. This is the
    one narrow cast at that seam.

    `AsyncMock` is the return type because it carries both the sync
    `Mock` surface (`assert_called_once_with`, `call_args_list`, …) and
    the await assertions (`assert_awaited_once_with`, `assert_not_awaited`),
    so a single helper covers `MagicMock` and `AsyncMock` patch sites
    alike. It does not make anything async: this is a `cast`, and the
    object handed back is the one passed in.
    """
    return cast(AsyncMock, attr)


def output_text(result: ToolResult) -> str:
    """`ToolResult.output` as the string a spoken-output test expects.

    `output` is declared `str | dict | None` because a handful of tools
    return structured payloads for the UI. Tools whose result is spoken
    return `str`, and their tests assert on it with `.lower()`,
    `.startswith()` and friends — which the union does not offer.

    Exactly equivalent to the `(result.output or "")` idiom it replaces:
    a dict output would raise the same `AttributeError` on the following
    string method that it raised before.
    """
    return cast(str, result.output or "")


def fake_module(name: str, **attrs: object) -> types.ModuleType:
    """A stub module carrying `attrs`, for `patch.dict(sys.modules, ...)`.

    Several tools import an optional dependency (`pyautogui`, `psutil`,
    `PIL`) lazily inside `execute()`, and the tests stand a stub module in
    front of that import rather than pull the real hooks into a unit test.
    `types.ModuleType` declares no attributes, so `fake.screenshot = ...`
    is an error the checker is right about and the test does not care
    about. Populating `__dict__` — which is precisely what attribute
    assignment does at runtime — says the same thing checkably.
    """
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module
