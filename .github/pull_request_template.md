## What changed

<!-- One or two sentences. What does this PR do? -->

## Why

<!-- The problem this solves. Link the issue if there is one: Fixes #123 -->

## How it was tested

<!--
Be specific. "Ran the tests" is not testing — which tests, and did you exercise
the change by voice or through the UI? Note the OS and Python version you used,
and whether anything could only be verified on Windows.
-->

## Checklist

- [ ] `ruff check jarvis tests` is clean
- [ ] `pyright` is clean
- [ ] `pytest` passes (new behavior has tests; `manual` marker only where a test genuinely needs real hardware or a live service)
- [ ] Docs updated if behavior changed — README feature list, the capability catalog in `jarvis/ui/capabilities.py`, and an entry under `## [Unreleased]` in `CHANGELOG.md`
- [ ] No new Windows-specific call outside `jarvis/platform/windows.py`, and no `PySide6` import outside `jarvis/ui/`
- [ ] Config schema changes bump `CURRENT_SCHEMA_VERSION` and add a migration
- [ ] Any new network destination is opt-in and documented in the README's *Security & privacy model* table
- [ ] Non-obvious decisions are explained in a comment at the point of the decision
