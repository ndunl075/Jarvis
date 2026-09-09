# Contributing to Jarvis

Thanks for taking the time. Jarvis is a fully local Windows voice assistant —
no subscription, no telemetry, no cloud account for the default pipeline. That
constraint is the product, so please keep it in mind when proposing changes.

- **What Jarvis is and does:** [`README.md`](README.md)
- **The technical contract:** [`SPEC.md`](SPEC.md) — module boundaries, config
  schema, event bus, state machine, tool registry
- **How it is built and packaged:** [`BUILD.md`](BUILD.md)
- **Reporting a vulnerability:** [`SECURITY.md`](SECURITY.md)

---

## Development environment

Jarvis targets **Windows 10/11 x64** and **Python 3.11+**. You can run the test
suite, ruff, and pyright on Linux or macOS, but the app itself only fully works
on Windows: everything under `jarvis/platform/windows.py` raises
`NotImplementedError` elsewhere, and the tools that depend on it return a failed
`ToolResult` rather than crashing.

### 1. Clone and create a virtual environment

PowerShell (Windows):

```powershell
git clone https://github.com/ndunl075/Jarvis.git
cd Jarvis
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

bash (Linux/macOS, for tests and linting only):

```bash
git clone https://github.com/ndunl075/Jarvis.git
cd Jarvis
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `dev` extra pulls in pytest, pytest-asyncio, ruff, pyright, and pyinstaller.

`pyautogui` and `pynput` are declared with a `sys_platform == "win32"` marker, so
they are not installed on Linux/macOS. Both are imported lazily at their call
sites and both degrade gracefully when missing, so the suite passes without
them — but screen capture and global hotkeys obviously only work on Windows.

On a bare Linux box, PySide6 and sounddevice also need a few system libraries
that are not Python packages. On Debian/Ubuntu:

```bash
sudo apt-get install -y libegl1 libgl1 libxkbcommon-x11-0 libdbus-1-3 libportaudio2
```

### 2. Install Ollama separately

**Ollama is not bundled and is not a pip dependency.** Jarvis talks to it over
HTTP on `localhost`. Install it from <https://ollama.com/download>, then pull at
least the default model:

```powershell
ollama pull qwen2.5:7b-instruct
```

Optional, depending on what you are working on:

```powershell
ollama pull qwen2.5:3b-instruct   # deep-research worker model
ollama pull llava:7b              # see_screen vision tool
```

Nothing in the test suite requires a running Ollama — the LLM client is
exercised against fakes — but you need one to run the app.

### 3. Run the app

```powershell
python -m jarvis
```

From source, logs go to the console; the packaged build writes to
`%APPDATA%\Jarvis\logs\jarvis.log` instead. Config lives at
`%APPDATA%\Jarvis\config.json` on Windows and at `~/.jarvis/config.json`
elsewhere.

There is also an audio-only harness that skips the LLM:

```powershell
python -m jarvis.dev.audio_loopback
```

---

## Tests, linting, and types

Run all three before opening a pull request. CI
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs exactly these
commands on Windows and Linux, on Python 3.11 and 3.12, and every one of them is
a blocking gate.

```powershell
ruff check jarvis tests
pyright
pytest
```

- **ruff** reads `[tool.ruff]` from `pyproject.toml`: line length 100, target
  `py311`, rule sets `E`, `F`, `I`, `B`, `UP`, `N`, `W`. Do not add per-file
  ignores to silence a finding you could fix instead.
- **pyright** reads `[tool.pyright]`: `standard` mode over `jarvis` and `tests`.
  Run it with no arguments — passing paths overrides the configured `include`.
- **pytest** reads `[tool.pytest.ini_options]`: `testpaths = tests`,
  `asyncio_mode = auto` (so `async def` tests need no decorator), and
  `addopts = -m 'not manual'`.

### The `manual` marker

Some checks cannot run unattended — they measure real process RSS after sleep
mode, or need a live external MCP server. Those are marked:

```python
@pytest.mark.manual
def test_rss_drops_after_sleeping(): ...
```

`manual` tests are **deselected by default** and are never run in CI. Opt in
explicitly:

```powershell
pytest -m manual
```

Use the marker only when a test genuinely requires real hardware, a real
network peer, or a human reading the result. "Slow" or "flaky" is not a reason —
fix the test.

### Qt / UI tests

`tests/ui/` constructs real `QWidget`s. `tests/ui/conftest.py` sets
`QT_QPA_PLATFORM=offscreen` before PySide6 is imported, which is what lets them
run headless. If you add a test that touches Qt from outside `tests/ui/`, set
that environment variable the same way, before the first PySide6 import.

---

## Architectural conventions

These are the rules the existing code follows. They come from `SPEC.md` and from
the module docstrings; they are not stylistic preferences.

### Windows-specific code lives in one file

Every Windows API call goes through `jarvis/platform/windows.py`. Quoting its
module docstring:

> Every Windows-specific call lives here so tool implementations stay pure
> Python and testable by patching this module. Tools never import ctypes,
> windll, subprocess, or webbrowser directly; they go through these functions
> and tests patch them.

Functions there raise `OSError` on Windows-API failure so callers can turn the
message into a `ToolResult.error`, and `NotImplementedError` on non-Windows
hosts. This single seam is what makes the tools testable at all — a tool test
patches the `winplat` alias in the tool module
(`patch("jarvis.tools.local.open_app.winplat.launch_app")`) rather than
monkey-patching `subprocess.Popen`.

`SPEC.md` states the boundary rules directly:

> No `win32` imports outside `platform/windows.py`. No file outside `ui/` may
> import from `PySide6`. No file outside `audio/` may import audio libraries.

### Tools implement the `Tool` protocol

Every callable tool — local or adapted from an external MCP server — satisfies
the `Tool` protocol in [`jarvis/tools/registry.py`](jarvis/tools/registry.py):

```python
name: str
description: str
args_schema: type[BaseModel]
requires_confirmation: bool

async def execute(self, args: BaseModel) -> ToolResult: ...
```

It is a `Protocol`, not a base class, so MCP wrappers satisfy it structurally.
The registry's rules matter as much as the shape:

- **Argument validation happens at the registry boundary.** `execute()` receives
  an already-validated pydantic model. Bad arguments surface as
  `ToolResult(success=False, error=...)`, never as an exception the router has
  to catch.
- **Failures are return values, not exceptions.** A tool that cannot do its job
  returns `ToolResult(success=False, error="...")`. The router speaks the error
  back; it should never need a `try`/`except` around a tool call.
- **Names are unique.** `register()` raises `ToolNameCollision`. Built-ins
  register first and win; MCP adapters catch the exception and drop just the
  colliding tool. No silent overwrites.
- **`list_enabled()` is the source of truth for visibility.** Per-tool
  enable/disable comes from config and is re-checked on every dispatch.
- Tool names must match `TOOL_NAME_REGEX` (`[a-zA-Z0-9_-]{1,64}`) — the
  constraint the Ollama/OpenAI function-calling surface imposes.

Add a new local tool under `jarvis/tools/local/`, register it in the composition
root in `jarvis/app.py`, and give it tests under `tests/tools/local/`.

### Heavy dependencies are imported lazily

`pyautogui`, `pynput`, and the ML runtimes are imported inside the function that
needs them, not at module scope. This keeps cold-start cheap for users who never
touch those features, lets tests inject fakes via `sys.modules`, and is what
allows the suite to run on a machine where those packages are not installed.
Keep new heavy imports local to their call site.

### Write the rationale where the decision lives

`BUILD.md` requires that architecture decisions are recorded as a comment block
at the point of the decision, including the alternatives considered and why they
were rejected. The codebase does this consistently — see the header of
`jarvis/tools/registry.py`, the library-choice note in `jarvis/ui/hotkeys.py`,
and the dependency comments in `pyproject.toml`. If you make a non-obvious
choice, leave the reasoning next to the code, not only in the PR description.

### Core contracts

Modules talk through four contracts in `jarvis/core/`, all defined in `SPEC.md`:
`config.py` (pydantic v2 schema persisted to `config.json`, with
`CURRENT_SCHEMA_VERSION` and a numbered migration per bump), `events.py` (typed
pub/sub), `state_machine.py` (two orthogonal axes, `Mode` and
`ConversationalState`, with an explicit transition table — illegal transitions
raise `IllegalTransition`), and `lifecycle.py` (the `Loadable` protocol).

**If you add or rename a config field, bump `CURRENT_SCHEMA_VERSION` and add the
matching migration function.** Existing installs must load without losing
settings.

### Privacy is a hard constraint

No analytics, no telemetry, no crash reporting, no phone-home. Any new network
call must be documented in the README's *Security & privacy model* table, must
be triggered by an explicit user action, and — if it needs credentials — must
degrade to the free local path when no key is configured.

---

## Commits and pull requests

- **Branch from `main`** and use a descriptive branch name
  (`fix/wake-word-sensitivity`, `feat/timer-tool`).
- **Keep commits focused.** One logical change per commit. A refactor and a
  behavior change belong in separate commits.
- **Write real commit messages.** Imperative subject under ~72 characters, then
  a body explaining *why* the change is needed. The existing history is the
  reference for tone.
- **Open a PR** and fill in the template
  ([`.github/pull_request_template.md`](.github/pull_request_template.md)): what
  changed, why, how you tested it, and the checklist.
- **CI must be green.** ruff, pyright, and pytest all run on Windows and Linux
  across Python 3.11 and 3.12. Do not disable a check to get a build passing.
- **Update the docs when behavior changes.** New voice commands and new
  capabilities belong in the README feature list and in the Help catalog
  (`jarvis/ui/capabilities.py`) so they show up in the in-app help panel and the
  command palette. Add an entry under `## [Unreleased]` in
  [`CHANGELOG.md`](CHANGELOG.md).
- **Large changes: open an issue first.** Anything that alters a `SPEC.md`
  contract, adds a dependency, or introduces a new network destination should be
  discussed before you write the code. `SPEC.md` is the contract and is not
  edited without explicit maintainer approval.

---

## Dependency lockfile

`pyproject.toml` pins only major-version ranges, so two installs a month apart
are not the same install. [`requirements.lock`](requirements.lock) records an
exact, hash-pinned resolution of the **runtime** dependencies (not the `dev`
extra) so a build is reproducible:

```powershell
pip install --require-hashes -r requirements.lock
```

Regenerate it whenever `pyproject.toml`'s dependencies change, and commit the
result in the same PR:

```powershell
pip install uv
uv pip compile --universal --generate-hashes --python-version 3.11 `
    --output-file requirements.lock pyproject.toml
```

`--universal` matters. An ordinary resolve is specific to the machine that ran
it: it drops every requirement whose environment marker is false there and
records wheel hashes for that one platform. Because Jarvis ships from Windows
and depends on Windows-only packages (`pyautogui`, `pynput`, and `pywin32`
transitively), a lockfile resolved on Linux would silently omit them. Universal
mode resolves for every supported platform at once and keeps the markers, so one
committed file serves both the Windows build and the Linux CI leg.

`pip-compile` from pip-tools is the alternative and produces the same file
format, but it has no universal mode — its output is only valid for the platform
that generated it. If you use it, run it on **Windows with Python 3.11** and
treat the result as Windows-only:

```powershell
pip install pip-tools
pip-compile --generate-hashes --output-file=requirements.lock pyproject.toml
```

The header comment inside `requirements.lock` records how the committed version
was produced. Do not hand-edit it, and never invent a hash.

---

## Reporting bugs and requesting features

Use the issue forms:
[bug report](https://github.com/ndunl075/Jarvis/issues/new?template=bug_report.yml)
· [feature request](https://github.com/ndunl075/Jarvis/issues/new?template=feature_request.yml).

For a bug, the log excerpt and your Ollama model matter more than anything else —
the bug form asks for both. Open questions and "how do I…" belong in
[Discussions](https://github.com/ndunl075/Jarvis/discussions).

**Do not file security vulnerabilities as public issues.** See
[`SECURITY.md`](SECURITY.md).

---

## License

Jarvis is MIT licensed. By contributing, you agree that your contributions are
licensed under the same terms.
