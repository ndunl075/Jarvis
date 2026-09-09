# Deep Research Ultra — Architecture

Deep Research Ultra is an optional, **$0/month** upgrade to Jarvis’s standard deep research pipeline. It keeps the same planner/worker architecture as Perplexity-style deep research but swaps in free-tier retrieval and an optional cloud planner (Groq) while the worker stays on local Ollama.

## Goals

| Goal | How Ultra addresses it |
|------|-------------------------|
| Better search ranking | Brave Search API (free 2,000 queries/mo) with DuckDuckGo fallback |
| Read JS-heavy pages | Jina Reader (`r.jina.ai`) when direct `httpx` fetch returns &lt;500 chars — the page URL is sent to Jina, so private/loopback URLs are never forwarded (see *Fetch safety*) |
| Stronger planning/synthesis | Groq `llama-3.3-70b-versatile` planner (free tier) with local fallback |
| Deeper coverage | Up to **3** gap-fill iterations per sub-question (vs 1 in standard mode) |
| Longer source context | **16,000** chars/page minimum when Ultra is on (vs 6,000 default) |
| Privacy / cost | Worker extraction remains **local Ollama**; no paid subscription required. Ultra does send each fetched page's URL (not its contents) to `r.jina.ai` when the direct fetch is thin |

## Modes compared

| Setting | Standard | Ultra |
|---------|----------|-------|
| Search | DuckDuckGo HTML scrape | Brave JSON API → DDG fallback |
| Page fetch | `httpx` + regex DOM | Same + Jina Reader fallback |
| Planner | `research.planner_model` or main LLM | `groq/llama-3.3-70b-versatile` (or custom) → local if no Groq key |
| Worker | Local Ollama | Local Ollama (unchanged) |
| Gap-fill passes | 1 | `research.ultra_gap_fill_iterations` (default 3) |
| Max page chars | `research.max_page_chars` | `max(6000, 16000)` |

## End-to-end pipeline

```
User: "deep research quantum computing"
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Voice: intent_router → deep_research tool                     │
│ UI: Deep Research panel → DeepResearchWorker                  │
│ Config: build_deep_research_config(research, main_llm_model)  │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────── PLANNERS ─────────────────┐
│ plan_sub_questions (depth N)              │  ← Groq or Ollama
│ expand_search_queries (3 per sub-Q)       │
│ identify_gap_queries (0–2, up to 3 loops) │
│ synthesize_executive_overview             │
└───────────────────────────────────────────┘
        │
        ▼
┌──────────────── WORKER (always Ollama) ───┐
│ extract_claims → JSON {summary, claims[]} │
└───────────────────────────────────────────┘
        │
        ▼
┌──────────────── RETRIEVAL ────────────────┐
│ fetch_search_snippets (Brave or DDG)      │
│ _diversify_snippets (domain cap)          │
│ fetch_page_text (+ Jina if Ultra)         │
└───────────────────────────────────────────┘
        │
        ▼
┌──────────────── PERSISTENCE ──────────────┐
│ %APPDATA%/Jarvis/deep_research/<id>/      │
│   state.json  — resumable session         │
│   report.md   — live markdown report      │
└───────────────────────────────────────────┘
```

## Fetch safety

`web_page_fetch.py` handles URLs that come from search results or from the
planner LLM, i.e. from outside Jarvis. Both the direct path and the Jina path
apply the same two guards:

| Guard | Behavior |
|-------|----------|
| Response size | Body is streamed and capped at **5 MiB** (`_MAX_RESPONSE_BYTES`); the rest is never pulled from the socket. `Accept-Encoding: identity` is sent so the budget measures wire bytes rather than post-decompression bytes. |
| Destination | Loopback, RFC1918 private, link-local (`169.254.0.0/16`), reserved, multicast and IPv6 unique-local destinations are refused. Hostnames are resolved first and **every** answer is checked, and redirects are followed manually (max 5 hops) so each hop is re-validated and cannot escape `http`/`https`. |

Consequence for Ultra: a URL that fails the destination check is not fetched
**and is not sent to `r.jina.ai`** — Jarvis will not hand a third party the
address of something on the user's own machine or LAN. Refused fetches return
`""`, so the pipeline falls back to the search snippet as usual.

Resolution happens a moment before the connect, so this narrows but does not
eliminate DNS rebinding.

## Module map

| Module | Role |
|--------|------|
| `jarvis/tools/local/deep_research_runner.py` | Pipeline orchestration, `DeepResearchConfig`, `build_deep_research_config()` |
| `jarvis/tools/local/deep_research_store.py` | Session/report persistence (`ultra_mode` flag on state) |
| `jarvis/tools/local/research_search.py` | Brave + DDG search |
| `jarvis/tools/local/web_page_fetch.py` | Direct HTML extract + optional Jina fallback; size cap + private-destination guard |
| `jarvis/tools/local/research_llm.py` | `chat_once` routing: `groq/…` → Groq API, else Ollama |
| `jarvis/tools/local/deep_research_tools.py` | Voice: start / pause / resume / delete |
| `jarvis/tools/local/deep_research_ultra_tools.py` | Voice: enable / disable Ultra |
| `jarvis/ui/deep_research_panel.py` | Qt panel + `DeepResearchWorker` thread |
| `jarvis/ui/settings/tabs/tools.py` | Settings UI toggle + API key fields |
| `jarvis/core/config.py` | `ResearchConfig.ultra_*` persisted fields (schema v17) |

## Configuration

### Persisted (`config.json` → `research`)

```json
{
  "ultra_enabled": false,
  "brave_api_key": "dpapi:AQAAANCMnd8BFdERjHoAwE/Cl+sBAAAA...",
  "groq_api_key": "",
  "ultra_planner_model": "groq/llama-3.3-70b-versatile",
  "ultra_gap_fill_iterations": 3
}
```

`brave_api_key` and `groq_api_key` are encrypted at rest (schema v21). A
non-empty value is written as `dpapi:<base64>` — Windows DPAPI ciphertext,
bound to your Windows user account. Empty values stay `""`. A key you paste
in as plaintext by hand still works and is encrypted on the next save. See
`jarvis/platform/secrets.py`.

### Environment variables (override config file)

| Variable | Purpose |
|----------|---------|
| `JARVIS_BRAVE_API_KEY` | Brave Search API subscription token |
| `JARVIS_GROQ_API_KEY` | Groq API key for `groq/*` planner models |

Resolution order: **env var → config field → empty (graceful fallback)**.

### Runtime `DeepResearchConfig`

Built by `build_deep_research_config()` when a session starts:

- `ultra=True` copies from `research.ultra_enabled`
- `search_provider="brave"` only if Brave key is present; else `"ddg"`
- `use_jina_reader=True` when Ultra is on
- `groq_api_key` passed into planner `chat_once` calls only
- `gap_fill_max_iterations` = 3 (Ultra) or 1 (standard)

## API keys (free signup)

1. **Brave**: https://api.search.brave.com — 2,000 queries/month free.
2. **Groq**: https://console.groq.com — free tier for `llama-3.3-70b-versatile`.

Without keys, Ultra still enables Jina + multi gap-fill + 16k pages; search stays DDG and planner stays local.

## Enable / disable

### Settings UI

**Settings → Tools → Deep research Ultra (free APIs)** → checkbox **Enable Deep Research Ultra mode**.

Optional Brave/Groq key fields (password echo). Changes persist via `save_config`.

### Voice

| Utterance | Tool |
|-----------|------|
| "enable deep research ultra" | `enable_deep_research_ultra` |
| "turn on ultra research" | `enable_deep_research_ultra` |
| "disable deep research ultra" | `disable_deep_research_ultra` |
| "use normal deep research" | `disable_deep_research_ultra` |

Pattern layer in `jarvis/llm/intent_router.py` (before topic-based `deep research …`).

## Gap-fill loop (Ultra)

Standard mode runs **one** planner review → optional re-search → re-extract.

Ultra runs up to `ultra_gap_fill_iterations` (default **3**):

```
for round in 1..N:
    gap_qs = identify_gap_queries(notes)
    if not gap_qs: break
    fetch gap sources → re-extract_claims
```

Each round is checkpointed to disk (pause/resume safe).

## LLM routing

`research_llm.chat_once()`:

- Model `groq/llama-3.3-70b-versatile` → `POST https://api.groq.com/openai/v1/chat/completions`
- Any other model string → Ollama `POST {endpoint}/api/chat`

Worker calls always use the worker model (typically local `qwen2.5:3b-instruct`).

## Report output

`report.md` header includes when Ultra was active:

```markdown
- **Mode:** Ultra (Brave + Jina + multi gap-fill)
```

Footer:

```markdown
_Generated by Jarvis deep research (Ultra mode)._
```

## Limitations (honest)

- Not equivalent to Perplexity Deep Research on DRACO (~70% vs ~40–50% estimated for this stack).
- No browser automation (Comet-style DOM agent).
- No sandboxed code execution for quantitative verification.
- Groq/Brave free tiers rate-limit under heavy use.
- API keys in config are encrypted at rest with Windows DPAPI (schema v21), which binds them to your Windows user account — not to a password. Anything running as you can still decrypt them; prefer env vars if that matters. Off Windows there is no DPAPI and the keys stay plaintext (logged at startup).

## Schema migration

Config `schema_version` **17** adds Ultra fields via `_migrate_v16_to_v17`.
Config `schema_version` **21** encrypts `brave_api_key` / `groq_api_key`
(and `mcp_servers[].auth_token`) via `_migrate_v20_to_v21`.

## Testing

- `tests/tools/local/test_research_search.py`
- `tests/tools/local/test_research_llm.py`
- `tests/tools/local/test_deep_research_ultra_config.py`
- `tests/tools/local/test_deep_research_ultra_tools.py`
- `tests/platform/test_secrets.py` — DPAPI seam, round-trip, corrupt ciphertext
- `tests/core/test_config.py` — v21 migration and the load/save/load cycle
- `tests/llm/test_intent_router.py` (ultra voice patterns)
- `tests/core/test_config.py` (v16→v17 migration)
