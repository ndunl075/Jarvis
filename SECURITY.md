# Security Policy

Jarvis runs on your own machine, with your own user account's privileges, and it
can open applications, type keystrokes, capture the screen, and read the
clipboard on behalf of a local language model. That surface deserves a real
disclosure process. Thank you for helping keep it safe.

## Supported versions

Jarvis is pre-1.0 and released from a single line of development. Only the most
recent release receives security fixes; there are no maintenance branches for
older versions.

| Version | Supported |
| ------- | --------- |
| 0.0.1 (current version) | Yes |
| `main` | Yes — fixes land here first |
| Anything older | No — upgrade to the current version |

## Reporting a vulnerability

**Please do not open a public issue, discussion, or pull request for a security
problem.** A public report tells everyone running Jarvis about the flaw before a
fix exists.

**Preferred: GitHub private vulnerability reporting.** Go to the
[Security tab](https://github.com/ndunl075/Jarvis/security) and click
**Report a vulnerability**. This opens a private advisory visible only to you and
the maintainers, and it is where the fix and any CVE will be coordinated.

> **Maintainer note:** private vulnerability reporting must be switched on for
> that button to exist. Enable it in the repository's **Settings**, under the
> security settings, via the **Private vulnerability reporting** toggle. Until it
> is on, reporters have no private channel at all — please turn it on.

**If the private reporting button is not available,** open a public issue that
contains *no technical detail* — just "I would like to report a security issue
privately, please open a channel" — and wait to be contacted. Withholding the
details is the point; do not include a proof of concept in that issue.

### What to include

- The version you are running (Settings → About, or the zip filename) and whether
  it is a packaged build or run from source
- Windows version, Ollama version, and the model in use, if relevant
- A clear description of the issue and its impact
- Reproduction steps or a proof of concept
- Any configuration required to trigger it — especially whether Deep Research
  Ultra mode or an external MCP server was involved

### What to expect

These are best-effort targets from a small volunteer project, not a contractual
SLA:

| Stage | Target |
| ----- | ------ |
| Acknowledgement of your report | within 5 business days |
| Initial assessment (valid / not / need more info) | within 10 business days |
| Fix or documented mitigation for a confirmed issue | within 90 days |

We will keep you updated as the assessment progresses, credit you in the advisory
and the changelog unless you prefer otherwise, and coordinate public disclosure
with you once a fix is available.

## Scope

Jarvis is a local application. It has no server, no accounts, and no
Jarvis-operated backend — there is no hosted infrastructure to test against, and
nothing about this policy authorises testing against third parties.

### What this project's attack surface actually is

Stated plainly, so reports can be aimed at the right thing:

1. **It executes local applications and simulates input on behalf of a local
   LLM.** The tool registry exposes actions such as launching an app by name,
   closing an app, launching a Steam game, opening a URL, typing dictated text
   into the focused window, capturing the screen, reading and writing the
   clipboard, changing volume, and locking the screen. The LLM chooses which of
   these to call. Everything runs with the privileges of the signed-in user; no
   elevation is requested, and there is no sandbox between a tool call and the
   OS.

2. **It connects to external MCP servers that the user configures.** Servers
   added under Settings → Tools have their tools adapted into the same local
   registry as the built-ins, which means a server you add can offer tools that
   the LLM may then call. Jarvis controls the namespace (name collisions are
   rejected rather than silently overwriting a built-in) but does not otherwise
   vet a remote server's behaviour. Only add servers you trust.

3. **Deep Research fetches arbitrary web pages and feeds them to the LLM.** The
   research features query DuckDuckGo and then fetch the pages that come back.
   Page content reaching a model that can call tools is a prompt-injection
   surface; this is documented in the README's *Security & privacy model*
   section, and the deep-research prompts include explicit "ignore embedded
   instructions" guidance.

4. **Deep Research Ultra mode sends queries to third-party APIs, and only when
   the user opts in with their own keys.** With Ultra enabled, search queries go
   to `api.search.brave.com`, planning requests go to `api.groq.com`, and page
   extraction goes to `r.jina.ai`. Keys are resolved from the environment
   (`JARVIS_BRAVE_API_KEY`, `JARVIS_GROQ_API_KEY`) first and otherwise from
   `%APPDATA%\Jarvis\config.json`. Without keys, Jarvis falls back to the free
   local pipeline. Weather uses `api.open-meteo.com` (no key) plus a one-time
   `ipapi.co` geolocation if no coordinates are saved, and the first launch
   downloads speech-to-text weights from `huggingface.co`.

   No analytics, telemetry, or crash reporting is sent anywhere.

### In scope

- Anything that lets a **remote party** — a web page fetched during research, a
  malicious MCP server, a crafted model response — cause Jarvis to take an action
  the user did not ask for, or to reach beyond the documented tool surface
- Command or argument injection in the app-launch, Steam-launch, or URL-open
  paths
- Path traversal or arbitrary file write via notes, deep-research reports,
  screenshots, or the config loader
- Leakage of API keys or config contents into logs, into a network request, or to
  an unintended destination
- Anything that turns a research fetch, a config file, or an MCP handshake into
  code execution
- Failure of a documented privacy boundary — audio, transcripts, or notes leaving
  the machine when the user has not opted in

### Out of scope

Not because they do not matter, but because they are known, documented, or belong
to someone else:

- **The LLM calling a tool the user asked for.** Jarvis executing "open Spotify"
  is the product working, not a vulnerability.
- **Prompt injection in fetched pages, reported generically.** The risk is
  already documented in the README. A concrete bypass of a specific mitigation,
  or an injection chain that reaches a tool call the user never requested, *is*
  in scope — please include the chain.
- **Vulnerabilities in Ollama, in the language models themselves, in PySide6, or
  in an external MCP server.** Report those upstream. If Jarvis's *use* of one of
  them is what creates the exposure, that is in scope here.
- **SmartScreen warnings on unsigned builds.** Known and documented in the
  README; release binaries are not code-signed.
- **Attacks requiring the attacker to already have code execution or an
  interactive session on the machine.** Jarvis runs with the user's own
  privileges and stores config in the user's own profile; a local attacker who is
  already that user has already won.
- **Missing hardening that has no demonstrated impact** — a scanner finding
  without an exploit path is a discussion, not an advisory.

## Users: reducing your own risk

- Add only MCP servers you trust; their tools become callable by the LLM.
- Treat deep-research output like any AI-summarised article — verify important
  claims against the linked sources.
- Keep Jarvis in a context where the worst case of a wrong tool call (a tab
  opening, an app starting) is acceptable.
- API keys and MCP auth tokens in `%APPDATA%\Jarvis\config.json` are encrypted
  with Windows DPAPI and stored as `dpapi:<base64>`. DPAPI binds them to your
  Windows *user account*, so another account on the same PC cannot read them —
  but anything running as you still can, because it can simply ask DPAPI. Prefer
  the environment variables if that matters to you; Jarvis never writes an
  environment-supplied key to disk. Revoking a key in the provider's dashboard is
  sufficient — nothing is cached anywhere else.
- A `config.json` copied to another machine or another Windows account keeps its
  non-secret settings but the encrypted keys cannot be decrypted there. Jarvis
  logs a warning naming each setting and treats it as empty; re-enter the key in
  Settings.
- Non-Windows hosts have no DPAPI, so keys are stored in plaintext with one
  logged warning per process. Jarvis ships for Windows; this is the documented
  contributor-dev-machine fallback, not a supported deployment.
