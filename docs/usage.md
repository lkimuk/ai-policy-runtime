# Usage Guide

This guide keeps the operational details for AI Policy Runtime. The README is
the project entry point; this file is the reference for day-to-day commands and
integration knobs.

## Install

For normal use, install the packaged command:

```powershell
npm install -g ai-policy-runtime
```

This exposes the `ai-policy` command.
The command uses the installed package as its policy and plugin root, so users
do not need to clone this repository.

See [NPM Install Guide](npm-install.md) for the end-user install flow,
Claude Desktop setup, runtime diagnostics, and troubleshooting.

## Task Analysis

Task analysis uses:

```text
exact matching for precise facts
deterministic project-context scanning for omitted repository facts
deterministic evidence resolution for final TaskContext
optional embedding semantic recall for rephrased intent
```

The default installation has no model dependency and does not download models.
It uses deterministic task analysis, project-context scanning, and the best
configured semantic provider:

```text
OpenAI-compatible /v1/embeddings endpoint
local sentence-transformers model
```

OpenAI-compatible is the preferred default provider shape. Use an
OpenAI-compatible embedding endpoint when you want strong multilingual
semantic matching without asking users to download a local model:

```powershell
$env:AI_POLICY_EMBEDDING_API_KEY="<key>"
$env:AI_POLICY_EMBEDDING_MODEL="text-embedding-3-small"
```

`AI_POLICY_EMBEDDING_PROVIDER` is optional for the common case. The runtime
automatically uses the OpenAI-compatible provider when
`AI_POLICY_EMBEDDING_API_KEY`, `OPENAI_API_KEY`, or
`AI_POLICY_EMBEDDING_BASE_URL` is set. Keep the provider variable for explicit
advanced overrides:

```powershell
$env:AI_POLICY_EMBEDDING_PROVIDER="openai-compatible" # force remote embeddings
$env:AI_POLICY_EMBEDDING_PROVIDER="local"             # force sentence-transformers
```

For OpenAI-compatible gateways, set the endpoint explicitly:

```powershell
$env:AI_POLICY_EMBEDDING_BASE_URL="https://gateway.example.com/v1"
```

If `AI_POLICY_EMBEDDING_MODEL` is omitted for a remote endpoint, the runtime
uses `text-embedding-3-small`.

If no remote provider is configured, the runtime tries the default local model
path shown below when that model has already been installed. If neither provider
is available, the runtime reports a configuration error because semantic
embedding recall is required for task analysis.

To verify which semantic path works in your environment, run:

```powershell
ai-policy explain "帮我写一个 C++20 低延迟队列"
```

The output should include structured context such as `hot_path: true`,
`scenario: low_latency_queue`, and semantic evidence whose source contains an
English skill phrase such as `semantic:low latency queue implementation`.

You can force each provider while testing:

```powershell
$env:AI_POLICY_EMBEDDING_PROVIDER="openai-compatible"
ai-policy explain "帮我写一个 C++20 低延迟队列"

$env:AI_POLICY_EMBEDDING_PROVIDER="local"
ai-policy explain "帮我写一个 C++20 低延迟队列"
```

In automatic mode, leave `AI_POLICY_EMBEDDING_PROVIDER` unset. Automatic mode
uses the remote OpenAI-compatible endpoint when endpoint credentials are
configured, otherwise the local model when installed. When a provider is forced
explicitly, configuration or endpoint errors are reported instead of silently
falling back to a weaker provider.

Local transformer-based semantic recall is optional. Install the optional extra
when you want to use a local sentence-transformers model:

```powershell
pip install "ai-policy-runtime[semantic]"
```

Then install the recommended local model into the policy asset root:

```powershell
ai-policy model install
```

Inspect local model status with:

```powershell
ai-policy model list
```

The default local model path is:

```text
models/paraphrase-multilingual-MiniLM-L12-v2
```

You can also point the runtime at another local sentence-transformers model:

```powershell
$env:AI_POLICY_EMBEDDING_MODEL="D:\path\to\model"
```

When `policy_root/models/paraphrase-multilingual-MiniLM-L12-v2` exists, the
high-level runtime uses it automatically. If no local transformer model is
configured, the runtime reports a configuration error rather than downloading
anything implicitly.

Semantic index vectors are cached under:

```text
.policy/cache/semantic-index/
```

Explain Task Analysis without resolving rules:

```powershell
ai-policy explain "写一个 C++20 数据通道，主循环里不能有分配和阻塞，尾延迟要稳"
```

The runtime scans project files before resolving a task. High-confidence facts
from build metadata can fill in details the user did not repeat in the prompt,
such as C++ standard, build system, and primary language. Supported sources
include:

```text
.policy/project.yaml
compile_commands.json
CMakeLists.txt
CMakePresets.json-compatible CMake files through CMakeLists scanning
pyproject.toml, Cargo.toml, package.json, go.mod
vcpkg.json, conanfile.txt, conanfile.py
source/header file layout
README.md weak tags
```

Facts are written with provenance to:

```text
.policy/current/project-context.json
.policy/current/trace.json
```

Manual project overrides can be declared in `.policy/project.yaml`:

```yaml
domain: cpp
build_system: cmake
context:
  standard: 20
  selected_standard_is_known: true
  hot_path: true
tags:
  - low_latency
```

Inspect the current resolved state:

```powershell
ai-policy inspect
```

Print bundled schemas:

```powershell
ai-policy schema skill
ai-policy schema pack
ai-policy schema effective-rules
```

List or clear semantic-index cache entries:

```powershell
ai-policy cache list
ai-policy cache clear
```

## Resolve a Task

```powershell
ai-policy resolve "帮我写一个 C++20 低延迟队列"
ai-policy resolve --pack cpp.low_latency "帮我写一个 C++20 低延迟队列"
ai-policy resolve --pack git.best_practices "Prepare a git commit message for the staged diff."
ai-policy resolve --pack cmake.best_practices "Modernize this CMakeLists.txt to use target-based CMake."
```

`resolve` prints the final agent-facing prompt by default. For explicitness in
test scripts, you can also pass `--format prompt`:

```powershell
ai-policy resolve --format prompt "帮我写一个 C++20 低延迟队列"
ai-policy resolve --format prompt --pack cpp.low_latency "帮我写一个 C++20 低延迟队列"
```

Use `--format json` only when a tool needs structured command output:

```powershell
ai-policy resolve --format json "帮我写一个 C++20 低延迟队列"
```

This writes the current task state to `.policy/current/`:

```text
task-context.json
effective-rules.json
effective-rules.yaml
effective-prompt.md
trace.json
```

## Validate Skills

```powershell
ai-policy validate
```

Validation combines bundled JSON Schema checks from `schemas/` with semantic
runtime checks such as dependency and pack-reference validation.

## Multiple Skill / Pack Directories

The runtime can load skills and packs from the bundled directory plus any
number of additional directories — useful for layering project-specific or
team-specific policy on top of the default library without forking the
repository.

Pass extra directories on the command line (repeatable):

```powershell
ai-policy resolve "fix the deadlock" `
    --extra-skills C:\team\policy\skills `
    --extra-packs  C:\team\policy\packs
```

Or persist them in `.policy/config.json`:

```json
{
  "extraSkillsDirs": ["custom/skills", "../shared-policy/skills"],
  "extraPacksDirs":  ["custom/packs"],
  "onDuplicate": "first_wins"
}
```

Paths are resolved relative to the policy root (or absolute paths are honored
as-is). The bundled `skills_dir` / `packs_dir` always loads first; extras are
appended in the order given. CLI flags merge with the config arrays — CLI
entries come first, then config entries, with duplicates removed.

When the same `skill_id` (or `pack_id`) appears in more than one directory,
`onDuplicate` controls the merge:

| Value         | Behavior                                                        |
| ------------- | --------------------------------------------------------------- |
| `error`       | Raise on duplicate (default — preserves the strict behavior).   |
| `first_wins`  | Keep the version loaded earliest; later duplicates are ignored. |
| `last_wins`   | Replace earlier definitions with later ones.                    |

Equivalent CLI flag: `--on-duplicate {error,first_wins,last_wins}`.

Backward compatibility: if no extras and no `onDuplicate` are configured, the
runtime behaves exactly as before — a single skills/packs directory with
strict duplicate detection.

## Inject Effective Rules

```powershell
ai-policy inject --target custom
ai-policy inject --target codex
ai-policy inject --target claude
ai-policy inject --target opencode
```

`codex` and `opencode` update the generated block in `AGENTS.md`; `claude`
updates `CLAUDE.md`; `custom` writes `.policy/current/injected-prompt.md`.

## Run Codex with Effective Rules

Use `policy-codex` when installed as a package:

```powershell
policy-codex --pack cpp.low_latency "帮我写一个 C++20 低延迟队列"
```

The wrapper performs:

```text
resolve -> inject AGENTS.md -> codex "<task>"
```

Add `--post-refine` when the first successful agent run should be followed by a
second, behavior-preserving production refinement pass:

```powershell
policy-codex --pack cpp.low_latency --post-refine "帮我写一个 C++20 低延迟队列"
```

Use an explicit mode when the workflow needs to be lighter or stricter:

```powershell
policy-codex --pack cpp.low_latency --post-refine-mode light "帮我写一个 C++20 低延迟队列"
policy-codex --pack cpp.low_latency --post-refine-mode strict --verify-target src "帮我写一个 C++20 低延迟队列"
```

Post-refinement modes are:

```text
off       preserve existing behavior
light     resolve and inject refinement context only
standard  run a second agent pass after a successful first pass
strict    run a second agent pass and pair it with --verify-target for release-quality checks
```

For dry runs that only refresh `AGENTS.md`:

```powershell
policy-codex --pack cpp.low_latency --no-exec "帮我写一个 C++20 低延迟队列"
```

To enhance a different project with this policy repository:

```powershell
policy-codex --root D:\work\target-project --policy-root D:\MilesLi\ai-policy-runtime --pack cpp.low_latency "帮我写一个低延迟队列"
```

Pass Codex CLI options before the task with repeated `--codex-arg`:

```powershell
policy-codex --pack cpp.low_latency --codex-arg "--approval-mode" --codex-arg "never" "帮我写一个 C++20 低延迟队列"
```

## Run OpenCode with Effective Rules

Use `policy-opencode` when installed as a package:

```powershell
policy-opencode --pack cpp.low_latency "帮我写一个 C++20 低延迟队列"
```

The wrapper performs:

```text
resolve -> inject AGENTS.md -> opencode run "<task>"
```

Pass OpenCode CLI options before the task with repeated `--opencode-arg`:

```powershell
policy-opencode --pack cpp.low_latency --opencode-arg "--model" --opencode-arg "anthropic/claude-sonnet-4" "帮我写一个 C++20 低延迟队列"
```

Configure a project for normal OpenCode usage:

```powershell
ai-policy configure opencode --root D:\work\target-project
ai-policy status --agent opencode --root D:\work\target-project
```

This enables the `opencode` agent in `.policy/config.json`, records the
installed package as `policyRoot`, adds `AGENTS.md` to `opencode.json`
instructions, and installs `.opencode/plugins/ai-policy-runtime.js`. OpenCode's
plugin API is event-based; the plugin dynamically injects Effective Rules when a
prompt event exposes prompt text and otherwise falls back to `AGENTS.md`.

When `postRefine` is enabled, the OpenCode plugin prepares a continuation prompt
on `session.idle` and writes release-testable state under:

```text
.policy/current/opencode-plugin-state.json
.policy/current/opencode-post-refine-prompt.md
```

Use `ai-policy status --agent opencode --root <project>` to check whether these
files were produced during a manual OpenCode session.

## Use as a Codex Plugin

This repository is also shaped as a Codex plugin. The plugin registers a
`UserPromptSubmit` hook that resolves the current user prompt into Effective
Rules and injects the rendered rules as Codex `additionalContext`. It also
registers a `Stop` hook that can ask Codex to continue once with a
post-refinement prompt before the turn ends.

Plugin files:

```text
.codex-plugin/plugin.json
hooks/codex-hooks.json
hooks/user_prompt_submit.py
hooks/stop_refinement.py
```

The hook bootstraps the Python package from this repository on first use:

```text
python -m pip install -e <plugin-root>
```

That installs the runtime dependencies declared in `pyproject.toml`, including
`PyYAML` and `jsonschema`. Set `AI_POLICY_AUTO_INSTALL=0` to disable this
automatic bootstrap and manage dependencies yourself.

Configuration sources:

| Entry point | Primary configuration | Notes |
| --- | --- | --- |
| VS Code Extension | VS Code workspace settings, synced to `.policy/config.json` | Friendly UI for workspace hooks, agents, packs, embeddings, and post-refinement. |
| Command-line hooks | `.policy/config.json` | Use `ai-policy embedding configure ...`; environment variables are useful for CI, secrets, and temporary overrides. |
| Python Runtime | `RuntimeConfig` constructor arguments | Pass embedding settings to `RuntimeConfig.from_values(...)`, or rely on environment variables/default local model. |

Configure embeddings for command-line hooks:

```powershell
ai-policy embedding configure --root D:\work\target-project --provider local
ai-policy embedding configure --root D:\work\target-project --provider openai-compatible --base-url https://api.openai.com/v1 --api-key <key> --model text-embedding-3-small
ai-policy embedding status --root D:\work\target-project
```

Configure embeddings for embedded Python Runtime code:

```python
runtime = PolicyRuntime(RuntimeConfig.from_values(
    root="D:/work/target-project",
    policy_root="D:/MilesLi/ai-policy-runtime",
    embedding_provider="openai-compatible",
    embedding_api_key="<key>",
    embedding_model="text-embedding-3-small",
))
```

Useful environment variables:

```text
AI_POLICY_ROOT=<path-to-policy-runtime>
AI_POLICY_PACKS=cpp.low_latency,cpp.safe_generation
AI_POLICY_AUTO_INSTALL=0
AI_POLICY_EMBEDDING_PROVIDER=openai-compatible
AI_POLICY_EMBEDDING_BASE_URL=https://api.openai.com/v1
AI_POLICY_EMBEDDING_API_KEY=<key>
AI_POLICY_EMBEDDING_MODEL=text-embedding-3-small
AI_POLICY_EMBEDDING_TIMEOUT=30
AI_POLICY_POST_REFINE=standard
AI_POLICY_POST_REFINE_PACKS=cpp.production_refinement
AI_POLICY_VERIFY_TARGET=src
```

The hook also reads project-local configuration from:

```text
.policy/config.json
```

This is the preferred control surface for editor integrations:

```json
{
  "enabled": true,
  "agents": ["codex"],
  "packs": ["cpp.safe_generation", "cpp.low_latency"],
  "autoInstall": true,
  "embeddingProvider": "openai-compatible",
  "embeddingBaseUrl": "https://api.openai.com/v1",
  "embeddingApiKey": "<key>",
  "embeddingModel": "text-embedding-3-small",
  "embeddingTimeout": "30",
  "postRefine": "standard",
  "postRefinePacks": ["cpp.production_refinement"],
  "verifyTarget": "src"
}
```

Use `"agents": ["codex", "claude", "opencode"]` when the same workspace should
be active for multiple supported agent integrations. Project embedding settings
take precedence when present so editor-saved provider choices are stable. Environment variables can
still override workspace-independent controls such as `AI_POLICY_ROOT`,
`AI_POLICY_PACKS`, and `AI_POLICY_VERIFY_TARGET`.

`postRefine` accepts `off`, `light`, `standard`, or `strict`. When enabled, the
`Stop` hook uses the agent continuation mechanism once per turn: it returns
`decision: block` with a refinement prompt, then allows the next stop event when
the agent reports that the stop hook is already active. This prevents an
infinite refinement loop.

## Configure Agents from VS Code

An agent-focused VS Code extension is included under:

```text
vscode-extension/
```

The extension does not reimplement the runtime. It writes `.policy/config.json`
for the current workspace and lets Codex, Claude Code, and OpenCode integrations inject
Effective Rules on each prompt. The configuration view includes target-agent
selection and a one-click Post-refinement switch that enables the `Stop`
continuation workflow by writing `postRefine` and `postRefinePacks`.

Available commands:

```text
AI Policy Runtime: Enable
AI Policy Runtime: Disable
AI Policy Runtime: Enable Post-Task Refinement
AI Policy Runtime: Configure Packs
AI Policy Runtime: Show Status
AI Policy Runtime: Show Effective Rules
AI Policy Runtime: Validate Runtime
```

During development, build and install the extension with:

```powershell
cd D:\MilesLi\ai-policy-runtime\vscode-extension
npm ci
npm run package
$vsix = Get-ChildItem .\ai-policy-runtime-*.vsix | Sort-Object LastWriteTime -Descending | Select-Object -First 1
& "$env:LOCALAPPDATA\Programs\Microsoft VS Code\bin\code.cmd" --install-extension $vsix.FullName --force
```

Reload VS Code after reinstalling:

```text
Developer: Reload Window
```

After updating the extension, run **AI Policy Runtime: Validate Runtime** in
each workspace that already uses the runtime. Validation refreshes generated
workspace hook files if they still point at an older extension install.

For local development in this repository, project `.codex/config.toml` can point
Codex at the same hook implementation used by the plugin.

For an installed NPM package, configure the shared project policy for Codex:

```powershell
ai-policy configure codex --root D:\work\target-project
ai-policy status --agent codex --root D:\work\target-project
```

This enables the `codex` agent in `.policy/config.json`, records the installed
package as `policyRoot`, writes project-local Codex hook settings, and leaves
Claude settings untouched:

```text
D:\work\target-project\.policy\config.json
D:\work\target-project\.codex\hooks.json
D:\work\target-project\.codex\config.toml
```

The generated `.codex/hooks.json` invokes the installed package's hook runner,
so users can run the normal `codex` CLI in that project without using
`policy-codex`.

After updating the npm package, rerun `ai-policy configure codex --root
D:\work\target-project` for each Codex workspace that should use the new
runtime. Reconfiguration replaces stale `policyRoot` and hook runner paths.

After publishing this repository to GitHub, users can add it as a Codex plugin
marketplace:

```powershell
codex plugin marketplace add lkimuk/ai-policy-runtime
```

Then install **AI Policy Runtime** from Codex:

```text
/plugins
```

The marketplace entry is declared in:

```text
.agents/plugins/marketplace.json
```

## Run Claude Code with Effective Rules

Use `policy-claude` when installed as a package:

```powershell
policy-claude --pack cpp.low_latency "帮我写一个 C++20 低延迟队列"
```

The wrapper performs:

```text
resolve -> inject CLAUDE.md -> claude "<task>"
```

The shared post-refinement flags also work with Claude Code:

```powershell
policy-claude --pack cpp.low_latency --post-refine "帮我写一个 C++20 低延迟队列"
```

For dry runs that only refresh `CLAUDE.md`:

```powershell
policy-claude --pack cpp.low_latency --no-exec "帮我写一个 C++20 低延迟队列"
```

To enhance a different project with this policy repository:

```powershell
policy-claude --root D:\work\target-project --policy-root D:\MilesLi\ai-policy-runtime --pack cpp.low_latency "帮我写一个低延迟队列"
```

Pass Claude Code CLI options before the task with repeated `--claude-arg`:

```powershell
policy-claude --pack cpp.low_latency --claude-arg "--dangerously-skip-permissions" "帮我写一个 C++20 低延迟队列"
```

## Use as a Claude Code Plugin

This repository is also shaped as a Claude Code plugin following Claude Code's
official plugin and hook layout. Claude Code loads the standard
`hooks/hooks.json` file, while the hook scripts reuse the same runtime logic as
the Codex integration.

Plugin files:

```text
.claude-plugin/plugin.json
hooks/hooks.json
hooks/claude_user_prompt_submit.py
hooks/claude_stop_refinement.py
```

Enable Claude Code in the workspace config:

```json
{
  "enabled": true,
  "agents": ["claude"],
  "packs": ["cpp.safe_generation"]
}
```

Use `"agents": ["codex", "claude"]` when the same workspace should be active
for both supported plugin integrations, or include `"opencode"` for OpenCode
workspaces.

## Use with Claude for Windows

Claude for Windows exposes Claude Code through the desktop Code tab. The plugin
integration above is the path for local and SSH Code sessions. Configure the
Claude Desktop client from Claude's plugin UI, not from the VS Code extension.

You can preconfigure a workspace with the helper script:

```powershell
ai-policy configure claude --root D:\work\target-project
```

Enable one-pass post-refinement during the same setup when the desktop client
should ask Claude to continue once after an applicable coding task:

```powershell
ai-policy configure claude --root D:\work\target-project --post-refine standard
```

Query or change the same workspace later without editing JSON:

```powershell
ai-policy status --root D:\work\target-project
ai-policy disable --root D:\work\target-project
ai-policy plugin enable --root D:\work\target-project
ai-policy plugin disable --root D:\work\target-project
ai-policy post-refine off --root D:\work\target-project
```

Inspect the installed runtime and managed Python environment:

```powershell
ai-policy doctor
ai-policy runtime rebuild
```

Depending on the command, the script updates:

```text
D:\work\target-project\.policy\config.json
D:\work\target-project\.claude\settings.local.json
```

The base setup command enables `agents: ["claude"]`, registers the installed
package as a local Claude plugin marketplace, and enables
`ai-policy-runtime@ai-policy-runtime` for that workspace. Plugin-only toggles
only update Claude settings. `--post-refine standard` writes `postRefine` and
`postRefinePacks` so the Stop hook can perform the second pass; `--post-refine
off` only disables that second pass. Use `--scope project` to write
`.claude/settings.json` instead, or `--scope user` to write
`%USERPROFILE%\.claude\settings.json`.

After updating the npm package, rerun `ai-policy configure claude --root
D:\work\target-project` for each Claude workspace that should use the new
runtime. Reconfiguration replaces stale plugin marketplace paths.

When working from a source checkout, the development helper is still available:

```powershell
python tools/configure_claude_desktop.py `
  --root D:\work\target-project `
  --plugin-root D:\MilesLi\ai-policy-runtime
```

In Claude for Windows:

1. Switch to the Code tab.
2. Start or open a local project session.
3. Open the prompt `+` menu.
4. Choose Plugins, add the AI Policy Runtime plugin root, and enable it.

Claude Desktop shares Claude Code configuration and supports plugins for local
and SSH sessions. Remote sessions do not load plugins, so use `CLAUDE.md`
injection or `policy-claude` for those workflows.

## Verify Outputs

```powershell
ai-policy verify --target path\to\output.cpp
```

The verifier writes `.policy/current/violations.json` and exits non-zero when
violations are found.

Verification is pluggable. The default verifier checks text-searchable
`forbid` rules, and additional deterministic verifiers can implement the
`RuleVerifier` protocol.

## Run the MVP Workflow

```powershell
ai-policy run --pack cpp.low_latency --agent custom "帮我写一个 C++20 低延迟队列"
```

This performs:

```text
resolve -> inject -> optional verify
```

## Notes

- JSON skill files work with the Python standard library.
- YAML skill files are supported when `PyYAML` is installed.
- This version does not call an LLM. It produces Effective Rules that can be
  injected into an LLM/Agent runtime later.
