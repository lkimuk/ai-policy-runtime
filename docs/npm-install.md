# NPM Install Guide

This is the end-user installation path. Users do not need to clone the source
repository or pass a checkout path as `--plugin-root`.

## Install

```powershell
npm install -g ai-policy-runtime
```

The package installs one command:

```text
ai-policy
```

## Requirements

- Node.js 18 or newer.
- Python 3.10 or newer.

The command creates and reuses a managed Python virtual environment. By default
the cache location is:

```text
Windows: %LOCALAPPDATA%\ai-policy-runtime\venv
macOS/Linux: ~/.cache/ai-policy-runtime/venv
```

Override the Python interpreter when needed:

```powershell
$env:AI_POLICY_PYTHON="C:\Path\To\Python\python.exe"
```

Override the managed runtime home when needed:

```powershell
$env:AI_POLICY_HOME="D:\tools\ai-policy-runtime-state"
```

## Verify Installation

```powershell
ai-policy doctor
```

Expected output is JSON with `ok: true`, a Python version, and resource checks
for the installed package:

```json
{
  "ok": true,
  "checks": {
    "claudePlugin": true,
    "claudeHooks": true,
    "codexPlugin": true,
    "codexHooks": true,
    "opencodeConfigure": true,
    "skills": true,
    "packs": true
  }
}
```

If the managed Python environment becomes stale or broken:

```powershell
ai-policy runtime rebuild
```

## Configure Claude Desktop / Claude for Windows

Configure a workspace:

```powershell
ai-policy configure claude --root D:\work\target-project
```

This writes:

```text
D:\work\target-project\.policy\config.json
D:\work\target-project\.claude\settings.local.json
```

It enables:

- AI Policy Runtime for the Claude agent.
- No policy packs by default; choose packs explicitly for each workspace.
- The installed NPM package as the Claude plugin marketplace root.
- `ai-policy-runtime@ai-policy-runtime` in Claude settings.

Then open Claude Desktop / Claude for Windows, switch to the Code tab, and use
the configured local workspace.

## Check Status

```powershell
ai-policy status --root D:\work\target-project
```

`status` is read-only. It does not create `.policy` or `.claude` files.

For Codex-specific policy status:

```powershell
ai-policy status --agent codex --root D:\work\target-project
```

For OpenCode-specific policy status:

```powershell
ai-policy status --agent opencode --root D:\work\target-project
```

## Use Remote Embeddings

Downloading a local model can take time. To make the CLI usable immediately,
configure an OpenAI-compatible remote embedding provider with environment
variables:

```powershell
$env:AI_POLICY_EMBEDDING_PROVIDER = "openai-compatible"
$env:AI_POLICY_EMBEDDING_BASE_URL = "https://openrouter.ai/api/v1"
$env:AI_POLICY_EMBEDDING_API_KEY = "<your-api-key>"
$env:AI_POLICY_EMBEDDING_MODEL = "<embedding-model>"
```

Then verify the provider:

```powershell
ai-policy embedding status --root D:\work\target-project
ai-policy embedding test --root D:\work\target-project
```

## Configure Codex

Configure the shared project policy for Codex:

```powershell
ai-policy configure codex --root D:\work\target-project
```

This writes:

```text
D:\work\target-project\.policy\config.json
D:\work\target-project\.codex\hooks.json
D:\work\target-project\.codex\config.toml
```

It enables the `codex` agent, leaves packs empty when no packs are configured,
records the installed NPM package as `policyRoot`, and
enables project-local Codex hooks for bare `codex` CLI usage. It does not write
`.claude/settings.local.json`.

After updating `ai-policy-runtime`, rerun this command for each Codex workspace
that should use the new package. Reconfiguration replaces stale `policyRoot`
and hook runner paths with the current installed package.

Disable only Codex in the shared policy:

```powershell
ai-policy configure codex --root D:\work\target-project --disable
```

## Configure OpenCode

Configure the shared project policy for OpenCode:

```powershell
ai-policy configure opencode --root D:\work\target-project
```

This writes:

```text
D:\work\target-project\.policy\config.json
D:\work\target-project\opencode.json
D:\work\target-project\.opencode\plugins\ai-policy-runtime.js
```

It enables the `opencode` agent, leaves packs empty when no packs are configured,
records the installed NPM package as `policyRoot`, adds `AGENTS.md` to OpenCode
instructions, and installs a project-local OpenCode plugin. Re-run the command
after updating `ai-policy-runtime` so the plugin points at the current installed
package.

For post-refinement smoke tests, enable `postRefine` in `.policy/config.json`,
run an OpenCode task, then inspect:

```text
D:\work\target-project\.policy\current\opencode-plugin-state.json
D:\work\target-project\.policy\current\opencode-post-refine-prompt.md
```

Disable only OpenCode in the shared policy and remove AI Policy's OpenCode
instruction/plugin entries:

```powershell
ai-policy configure opencode --root D:\work\target-project --disable
```

## Toggle Runtime and Plugin

Disable both the runtime and Claude plugin for a workspace:

```powershell
ai-policy disable --root D:\work\target-project
```

Toggle only the Claude plugin setting:

```powershell
ai-policy plugin enable --root D:\work\target-project
ai-policy plugin disable --root D:\work\target-project
```

Plugin-only toggles update Claude settings without creating or changing
`.policy/config.json`.

After updating `ai-policy-runtime`, rerun `ai-policy configure claude --root
<project>` for each Claude Code workspace that should use the new package.
Reconfiguration refreshes both `.policy/config.json` and Claude plugin settings.

## Clean Workspace Configuration

Before uninstalling or when resetting a project, remove AI Policy Runtime
workspace integration state:

```powershell
ai-policy cleanup --root D:\work\target-project
```

Cleanup removes AI Policy entries from Codex, Claude, and OpenCode settings,
deletes `.policy/config.json`, and removes generated `.policy/current` state. It leaves
caches, local models, and unrelated agent settings in place. Use
`--keep-current` if you want to preserve the generated current-state files for
debugging.

## Post-Task Refinement

Enable one extra Stop-hook refinement pass after applicable coding tasks:

```powershell
ai-policy post-refine standard --root D:\work\target-project
```

Disable it:

```powershell
ai-policy post-refine off --root D:\work\target-project
```

`post-refine off` only disables the second pass. On a new project it does not
enable the runtime or Claude plugin.

## Runtime Commands

The NPM command also forwards runtime commands and automatically uses the
installed package as the policy asset root:

```powershell
ai-policy explain "帮我写一个 C++20 低延迟队列"
ai-policy resolve --pack cpp.low_latency "帮我写一个 C++20 低延迟队列"
ai-policy validate
ai-policy schema skill
```

For lower-level Python CLI access:

```powershell
ai-policy runtime resolve "帮我写一个 C++20 低延迟队列"
```

## Development Override

When developing from a source checkout, point Claude configuration at that
checkout explicitly:

```powershell
ai-policy configure claude `
  --root D:\work\target-project `
  --plugin-root D:\MilesLi\ai-policy-runtime
```

Most users should not need this.

## Troubleshooting

Run:

```powershell
ai-policy doctor
```

Common fixes:

- Python not found: install Python 3.10+ or set `AI_POLICY_PYTHON`.
- Stale runtime: run `ai-policy runtime rebuild`.
- Claude plugin not visible: rerun `ai-policy configure claude --root <project>`
  and restart the Claude Desktop Code session.
- Wrong workspace: check `ai-policy status --root <project>`.
