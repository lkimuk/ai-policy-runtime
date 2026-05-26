![AI Policy Runtime banner](docs/assets/banner.png)

# AI Policy Runtime

Generate better AI code with task-aware policies for Codex, Claude Code, and OpenCode.

AI Policy Runtime gives your AI coding agent focused engineering rules for the
task at hand. It can guide implementation, review, refactoring, API design,
performance-sensitive work, Git workflows, CMake, Python, and post-task
refinement without making you rewrite prompts for every request.

The result is more consistent agent behavior across a workspace.

## How it works

AI Policy Runtime sits between the user's task and the coding agent. It reads
the current request, workspace signals, selected packs, and Skill DSL assets,
then compiles the relevant policy into a small task-specific rule set.

The agent does not receive the full Skill library. It receives the generated
Effective Rules for the current task, rendered as prompt text and structured
JSON. Those same rules can also be used after the agent acts for review,
refinement, and repair.

## The Basic Workflow

```text
User Task
  ↓
Task Analyzer
  ↓
Structured Context
  ↓
Policy Compilation
  - activate Skills and selected Packs
  - apply dependencies and when / unless filters
  - resolve conflicts
  - reduce rules
  ↓
Effective Rules
  - effective-prompt.md
  - effective-rules.json
  ↓
Agent
  ↓
Post refinement
  - review output
  - apply refinement packs
  - verify / repair when needed
  ↓
Agent
  ↓
Work done
```

1. The user sends a task to Codex, Claude Code, or OpenCode.
2. The runtime analyzes the task and builds structured context.
3. Matching Skills, selected Packs, dependencies, and rule conditions are
   evaluated.
4. Conflicts are resolved and redundant rules are reduced.
5. Effective Rules are written to the workspace and injected into the agent.
6. The agent works with the task-specific rules.
7. Optional post-task refinement reviews the output and can run a constrained
   second pass.

## What It Does

- Detects the task type and applies focused coding rules.
- Shares one workspace configuration across Codex, Claude Code, and OpenCode.
- Uses embeddings for stronger multilingual task matching.
- Can run a post-task refinement pass before the agent finishes.
- Shows the exact Effective Rules used for the latest prompt.

C++ currently has the deepest coverage, including ownership, lifetime, bounds
safety, API design, modern C++, and low-latency work. Python, CMake, Git, and
general refinement packs are also included.

## Install

### VS Code

Install the **AI Policy Runtime** extension, then open its side bar in your
workspace.

Use it with the Codex, Claude Code, or OpenCode extension you already use. Enable the
runtime, select the agent, choose policy packs, and configure embeddings if you
want semantic task matching.

For Codex, trust the generated workspace hooks when Codex asks. Until the hooks
are trusted, Codex will not run them, so AI Policy Runtime will appear enabled
but no rules will be injected.

After updating the VS Code extension, reload the window and run
**AI Policy Runtime: Validate Runtime**. Validation refreshes workspace hook
files when they still point at an older extension runtime.

### Command Line

```powershell
npm install -g ai-policy-runtime
ai-policy doctor
```

Configure a project for Codex:

```powershell
ai-policy configure codex --root D:\work\project
```

Configure a project for Claude Code:

```powershell
ai-policy configure claude --root D:\work\project
```

Configure a project for OpenCode:

```powershell
ai-policy configure opencode --root D:\work\project
```

After updating the npm package, rerun `ai-policy configure codex` or
`ai-policy configure claude` or `ai-policy configure opencode` for each
workspace that should use the new runtime. The command refreshes
`.policy/config.json` and agent hook/plugin paths to the current installed
package.

## Embeddings

AI Policy Runtime uses embeddings for product-quality multilingual task
matching. You choose the provider:

- **Auto**: use a configured OpenAI-compatible endpoint, otherwise use a
  configured local model when available.
- **OpenAI-compatible**: use an endpoint such as OpenAI, OpenRouter, or another
  `/v1/embeddings` service.
- **Local**: use a sentence-transformers model path that you provide.

The package does not include or download a local model by default.

If you want the CLI to use a remote embedding model immediately, configure an
OpenAI-compatible endpoint through environment variables:

```powershell
$env:AI_POLICY_EMBEDDING_PROVIDER = "openai-compatible"
$env:AI_POLICY_EMBEDDING_BASE_URL = "https://openrouter.ai/api/v1"
$env:AI_POLICY_EMBEDDING_API_KEY = "<your-api-key>"
$env:AI_POLICY_EMBEDDING_MODEL = "<embedding-model>"
```

Configure a local model from the CLI:

```powershell
ai-policy embedding configure --root D:\work\project --provider local --model D:\models\paraphrase-multilingual-MiniLM-L12-v2
```

Or download the recommended default model:

```powershell
ai-policy embedding configure --root D:\work\project --provider local --install
```

Check the current embedding setup:

```powershell
ai-policy embedding status --root D:\work\project
ai-policy embedding test --root D:\work\project
```

Clean AI Policy Runtime workspace integration before uninstalling or when
resetting a project:

```powershell
ai-policy cleanup --root D:\work\project
```

Cleanup removes AI Policy Codex/Claude/OpenCode integration entries,
`.policy/config.json`, and generated `.policy/current` state. It keeps caches,
local models, and unrelated agent settings.

## Common Commands

```powershell
ai-policy status --root D:\work\project
ai-policy resolve --pack cpp.safe_generation "Implement this C++ API"
ai-policy explain "Review this change for ownership and lifetime risks"
ai-policy validate
```

## Policy Packs

Included packs:

- `cpp.safe_generation`
- `cpp.low_latency`
- `cpp.code_review`
- `cpp.library_api_design`
- `cpp.modernization`
- `cpp.production_refinement`
- `python.best_practices`
- `python.production_refinement`
- `cmake.best_practices`
- `cmake.production_refinement`
- `git.best_practices`
- `generic.production_refinement`

## Workspace Files

AI Policy Runtime stores transparent project state in workspace files:

- `.policy/config.json`
- `.policy/current/effective-prompt.md`
- `.policy/current/agent-hook-state.json`
- `.codex/hooks.json` and `.codex/config.toml` when Codex is enabled
- `.claude/settings.local.json` when Claude Code is enabled
- `opencode.json` and `.opencode/plugins/ai-policy-runtime.js` when OpenCode is enabled

These files make the active policy visible and reproducible. Review them before
committing workspace-specific settings.

## Notes

- AI Policy Runtime is not an LLM. It prepares task-aware policy context for AI
  coding agents.
- Local models are optional and user-configured.
- Remote embedding requests are only used when you configure a remote provider
  or credentials.
- See [docs/usage.md](docs/usage.md) for CLI and advanced setup details.

## Philosophy

Large rule libraries are useful for humans and tools, but they are noisy when
sent directly to an agent. AI Policy Runtime treats policy as something to
compile, not paste.

The goal is to preserve useful complexity in the Skill library while giving the
agent only the rules that matter for the current task. Packs expand candidate
Skills, but conditions, conflict resolution, and reduction decide what becomes
effective.

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
