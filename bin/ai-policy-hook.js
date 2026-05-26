#!/usr/bin/env node
"use strict";

const path = require("path");
const { spawnSync } = require("child_process");

const PACKAGE_ROOT = path.resolve(__dirname, "..");
const CLI = path.join(PACKAGE_ROOT, "bin", "ai-policy.js");

const AGENT_HOOKS = {
  codex: ["user_prompt_submit.py", "stop_refinement.py"],
  claude: ["claude_user_prompt_submit.py", "claude_stop_refinement.py"],
  opencode: ["opencode_user_prompt_submit.py", "opencode_stop_refinement.py"],
};
const HOOK_EVENTS = ["user-prompt-submit", "stop-refinement"];
const HOOKS = Object.fromEntries(
  Object.entries(AGENT_HOOKS).flatMap(([agent, scripts]) =>
    HOOK_EVENTS.map((event, index) => [
      `${agent}-${event}`,
      path.join(PACKAGE_ROOT, "hooks", scripts[index]),
    ]),
  ),
);

const hook = process.argv[2];
const script = HOOKS[hook];
if (!script) {
  console.error(`ai-policy-hook: expected one of: ${Object.keys(HOOKS).join(", ")}`);
  process.exit(1);
}

const result = spawnSync(process.execPath, [CLI, "runtime-python", script], {
  stdio: "inherit",
  env: {
    ...process.env,
    AI_POLICY_HOOK_SCRIPT: script,
  },
});

if (result.error) {
  console.error(`ai-policy-hook: ${result.error.message}`);
  process.exit(1);
}
process.exit(result.status === null ? 1 : result.status);
