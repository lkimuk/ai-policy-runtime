const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const PACKAGE_ROOT = "__AI_POLICY_RUNTIME_ROOT__";
const HOOK = path.join(PACKAGE_ROOT, "bin", "ai-policy-hook.js");
const SERVICE = "ai-policy-runtime";
const OPENCODE_STATE = path.join(".policy", "current", "opencode-plugin-state.json");
const OPENCODE_POST_REFINE_PROMPT = path.join(
  ".policy",
  "current",
  "opencode-post-refine-prompt.md",
);

function runHook(name, payload) {
  const result = spawnSync(process.execPath, [HOOK, name], {
    input: JSON.stringify(payload),
    encoding: "utf8",
    env: {
      ...process.env,
      AI_POLICY_AGENT: "opencode",
    },
  });
  if (result.error) {
    return { ok: false, error: result.error.message };
  }
  const output = result.stdout.trim();
  if (!output) {
    return { ok: result.status === 0, response: {} };
  }
  try {
    return { ok: result.status === 0, response: JSON.parse(output), stderr: result.stderr };
  } catch (error) {
    return { ok: false, error: error.message, stdout: output, stderr: result.stderr };
  }
}

function promptFrom(input, output) {
  return (
    input?.prompt ??
    input?.text ??
    input?.message ??
    output?.prompt ??
    output?.text ??
    ""
  );
}

function idsFrom(source) {
  const properties = source?.properties ?? source ?? {};
  return {
    session_id: properties.sessionID ?? properties.session_id ?? properties.session?.id ?? null,
    turn_id: properties.messageID ?? properties.message_id ?? properties.message?.id ?? null,
  };
}

function appendContext(output, context) {
  if (!context || !output) {
    return;
  }
  if (typeof output.prompt === "string") {
    output.prompt = `${output.prompt}\n\n${context}`;
    return;
  }
  if (typeof output.text === "string") {
    output.text = `${output.text}\n\n${context}`;
    return;
  }
  if (Array.isArray(output.context)) {
    output.context.push(context);
  }
}

async function log(client, level, message, extra) {
  await client?.app?.log?.({
    body: { service: SERVICE, level, message, extra },
  });
}

function writeState(cwd, state) {
  const statePath = path.join(cwd, OPENCODE_STATE);
  fs.mkdirSync(path.dirname(statePath), { recursive: true });
  fs.writeFileSync(statePath, `${JSON.stringify(state, null, 2)}\n`, "utf8");
}

function writePostRefinePrompt(cwd, prompt) {
  const promptPath = path.join(cwd, OPENCODE_POST_REFINE_PROMPT);
  fs.mkdirSync(path.dirname(promptPath), { recursive: true });
  fs.writeFileSync(promptPath, `${prompt}\n`, "utf8");
}

function removePostRefinePrompt(cwd) {
  fs.rmSync(path.join(cwd, OPENCODE_POST_REFINE_PROMPT), { force: true });
}

async function AiPolicyRuntime({ client, directory, worktree }) {
  const cwd = worktree || directory || process.cwd();
  await log(client, "info", "OpenCode plugin initialized", { packageRoot: PACKAGE_ROOT, cwd });

  return {
    "shell.env": async (_input, output) => {
      output.env = output.env || {};
      output.env.AI_POLICY_AGENT = "opencode";
      output.env.AI_POLICY_ROOT = PACKAGE_ROOT;
    },

    "tui.prompt.append": async (input, output) => {
      const prompt = promptFrom(input, output);
      if (!prompt || typeof prompt !== "string") {
        return;
      }
      const result = runHook("opencode-user-prompt-submit", {
        cwd,
        prompt,
        ...idsFrom(input),
      });
      const context = result.response?.hookSpecificOutput?.additionalContext;
      appendContext(output, context);
      if (!result.ok) {
        await log(client, "warn", "User prompt hook failed", result);
      }
    },

    event: async ({ event }) => {
      if (event?.type !== "session.idle") {
        return;
      }
      const result = runHook("opencode-stop-refinement", {
        cwd,
        ...idsFrom(event),
      });
      if (!result.ok || result.response?.decision !== "block") {
        removePostRefinePrompt(cwd);
        writeState(cwd, {
          event: "session.idle",
          postRefinePrepared: false,
          hookOk: result.ok,
          decision: result.response?.decision ?? null,
          error: result.error ?? null,
        });
        return;
      }
      writePostRefinePrompt(cwd, String(result.response.reason || ""));
      writeState(cwd, {
        event: "session.idle",
        postRefinePrepared: true,
        reasonChars: String(result.response.reason || "").length,
      });
      await log(client, "info", "Post-refinement prompt prepared", {
        reasonChars: String(result.response.reason || "").length,
      });
    },
  };
}

module.exports = { AiPolicyRuntime };
