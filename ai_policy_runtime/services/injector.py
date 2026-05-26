from __future__ import annotations

from pathlib import Path


BEGIN = "<!-- POLICY_RUNTIME_BEGIN -->"
END = "<!-- POLICY_RUNTIME_END -->"
AGENTS_MD_TARGETS = frozenset({"codex", "opencode"})


def inject_current_prompt(root: str | Path, target: str) -> Path:
    root_path = Path(root)
    prompt_path = root_path / ".policy" / "current" / "effective-prompt.md"
    prompt = prompt_path.read_text(encoding="utf-8")

    output = _target_prompt_path(root_path, target)

    block = f"{BEGIN}\n{prompt}\n{END}"
    if target == "custom":
        output.write_text(block + "\n", encoding="utf-8")
        return output

    existing = output.read_text(encoding="utf-8") if output.exists() else "# Project Rules\n"
    output.write_text(_replace_block(existing, block), encoding="utf-8")
    return output


def clear_injected_prompt(root: str | Path, target: str) -> Path | None:
    """Remove a previously injected policy block from an agent file."""

    root_path = Path(root)
    output = _target_prompt_path(root_path, target)

    if not output.exists():
        return None
    if target == "custom":
        output.unlink()
        return output

    existing = output.read_text(encoding="utf-8")
    cleaned = _remove_block(existing)
    if cleaned != existing:
        output.write_text(cleaned, encoding="utf-8")
    return output


def _replace_block(text: str, block: str) -> str:
    if BEGIN in text and END in text:
        start = text.index(BEGIN)
        end = text.index(END) + len(END)
        return text[:start].rstrip() + "\n\n" + block + "\n" + text[end:].lstrip()
    return text.rstrip() + "\n\n" + block + "\n"


def _target_prompt_path(root: Path, target: str) -> Path:
    if target in AGENTS_MD_TARGETS:
        return root / "AGENTS.md"
    if target == "claude":
        return root / "CLAUDE.md"
    if target == "custom":
        return root / ".policy" / "current" / "injected-prompt.md"
    raise ValueError(f"Unsupported injection target: {target}")


def _remove_block(text: str) -> str:
    if BEGIN not in text or END not in text:
        return text
    start = text.index(BEGIN)
    end = text.index(END) + len(END)
    cleaned = text[:start].rstrip() + "\n\n" + text[end:].lstrip()
    return cleaned.rstrip() + "\n"
