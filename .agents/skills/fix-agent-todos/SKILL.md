---
name: fix-agent-todos
description: Find and action AGENT_TODO: markers in the codebase, treating each marker as an instruction to follow. Use only when the user explicitly asks to fix or action AGENT_TODO markers.
---

# Fix Agent TODOs

`AGENT_TODO:` markers in the code are instructions left for the agent. Action each one.

## Steps

1. Find all markers: `rg -n "AGENT_TODO:" --no-ignore --hidden -g '!.venv' -g '!.git' -g '!.mypy_cache' -g '!.pytest_cache' -g '!.agents/skills/fix-agent-todos'`
   - `--hidden` is needed so dot-directories (e.g. `.agents`, `.github`) are searched; `--no-ignore` alone does not cover them.
   - If none are found, tell the user there are no pending AGENT_TODOs and stop.
2. Group the markers by file. For each marker, read enough surrounding context to understand the instruction.
3. Action the instruction: it is a directive from the human to you, so follow it exactly as written (refactor, fix, rename, whatever the marker asks for). Keep the change consistent with the surrounding code and project conventions (see AGENTS.md).
4. After actioning a marker, remove its `AGENT_TODO:` comment. Remove only the marker comment — keep any other comments on the same line/block that are not part of the marker.
5. Report what each marker asked for and what you did about it, one by one.

## Rules

- Only action markers the user explicitly asked you to by invoking this skill; do not act on AGENT_TODOs during unrelated work.
- If a marker's instruction is ambiguous or conflicts with project conventions, ask the user before acting; do not remove the marker until the question is resolved.
- Only remove a marker after actioning it, or when the user explicitly says to remove it.
