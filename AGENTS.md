# AGENTS.md

Keep this file short — it is loaded into agent context.

## Dependency pins

Always pin dependencies in `pyproject.toml` with three-component compatible-release constraints (e.g. `pydantic~=2.14.2`), never bare `>=` or two-component pins.

## Docker images

Always pin docker images in `docker-compose.yml` to a specific version tag, never `latest`.

## Architecture diagram

The README contains a Mermaid architecture diagram. The bootstrap script itself is not part of the diagram, but the connections it establishes between components are. Remind the user to update it whenever the implementation changes and the diagram no longer matches.

## Tests

Do not write new tests unless explicitly instructed to. Modifying or extending existing tests is allowed, but only by following an already established test pattern. Do not introduce new types of tests.

## Git

Do not stage, unstage, or commit files, unless explicitly instructed. The user might stage files as they review your work.

## Pre-commit hooks

Check them by delegating a `pre-commit run --all-files` run to a sub-agent, so the output does not clutter the main conversation.

## Docstrings

Short and to the point. Describe what the function does without repeating its name. Document `Args:`/`Returns:` only when something important needs to be stated (units, invariants, side effects, edge cases); otherwise omit them.
