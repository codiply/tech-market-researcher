# AGENTS.md

Keep this file short — it is loaded into agent context.

## Dependency pins

Always pin dependencies in `pyproject.toml` with three-component compatible-release constraints (e.g. `pydantic~=2.14.2`), never bare `>=` or two-component pins.

## Docker images

Always pin docker images in `docker-compose.yml` to a specific version tag, never `latest`.

## Docstrings

Short and to the point. Describe what the function does without repeating its name. Document `Args:`/`Returns:` only when something important needs to be stated (units, invariants, side effects, edge cases); otherwise omit them.
