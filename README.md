# Tech Market Researcher

> **Built with the [Pi](https://pi.dev/) agent harness** and a [Jundot/Qwen3.8-27B-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp) model (Qwen 3.8, 27B parameters), running locally on an M5 Pro MacBook via [OMLX](https://omlx.ai/). Development was heavily human-steered, with direction given commit by commit.

## Design

See [design.md](design.md) for the design and architecture diagrams.

## Start

```bash
./stack.sh up -d
```

This also runs the bootstrap, which provisions the stack's resources (S3, MLflow, Arize) from `config/`. It is idempotent; re-run it any time with `./stack.sh bootstrap` (e.g. after updating the bootstrap config). Note that gateway endpoints and their API keys are written into the AI Gateway during the bootstrap, so if you change an endpoint URL or a secret in `.env` you must re-run the bootstrap for the gateway to pick it up.

The stack exposes:

| Service | URL | Notes |
| --- | --- | --- |
| [MLflow](https://mlflow.org) (UI) | [http://0.0.0.0:5000](http://0.0.0.0:5000) | experiments, models, AI Gateway |
| [SearXNG](https://docs.searxng.org/) (UI) | [http://0.0.0.0:8384](http://0.0.0.0:8384) | local metasearch backend for the search tool; its JSON API powers the agent, no login |
| [MinIO](https://min.io) (UI) | [http://0.0.0.0:9001](http://0.0.0.0:9001) | log in with `minio` / `minio12345` (fixed local default; MinIO requires an 8+ char root password) |
| [pgAdmin](https://www.pgadmin.org) (UI) | [http://0.0.0.0:5050](http://0.0.0.0:5050) | log in with `admin@example.com` / `pgadmin12345` (fixed local default, override with `PGADMIN_PASSWORD`); the MLflow and Arize databases are pre-configured with the `MLFLOW_DB_*` / `ARIZE_DB_*` credentials from `.env` |
| [Arize](https://arize.com) (UI) | [http://0.0.0.0:6006](http://0.0.0.0:6006) | Arize: LLM/agent observability server + UI; also the OTLP HTTP collector on `6006` and gRPC collector on `4317` |
| [MinIO](https://min.io) (S3 API) | `http://0.0.0.0:9000` | S3 endpoint for MLflow artifacts (`MLFLOW_S3_ENDPOINT_URL`) |
| MLflow database ([Postgres](https://www.postgresql.org)) | `0.0.0.0:5433` | backend store; credentials in `.env` (`MLFLOW_DB_*`) |
| Arize database ([Postgres](https://www.postgresql.org)) | `0.0.0.0:5434` | Arize backend store; credentials in `.env` (`ARIZE_DB_*`) |

## Run the agent

```bash
uv run agent_cli
```

Starts an interactive chat REPL against the MLflow AI Gateway configured in `.env`. Flags: `--config-name` (name of the config to load; its file `<name>.yaml` is read, default `base`), `--user-id`, `--session-id`, and `--time-zone` (user IANA time zone used by tools that report times, e.g. `Europe/Madrid`).

## Run the evaluations

```bash
uv run evals
```

Runs the agent against an evaluation dataset; each record is scored by the configured evaluators — LLM judges served through the AI Gateway, plus platform-specific code evaluators (e.g. exact match and regex checks on Arize). The platform is selected with `--platform` (`mlflow`, default, or `arize`): with `mlflow` the results are stored as MLflow evaluation runs under the dataset's experiment, and with `arize` they are stored as a Phoenix experiment attached to the dataset. Flags: `--agent-config-name` (name of the agent config to load, default `base`), `--evals-config-name` (name of the evaluation config under `config/evaluations/<platform>/`, default `main`), and `--platform` (`mlflow` or `arize`, default `mlflow`). Requires the docker compose stack to be running and the dataset to have been bootstrapped.

## Common commands

```bash
./stack.sh ps                      # status
./stack.sh logs -f mlflow_server   # follow one service
./stack.sh down                    # stop and remove containers
./stack.sh bootstrap               # re-run the (idempotent) bootstrap, e.g. after updating its config
```

## Pre-commit hooks

The repo uses the [pre-commit](https://pre-commit.com) framework: `.pre-commit-config.yaml` (tracked in git) declares the code-quality checks, and a git hook runs them on every `git commit` against the staged Python files. Formatting tools auto-fix and re-stage any changes; checkers fail the commit on errors.

### Install

Setup is per-clone:

```bash
uv sync --group dev         # 1. install the checkers into the project venv
uv tool install pre-commit  # 2. install the pre-commit tool
pre-commit install          # 3. wire the hook into git (writes .git/hooks/pre-commit)
```

If `uv tool` is not available, you can install pre-commit manually (e.g. `pip install pre-commit` or `pipx install pre-commit`). The reason pre-commit is installed as a standalone tool rather than added to `pyproject.toml` is that the git hook it generates calls the `pre-commit` binary from your `PATH` on every commit, so it must exist outside the project venv.

### Updating the config

Changes to `.pre-commit-config.yaml` are picked up automatically. Re-run `pre-commit install` only if you change the hook stage/type (e.g. `--hook-type pre-merge-commit`).

Do not install a pre-commit `pre-push` hook here — it would overwrite the custom force-push guard documented below.

### Run the checks manually

```bash
pre-commit run --all-files   # run all checks on every file
pre-commit run <id>          # run a single check (ids in .pre-commit-config.yaml)
```

You can also run any of the checkers in the `dev` group directly, e.g. `uv run <tool> src/`.

### Bypassing the hook in an emergency

If you need to save work fast (e.g. battery about to die and need to push my work) and the checks are blocking you, skip the hook with:

```bash
git commit --no-verify -m "emergency save"
```

This commits exactly what you have without running any of the checks. Fix the lint/type issues later with a normal commit.

## Pushing to main

This is a personal repo: pushing directly to `main` is allowed and expected — there is no PR workflow. Regular (fast-forward) pushes to `main` and all other branches work normally.

GitHub does not offer branch protection on free accounts, so server-side branch rules (e.g. disallowing force push) cannot be configured. Instead, a local `pre-push` hook guards against the dangerous operations on `main`: **force pushes (non-fast-forward) and deletion of the branch are rejected**, so an accidental `git push --force` or `git push --delete main` can't rewrite or erase published history. Everything else — fast-forward pushes, creating `main`, other branches, deleting feature branches — is unaffected.

### Install the pre-push hook

The hook lives in `scripts/git-hooks/pre-push` (tracked in git). Setup is per-clone, since git hooks are local to each `.git` directory:

```bash
cp scripts/git-hooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

## Learning Resources

- [LangChain Deep Agents documentation](https://docs.langchain.com/oss/python/deepagents/overview) — docs for the `deepagents` framework the agent is built on.
- [Evaluating and Improving Agentic Applications](https://learn.deeplearning.ai/courses/evaluating-and-improving-agentic-applications) — DeepLearning.AI course on evaluating and improving agents, which uses Arize.

## License

Licensed under the [MIT License](LICENSE).
