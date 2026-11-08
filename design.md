# Design

## Main Architecture

```mermaid
flowchart LR
    subgraph agent["Agent"]
        main["main / supervisor agent"]
    end

    subgraph evals["Evaluations"]
        mlflow_evals["MLflow evaluations"]
        arize_evals["Arize evaluations"]
    end

    subgraph docker["Docker (docker compose)"]
        subgraph mlflow_box["mlflow_server — MLflow"]
            gateway["AI Gateway<br/>/gateway/mlflow/v1"]
            traces["Traces from the agent"]
            datasets["Datasets"]
            eval_runs["Evaluation runs"]
        end
        mlflow_db["mlflow_db<br/>Postgres 17"]
        searxng["searxng<br/>SearXNG metasearch"]
        valkey["searxng_valkey<br/>Valkey (rate limiting)"]
        minio["minio<br/>S3 storage (MLflow artifacts)"]
        pgadmin["pgadmin<br/>Postgres web UI"]
        subgraph arize_box["arize — Arize"]
            arize["Observability<br/>& traces"]
            arize_datasets["Datasets<br/>& experiments"]
        end
        arize_db["arize_db<br/>Postgres 17"]
    end

    macbook["MacBook (M5 Pro)<br/>OMLX server<br/>e.g. Qwen3.8-27B-oQ4e-mtp"]
    aws["AWS Bedrock<br/>e.g. eu.amazon.nova-2-lite-v1"]
    tavily["Tavily<br/>search API"]
    engines["Search engines"]
    web["Web pages"]

    main -->|LLM requests| gateway
    main -->|trace logging| traces
    main -->|trace logging| arize
    main -->|search tool| searxng
    main -->|search tool - alternative| tavily
    main -->|fetch_page tool| web
    mlflow_evals -->|runs the agent| main
    mlflow_evals -->|judge requests| gateway
    mlflow_evals -->|reads datasets| datasets
    mlflow_evals -->|evaluation runs| eval_runs
    arize_evals -->|runs the agent| main
    arize_evals -->|judge requests| gateway
    arize_evals -->|reads datasets| arize_datasets
    arize_evals -->|experiments| arize_datasets
    arize -->|stores datasets & experiments| arize_datasets
    arize -->|providers call LLMs via gateway| gateway
    gateway -->|local model| macbook
    gateway -->|cloud model| aws
    searxng -->|queries| engines
    searxng --> valkey
    mlflow_box -->|backend store| mlflow_db
    mlflow_box -->|artifacts| minio
    pgadmin -->|manages| mlflow_db
    arize -->|backend store| arize_db
    pgadmin -->|manages| arize_db
```

- **Agent** — an agent built on [LangChain](https://www.langchain.com) [deepagents](https://github.com/langchain-ai/deepagents) framework (currently just the main/supervisor agent). LLM requests go through the [MLflow](https://mlflow.org) AI Gateway; every run is traced back to MLflow.
- **Evaluations** — the evaluation code runs the agent over an evaluation dataset and scores each record with an LLM judge that also goes through the AI Gateway. Two platforms are supported: **MLflow** (datasets and results live in MLflow) and **Arize** (datasets and results are stored as a Phoenix experiment in Arize).
- **MLflow AI Gateway** — proxies OpenAI-compatible requests to either the local [OMLX](https://omlx.ai/) server on the MacBook (default) or [AWS Bedrock](https://aws.amazon.com/bedrock/), and records the agent's traces. It also monitors token usage and cost in real time, and supports budgets that cap spending, so a runaway agent run can't keep consuming tokens and money indefinitely.
- **Search** — the agent's search tool uses either the local [SearXNG](https://docs.searxng.org/) metasearch backend (which fans out to many public search engines) or the [Tavily](https://tavily.com) API, one or the other. It returns result titles, URLs and snippets only; a separate `fetch_page` tool fetches the content of individual pages (as markdown) when the agent needs it.
- **Supporting containers** — [Postgres](https://www.postgresql.org) (MLflow's backend store), [Arize](https://arize.com) (LLM/agent observability) with its own Postgres store, [MinIO](https://min.io) (artifact storage), and [pgAdmin](https://www.pgadmin.org) (database UI).

## Bootstrapping

The bootstrap script provisions the services below (node names match the Main Architecture diagram). It is intentionally not part of the main diagram.

```mermaid
flowchart LR
    bs["Bootstrap script"]

    subgraph docker["Docker (docker compose)"]
        subgraph mlflow_box["mlflow_server — MLflow"]
            gateway["AI Gateway<br/>/gateway/mlflow/v1"]
            datasets["Datasets"]
        end
        minio["minio<br/>S3 storage (MLflow artifacts)"]
        arize["arize<br/>Arize (LLM observability)"]
    end

    bs -->|creates buckets| minio
    bs -->|configures gateways and models| gateway
    bs -->|creates experiments and loads datasets| datasets
    bs -->|loads datasets| arize
    bs -->|configures LLM providers and models| arize
```

The script runs in this order:

1. **MinIO** — creates the S3 buckets (the gateway's artifact root is `s3://mlflow-artifacts`, so the bucket must exist first).
2. **MLflow AI Gateway** — registers gateways and models.
3. **MLflow** — creates experiments and loads the (filtered) datasets.
4. **Arize** — loads datasets, then configures its custom LLM providers and models to point at the MLflow AI Gateway (so this must come after the gateway is set up).

## Evaluations

```mermaid
flowchart LR
    repo["Datasets<br/>(git repo)"]
    jinja["Jinja rendering<br/>(prompt templates)"]
    loader["Loader<br/>filtering"]
    mlflow_ds["MLflow<br/>datasets"]
    arize_ds["Arize<br/>datasets"]
    evals["Evals script"]

    repo --> jinja
    jinja --> loader
    loader -->|loads| mlflow_ds
    loader -->|loads| arize_ds
    mlflow_ds -->|loads dataset| evals
    arize_ds -->|loads dataset| evals
```

Evaluation flow, step by step:

- **Datasets** — stored in the git repo as the source of truth.
- **Jinja logic** — dataset records can use Jinja templating to render prompts before they are sent to the agent.
- **Loader** — a loader script applies filtering to the dataset and loads the records into MLflow (as a dataset) and Arize. It can load multiple datasets and merge them into a single MLflow or Arize dataset.
- **Evals script** — runs the agent over the evaluation dataset, loading it from MLflow or Arize depending on the selected platform.

## Configuration

Configs live in `config/` (agent configs under `config/agent/`) and are resolved by a generic `ConfigLoader` (`src/tech_market_researcher/core/config_loader.py`). For a config name `X`, the loader merges these layers, in this order (later layers win):

1. **Common config** — `_common.yaml` (optional): the shared baseline shared by all configs.
2. **Named config** — `X.yaml` (required): the per-environment config (e.g. `dev.yaml`, `prd.yaml`). A missing named file is an error.
3. **Local override** — `X.local.yaml` (optional): machine-local tweaks. Ignored by `validate_all_configs` and git-ignored (`config/**/*.local.yaml`), so it is never committed.

```mermaid
flowchart LR
    common["_common.yaml<br/>(optional)"]
    common_mixins["mixins/*.yaml<br/>declared by _common.yaml"]
    named["X.yaml<br/>(required)"]
    named_mixins["mixins/*.yaml<br/>declared by X.yaml"]
    local["X.local.yaml<br/>(optional, not committed)"]
    local_mixins["mixins/*.yaml<br/>declared by X.local.yaml"]
    merge["Deep merge<br/>later layers win"]
    validated["ApplicationConfig<br/>pydantic, extra=forbid"]

    common --> common_mixins
    common_mixins --> named
    named --> named_mixins
    named_mixins --> local
    local --> local_mixins
    common_mixins --> merge
    named_mixins --> merge
    local_mixins --> merge
    merge --> validated
```

- **Deep merge** — dicts are merged recursively; scalars and arrays are replaced, never concatenated, so a later layer fully overrides a key.
- **Mixins** — any of the three layers may declare a `_mixins` list; each entry references `mixins/<name>.yaml`, which is merged on top of the layer that declares it (in declaration order). Mixins let shared fragments be reused across configs and local overrides; the `_mixins` field itself is dropped before validation. The `mixins/` folder is committed like the rest of the configs.
- **Validation** — the merged result is validated into a pydantic model (`ApplicationConfig` for the agent) with `extra="forbid"`, so a typo'd key in any YAML file fails at load time. `ConfigLoader.validate_all_configs()` loads every named config (without local overrides) so committed configs are checked by the unit tests and pre-commit.
- **Local overrides** — files named `X.local.yaml` are the mechanism for per-machine changes (e.g. pointing `dev` at a different local model) without touching committed files.
- **Loader configuration** — the loader is configurable via its constructor, e.g. base folder, common config filename, mixins folder, mixins field name, local-override infix, and file extension.

## Developer Setup

```mermaid
flowchart TB
    subgraph home["Home network"]
        subgraph mac["Newer MacBook (M5 Pro)"]
            omix["OMLX server<br/>e.g. Qwen3.8-27B-oQ4e-mtp"]
        end
        subgraph laptop["Older MacBook (host)"]
            vscode["VS Code"]
            browser["Browser"]
            subgraph vm["Ubuntu VM (VMware)"]
                code["Code (git repo)"]
                subgraph docker["Docker (docker compose)"]
                    gateway["AI Gateway<br/>/gateway/mlflow/v1"]
                    mlflow_ui["MLflow UI"]
                    arize["Arize"]
                end
                main["Tech Market Researcher Agent<br/>(main / supervisor)"]
                subgraph cli["CLI"]
                    pi["Pi (agentic harness)"]
                end
            end
        end
    end
    aws["AWS Bedrock<br/>e.g. eu.amazon.nova-2-lite-v1"]

    vscode -->|SSH| cli
    vscode -->|Remote-SSH extension| code
    browser -->|VS Code exposes VM ports on 0.0.0.0| mlflow_ui
    browser -->|VS Code exposes VM ports on 0.0.0.0| arize
    pi -->|LLM requests| omix
    cli -->|runs| main
    main --> gateway
    gateway -->|local model| omix
    gateway -->|cloud model via Bedrock long-term API key| aws
```

- **Newer MacBook (M5 Pro)** — dedicated to serving models through [OMLX](https://omlx.ai/), reachable over the local network. Its wired memory limit is raised as high as possible and no other applications run on it, so it is used for model serving only.
- **Older MacBook (host)** — runs VS Code on the host, which opens a remote CLI in the Ubuntu VM over SSH and edits the code in the VM directly using the Remote-SSH extension. A browser on the host is used to view the MLflow and Arize UIs; VS Code exposes the VM's service ports on 0.0.0.0 on the host. [VMware](https://www.vmware.com/) hosts the Ubuntu VM on this machine.
- **Ubuntu VM** — stores the code (the git repo) and runs all the scripts, the Docker services, and the [Pi](https://pi.dev) agentic harness. The Tech Market Researcher Agent is run from the CLI. It is deliberately a sandbox: it holds no secrets other than the ones this project needs and no personal user data.
- **Models** — Pi talks directly to the OMLX server; the DeepAgents agent goes through the MLflow AI Gateway, which routes to the OMLX server (default) or to [AWS Bedrock](https://aws.amazon.com/bedrock/). The gateway uses a long-term Bedrock API key to access Bedrock, which grants access to specific models only (Amazon Nova) and can be activated or deactivated at will.
