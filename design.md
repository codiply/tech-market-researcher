# Design

## Main Architecture

```mermaid
flowchart LR
    subgraph agent["Agent"]
        main["main / supervisor agent"]
    end

    subgraph evals["Evaluations"]
        mlflow_evals["MLflow evaluations"]
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
        arize["arize<br/>Arize (LLM observability)"]
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
    main -.->|search tool - alternative| tavily
    main -->|fetch_page tool| web
    mlflow_evals -->|runs the agent| main
    mlflow_evals -->|judge requests| gateway
    mlflow_evals -->|reads datasets| datasets
    mlflow_evals -->|evaluation runs| eval_runs
    arize -->|providers call LLMs via gateway| gateway
    gateway -->|local model| macbook
    gateway -.->|cloud model| aws
    searxng -->|queries| engines
    searxng --> valkey
    mlflow_box -->|backend store| mlflow_db
    mlflow_box -->|artifacts| minio
    pgadmin -->|manages| mlflow_db
    arize -->|backend store| arize_db
    pgadmin -->|manages| arize_db
```

- **Agent** — an agent built on [LangChain](https://www.langchain.com) [deepagents](https://github.com/langchain-ai/deepagents) framework (currently just the main/supervisor agent). LLM requests go through the [MLflow](https://mlflow.org) AI Gateway; every run is traced back to MLflow.
- **Evaluations** — the evaluation code runs the agent over the evaluation dataset stored in MLflow and scores each record with an LLM judge that also goes through the AI Gateway; results are stored back as MLflow evaluation runs.
- **MLflow AI Gateway** — proxies OpenAI-compatible requests to either the local [OMLX](https://omlx.ai/) server on the MacBook (default) or [AWS Bedrock](https://aws.amazon.com/bedrock/), and records the agent's traces.
- **Search** — the agent's search tool uses either the local [SearXNG](https://docs.searxng.org/) metasearch backend (which fans out to many public search engines) or the [Tavily](https://tavily.com) API, one or the other. It returns result titles, URLs and snippets only; a separate `fetch_page` tool fetches the content of individual pages (as markdown) when the agent needs it.
- **Supporting containers** — [Postgres](https://www.postgresql.org) (MLflow's backend store), [Arize](https://arize.com) (LLM/agent observability) with its own Postgres store, [MinIO](https://min.io) (artifact storage), and [pgAdmin](https://www.pgadmin.org) (database UI).

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
```

Evaluation flow, step by step:

- **Datasets** — stored in the git repo as the source of truth.
- **Jinja logic** — dataset records can use Jinja templating to render prompts before they are sent to the agent.
- **Loader** — a loader script applies filtering to the dataset and loads the records into MLflow (as a dataset) and Arize. It can load multiple datasets and merge them into a single MLflow or Arize dataset.
- **Evals script** — runs the agent over the evaluation dataset, loading it from MLflow.

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
