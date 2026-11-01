# Tech Market Researcher

## Start

```bash
./stack.sh up -d
```

The stack exposes:

| Service | URL | Notes |
| --- | --- | --- |
| MLflow (UI) | [http://0.0.0.0:5000](http://0.0.0.0:5000) | experiments, models, AI Gateway |
| MinIO (UI) | [http://0.0.0.0:9001](http://0.0.0.0:9001) | log in with `minio` / `minio12345` (fixed local default; MinIO requires an 8+ char root password) |
| pgAdmin (UI) | [http://0.0.0.0:5050](http://0.0.0.0:5050) | log in with `admin@example.com` / `pgadmin12345` (fixed local default, override with `PGADMIN_PASSWORD`); the MLflow database is pre-configured with the `MLFLOW_DB_*` credentials from `.env` |
| MinIO (S3 API) | `http://0.0.0.0:9000` | S3 endpoint for MLflow artifacts (`MLFLOW_S3_ENDPOINT_URL`) |
| MLflow database (Postgres) | `0.0.0.0:5433` | backend store; credentials in `.env` (`MLFLOW_DB_*`) |

## Common commands

```bash
./stack.sh ps                      # status
./stack.sh logs -f mlflow_server   # follow one service
./stack.sh down                    # stop and remove containers
```
