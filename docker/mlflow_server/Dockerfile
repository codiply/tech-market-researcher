# MLflow with the PostgreSQL driver preinstalled.
# boto3/botocore are needed server-side because artifacts go through the
# S3-compatible MinIO endpoint (MLFLOW_S3_ENDPOINT_URL); the base image
# does not include the mlflow[s3] extras.
FROM ghcr.io/mlflow/mlflow:v3.17.2

RUN pip install --no-cache-dir psycopg[binary] boto3
