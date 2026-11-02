"""Bootstrap the MLflow stack: MinIO artifact bucket + AI Gateway models.

1. Creates the MinIO bucket used for MLflow artifacts (via boto3) if it
   does not exist yet — neither MinIO nor MLflow create it automatically.
2. Provisions one gateway secret for the OpenAI-compatible provider
   (e.g. an oMLX server) plus a model definition + endpoint per model in
   ``MODELS`` below.

Safe to re-run: existing objects are reused, not duplicated.

Usage:
    # from the repo root (loads .env automatically if present)
    uv run python scripts/bootstrap.py
"""

import os
import sys

import boto3
import requests
from dotenv import load_dotenv

load_dotenv()

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000").rstrip(
    "/"
)
BASE = f"{TRACKING_URI}/api/3.0/mlflow/gateway"
TIMEOUT = 30  # seconds

# If your server has basic auth enabled, set MLFLOW_TRACKING_USERNAME / PASSWORD.
_user = os.environ.get("MLFLOW_TRACKING_USERNAME")
_pass = os.environ.get("MLFLOW_TRACKING_PASSWORD")
AUTH = (_user, _pass) if _user and _pass else None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# MinIO bucket for MLflow artifacts (must match --default-artifact-root
# in docker/docker-compose.yml: s3://mlflow-artifacts).
S3_BUCKET = os.environ.get("MLFLOW_S3_BUCKET", "mlflow-artifacts")
S3_ENDPOINT = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

PROVIDER = "openai"  # any OpenAI-compatible API (OpenAI, oMLX, vLLM, ...)

SECRET_NAME = "omlx-server"

# OPENAI_BASE_URL is required for non-OpenAI providers (e.g. oMLX);
# leave unset for the real OpenAI API.
# (set in .env, e.g. http://<oMLX server IP>:11436/v1)

# One (model_name, endpoint_name) pair per model you want exposed.
# model_name is the model id the provider serves; endpoint_name is the name
# of the gateway endpoint (what callers use).
MODELS: list[tuple[str, str]] = [
    ("Qwen3.8-27B-oQ4e-mtp", "oMLX"),
    # ("qwen3:8b", "my-omlx-endpoint"),
]


class BootstrapError(RuntimeError):
    """Raised when the MLflow stack is unreachable or in an inconsistent state."""


# ---------------------------------------------------------------------------
# MinIO bucket
# ---------------------------------------------------------------------------


def ensure_bucket(bucket: str, endpoint: str) -> None:
    """Create the artifact bucket if it does not exist (idempotent)."""
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    try:
        names = {b["Name"] for b in s3.list_buckets()["Buckets"]}
    except s3.exceptions.ClientError as e:
        raise BootstrapError(f"Cannot reach MinIO at {endpoint}: {e}") from e
    if bucket in names:
        print(f"MinIO bucket '{bucket}' already exists.")
        return
    s3.create_bucket(Bucket=bucket)
    print(f"Created MinIO bucket '{bucket}'.")


# ---------------------------------------------------------------------------
# AI Gateway
# ---------------------------------------------------------------------------


def _call(method, path, *, params=None, body=None, allow_404=False):
    """Call the gateway REST API. Returns parsed JSON, or None on an allowed 404."""
    url = f"{BASE}/{path}"
    try:
        resp = requests.request(
            method, url, params=params, json=body, auth=AUTH, timeout=TIMEOUT
        )
    except requests.ConnectionError as e:
        raise BootstrapError(
            f"Cannot reach MLflow server at {TRACKING_URI}: {e}"
        ) from e
    except requests.Timeout as e:
        raise BootstrapError(f"Timed out calling {url}") from e

    if resp.status_code == 404 and allow_404:
        return None
    if not resp.ok:
        # MLflow usually returns {"error_code": "...", "message": "..."}
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise BootstrapError(f"{method} {path} failed ({resp.status_code}): {detail}")
    return resp.json()


def ensure_secret(name, provider, secret_value, auth_config=None):
    """Create or reuse a gateway secret.

    Note: secrets/get requires a secret_id, so we search secrets/list by name.
    """
    for secret in _call("GET", "secrets/list").get("secrets", []):
        if secret.get("secret_name") == name:
            print(f"Secret '{name}' already exists - reusing.")
            return secret
    body = {"secret_name": name, "provider": provider, "secret_value": secret_value}
    if auth_config:
        body["auth_config"] = auth_config
    print(f"Creating secret '{name}'...")
    return _call("POST", "secrets/create", body=body)["secret"]


def ensure_model_definition(name, secret_id, provider, model_name):
    """Create or reuse a model definition, re-linking it if it points at another secret."""
    listing = _call("GET", "model-definitions/list", params={"provider": provider})
    for md in listing.get("model_definitions", []):
        if md.get("name") == name:
            # secret_id is absent from the response when NULL in the DB (proto3
            # omits unset fields) - e.g. after the linked secret was deleted.
            if md.get("secret_id") == secret_id:
                print(f"Model definition '{name}' already exists - reusing.")
                return md
            print(
                f"Model definition '{name}' is linked to a different secret "
                f"({md.get('secret_id') or 'none'}) - re-linking to '{secret_id}'."
            )
            return _call(
                "POST",
                "model-definitions/update",
                body={
                    "model_definition_id": md["model_definition_id"],
                    "secret_id": secret_id,
                },
            )["model_definition"]
    print(f"Creating model definition '{name}'...")
    return _call(
        "POST",
        "model-definitions/create",
        body={
            "name": name,
            "secret_id": secret_id,
            "provider": provider,
            "model_name": model_name,
        },
    )["model_definition"]


def ensure_endpoint(name, model_definition_id):
    """Create or reuse an endpoint, ensuring a PRIMARY link to the given model."""
    for endpoint in _call("GET", "endpoints/list").get("endpoints", []):
        if endpoint.get("name") == name:
            mappings = endpoint.get("model_mappings", [])
            linked = [
                m
                for m in mappings
                if m.get("model_definition_id") == model_definition_id
                and m.get("linkage_type") == "PRIMARY"
            ]
            if linked:
                print(f"Endpoint '{name}' already exists - reusing.")
                return endpoint
            print(f"Endpoint '{name}' has no PRIMARY link to the model - updating.")
            # endpoints/update replaces all model linkages when model_configs
            # is provided.
            return _call(
                "POST",
                "endpoints/update",
                body={
                    "endpoint_id": endpoint["endpoint_id"],
                    "model_configs": [
                        {
                            "model_definition_id": model_definition_id,
                            "linkage_type": "PRIMARY",
                        }
                    ],
                },
            )["endpoint"]
    print(f"Creating endpoint '{name}'...")
    return _call(
        "POST",
        "endpoints/create",
        body={
            "name": name,
            "model_configs": [
                {"model_definition_id": model_definition_id, "linkage_type": "PRIMARY"}
            ],
        },
    )["endpoint"]


def verify(secret, model_def, endpoint):
    """Print the linkage chain so a broken link (e.g. NULL secret_id) is obvious."""
    print("\nVerification:")
    print(f"  secret:            {secret['secret_name']} ({secret['secret_id']})")
    linked = (model_def.get("secret_id") or "none") == secret["secret_id"]
    print(
        f"  model definition:  {model_def['name']} -> "
        f"{model_def.get('secret_id') or 'NO SECRET (broken)'}"
    )
    primary = [
        m["model_definition_id"]
        for m in endpoint.get("model_mappings", [])
        if m.get("linkage_type") == "PRIMARY"
    ]
    print(
        f"  endpoint:          {endpoint['name']} PRIMARY models: {primary or 'NONE (broken)'}"
    )
    if not linked or primary != [model_def["model_definition_id"]]:
        raise BootstrapError(
            "Verification failed: linkage chain is broken, see details above."
        )
    print("  OK: secret -> model definition -> endpoint chain is intact.")


def main():
    """Provision the stack and verify the secret -> model -> endpoint linkage."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY in your environment (or .env) first.")

    try:
        ensure_bucket(S3_BUCKET, S3_ENDPOINT)

        # OPENAI_BASE_URL is optional; non-OpenAI providers (or any custom
        # base URL) are configured via the secret's auth_config.
        api_base = os.environ.get("OPENAI_BASE_URL")  # e.g. oMLX server
        auth_config = None
        if api_base:
            auth_config = {"auth_mode": "api_key", "api_base": api_base}

        secret = ensure_secret(SECRET_NAME, PROVIDER, {"api_key": api_key}, auth_config)
        for model_name, endpoint_name in MODELS:
            model_def = ensure_model_definition(
                model_name, secret["secret_id"], PROVIDER, model_name
            )
            endpoint = ensure_endpoint(endpoint_name, model_def["model_definition_id"])
            print(
                f"Endpoint '{endpoint_name}' -> "
                f"{TRACKING_URI}/gateway/proxy/{endpoint_name}/v1/chat/completions"
            )
            verify(secret, model_def, endpoint)
    except BootstrapError as e:
        sys.exit(f"Bootstrap failed: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
