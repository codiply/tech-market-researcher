"""MLflow AI Gateway bootstrapping."""

import logging
import os

import requests

from bootstrapping.common.configuration import BootstrapConfig, BudgetPolicyConfig
from bootstrapping.common.exceptions import BootstrapError

logger = logging.getLogger(__name__)

GATEWAY_API_BASE = "api/3.0/mlflow/gateway"
TIMEOUT = 30  # seconds


def _call(server_uri: str, method: str, path: str, *, params=None, body=None) -> dict:
    """Call the gateway REST API and return the parsed JSON response."""
    url = f"{server_uri}/{GATEWAY_API_BASE}/{path}"
    try:
        resp = requests.request(method, url, params=params, json=body, timeout=TIMEOUT)
    except requests.ConnectionError as e:
        raise BootstrapError(f"Cannot reach MLflow server at {server_uri}: {e}") from e
    except requests.Timeout as e:
        raise BootstrapError(f"Timed out calling {url}") from e
    if not resp.ok:
        # MLflow usually returns {"error_code": "...", "message": "..."}
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise BootstrapError(f"{method} {path} failed ({resp.status_code}): {detail}")
    return resp.json()


def _find_secret(server_uri: str, name: str) -> dict | None:
    """Return the gateway secret with the given name, or None.

    secrets/get requires a secret_id, so we search secrets/list by name.
    """
    for secret in _call(server_uri, "GET", "secrets/list").get("secrets", []):
        if secret.get("secret_name") == name:
            return secret
    return None


def _find_model_definition(server_uri: str, provider: str, name: str) -> dict | None:
    """Return the model definition with the given name for the provider, or None."""
    listing = _call(server_uri, "GET", "model-definitions/list", params={"provider": provider})
    for model_def in listing.get("model_definitions", []):
        if model_def.get("name") == name:
            return model_def
    return None


def _ensure_model_definition(server_uri: str, name: str, secret_id: str, provider: str, model_name: str) -> dict:
    """Create or reuse a model definition, re-linking it if it points at another secret."""
    existing = _find_model_definition(server_uri, provider, name)
    if existing is not None:
        if existing.get("secret_id") == secret_id:
            logger.info("Model definition '%s' already exists - reusing.", name)
            return existing
        logger.info(
            "Model definition '%s' is linked to a different secret - re-linking to '%s'.",
            name,
            secret_id,
        )
        return _call(
            server_uri,
            "POST",
            "model-definitions/update",
            body={
                "model_definition_id": existing["model_definition_id"],
                "secret_id": secret_id,
            },
        )["model_definition"]
    logger.info("Creating model definition '%s'.", name)
    return _call(
        server_uri,
        "POST",
        "model-definitions/create",
        body={
            "name": name,
            "secret_id": secret_id,
            "provider": provider,
            "model_name": model_name,
        },
    )["model_definition"]


def _ensure_endpoint(server_uri: str, name: str, model_definition_id: str) -> dict:
    """Create or reuse an endpoint, ensuring a PRIMARY link to the given model."""
    for endpoint in _call(server_uri, "GET", "endpoints/list").get("endpoints", []):
        if endpoint.get("name") == name:
            linked = [
                m
                for m in endpoint.get("model_mappings", [])
                if m.get("model_definition_id") == model_definition_id and m.get("linkage_type") == "PRIMARY"
            ]
            if linked:
                logger.info("Endpoint '%s' already exists - reusing.", name)
                return endpoint
            logger.info("Endpoint '%s' has no PRIMARY link to the model - updating.", name)
            # endpoints/update replaces all model linkages when model_configs
            # is provided.
            return _call(
                server_uri,
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
    logger.info("Creating endpoint '%s'.", name)
    return _call(
        server_uri,
        "POST",
        "endpoints/create",
        body={
            "name": name,
            "model_configs": [{"model_definition_id": model_definition_id, "linkage_type": "PRIMARY"}],
        },
    )["endpoint"]


def _create_models_and_endpoints(server_uri: str, config: BootstrapConfig) -> None:
    """Create one model definition + endpoint per configured model, reusing existing ones."""
    for endpoint in config.mlflow.endpoints:
        connection = config.mlflow.connections_by_key_name[endpoint.api_key_name]
        secret = _find_secret(server_uri, endpoint.api_key_name)
        if secret is None:
            raise BootstrapError(
                f"LLM connection '{endpoint.api_key_name}' not found; " "create the connections first."
            )
        # The provider model id doubles as the definition name; two connections
        # serving the same model id will collide on re-link (last one wins).
        model_def = _ensure_model_definition(
            server_uri,
            endpoint.model_name,
            secret["secret_id"],
            connection.provider,
            endpoint.model_name,
        )
        _ensure_endpoint(server_uri, endpoint.name, model_def["model_definition_id"])
        logger.info("Endpoint '%s' serves model '%s'.", endpoint.name, endpoint.model_name)


def _create_llm_connections(server_uri: str, config: BootstrapConfig) -> None:
    """Create or update one gateway secret per LLM connection.

    Existing secrets are always re-synced, so .env changes (key rotation,
    base URL) take effect on re-run.
    """
    for connection in config.mlflow.llm_connections:
        name = connection.api_key_name
        secret_value = {"api_key": connection.api_key}
        # Non-OpenAI providers (or any custom base URL) are configured via the
        # secret's auth_config. An empty map is an explicit clear, so it is
        # sent when no base URL is configured to drop a stale one.
        if (api_base_url := connection.api_base_url) is not None:
            auth_config = {"auth_mode": "api_key", "api_base": api_base_url}
        else:
            auth_config = {}
        # Bedrock reads its region from auth_config; the gateway does not fall
        # back to the environment for it.
        if connection.region_name is not None:
            auth_config["aws_region_name"] = connection.region_name
        secret = _find_secret(server_uri, name)
        if secret is None:
            _call(
                server_uri,
                "POST",
                "secrets/create",
                body={
                    "secret_name": name,
                    "provider": connection.provider,
                    "secret_value": secret_value,
                    "auth_config": auth_config,
                },
            )
            logger.info("Created LLM connection '%s'.", name)
            continue
        _call(
            server_uri,
            "POST",
            "secrets/update",
            body={
                "secret_id": secret["secret_id"],
                "secret_value": secret_value,
                "auth_config": auth_config,
            },
        )
        logger.info("Updated LLM connection '%s'.", name)


def _budget_policy_matches(policy: dict, duration: dict, action: str) -> bool:
    """True if the policy matches the configured one ignoring amount (global scope, window, action)."""
    return (
        policy.get("target_scope") == "GLOBAL"
        and policy.get("duration") == duration
        and policy.get("budget_action") == action
    )


def _ensure_budget_policy(server_uri: str, budget: BudgetPolicyConfig) -> None:
    """Create or update a global budget policy in place, never creating a duplicate.

    A policy matching the configured window and action (ignoring amount) is
    updated to the configured amount. Several such policies is an error.
    """
    duration = {"unit": budget.reset_period.duration_unit, "value": 1}
    action = budget.action.gateway_action
    policies = _call(server_uri, "GET", "budgets/list").get("budget_policies", [])
    matching = [p for p in policies if _budget_policy_matches(p, duration, action)]
    if len(matching) > 1:
        raise BootstrapError(
            f"Found {len(matching)} global budget policies matching the configured "
            f"({budget.reset_period.value}, {budget.action.value}). "
            "Remove the stale ones in the MLflow UI and re-run."
        )
    if matching:
        _call(
            server_uri,
            "POST",
            "budgets/update",
            body={
                "budget_policy_id": matching[0]["budget_policy_id"],
                "budget_unit": "USD",
                "budget_amount": budget.amount_usd,
                "duration": duration,
                "budget_action": action,
            },
        )
        logger.info(
            "Updated existing global budget policy to %s USD, %s.", budget.amount_usd, budget.reset_period.value
        )
        return
    _call(
        server_uri,
        "POST",
        "budgets/create",
        body={
            "budget_unit": "USD",
            "budget_amount": budget.amount_usd,
            "duration": duration,
            "target_scope": "GLOBAL",
            "budget_action": action,
        },
    )
    logger.info("Created global budget policy (%s USD, %s).", budget.amount_usd, budget.reset_period.value)


def _create_or_update_budget_policies(server_uri: str, config: BootstrapConfig) -> None:
    """Create or update one global budget policy per configured budget policy."""
    for budget in config.mlflow.budget_policies:
        _ensure_budget_policy(server_uri, budget)


def bootstrap_mlflow_ai_gateway(config: BootstrapConfig) -> None:
    """Provision the AI Gateway: LLM connections, models/endpoints, budget policies."""
    server_uri = os.environ["MLFLOW_SERVER_URI"].rstrip("/")
    _create_llm_connections(server_uri, config)
    _create_models_and_endpoints(server_uri, config)
    _create_or_update_budget_policies(server_uri, config)
