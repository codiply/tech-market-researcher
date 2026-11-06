"""Arize (Phoenix) custom LLM provider and model bootstrapping via the Phoenix GraphQL API."""

import logging
import os

import httpx

from bootstrapping.common.configuration import ArizeAIProviderConfig, ArizeModelConfig, BootstrapConfig
from bootstrapping.common.exceptions import BootstrapError

logger = logging.getLogger(__name__)

CREATE_PROVIDER = """
mutation CreateProvider($input: CreateGenerativeModelCustomProviderMutationInput!) {
  createGenerativeModelCustomProvider(input: $input) {
    provider { name }
  }
}
"""

PATCH_PROVIDER = """
mutation PatchProvider($input: PatchGenerativeModelCustomProviderMutationInput!) {
  patchGenerativeModelCustomProvider(input: $input) {
    provider { name }
  }
}
"""

LIST_MODELS = """
query ListModels($first: Int) {
  generativeModels(first: $first) {
    edges { node { id name provider namePattern } }
  }
}
"""

CREATE_MODEL = """
mutation CreateModel($input: CreateModelMutationInput!) {
  createModel(input: $input) {
    model { name }
  }
}
"""

UPDATE_MODEL = """
mutation UpdateModel($input: UpdateModelMutationInput!) {
  updateModel(input: $input) {
    model { name }
  }
}
"""

# Phoenix requires at least one input and one output token price; the local
# models here are free, so both are zero.
ZERO_COSTS = [
    {"tokenType": "input", "costPerMillionTokens": 0.0, "kind": "PROMPT"},
    {"tokenType": "output", "costPerMillionTokens": 0.0, "kind": "COMPLETION"},
]


def _graphql_request(client: httpx.Client, query: str, variables: dict | None = None, context: str = "") -> dict:
    """Run a GraphQL request, raising BootstrapError on the response errors list.

    Phoenix returns HTTP 200 even for GraphQL errors, so the body's errors
    field decides success.
    """
    response = client.post("/graphql", json={"query": query, "variables": variables or {}})
    response.raise_for_status()
    body = response.json()
    if errors := body.get("errors"):
        raise BootstrapError(f"Failed {context or 'GraphQL request'}: {errors}")
    return body["data"]


def _find_provider(client: httpx.Client, name: str) -> dict | None:
    """Return the custom provider with the given name, or None."""
    response = client.get("/v1/custom_model_providers", params={"limit": 100})
    response.raise_for_status()
    return next(
        (provider for provider in response.json().get("data", []) if provider["name"] == name),
        None,
    )


def _find_model(client: httpx.Client, name: str, provider: str | None) -> dict | None:
    """Return the generative model with the given name and provider, or None."""
    data = _graphql_request(client, LIST_MODELS, {"first": 100})
    models = [edge["node"] for edge in data["generativeModels"]["edges"]]
    return next(
        (model for model in models if model["name"] == name and (model["provider"] or "") == (provider or "")),
        None,
    )


def _client_config(provider: ArizeAIProviderConfig) -> dict:
    """Phoenix clientConfig payload for the provider's SDK."""
    settings = provider.sdk_specific_settings
    client_kwargs: dict = {"openaiAuthenticationMethod": {"apiKey": settings.api_key}}
    if settings.api_base_url is not None:
        client_kwargs["openaiClientKwargs"] = {"baseUrl": settings.api_base_url}
    client_kwargs["openaiApiType"] = settings.api_type.graphql_value
    return {"openai": client_kwargs}


def _provider_input(provider: ArizeAIProviderConfig) -> dict:
    """Phoenix create/patch input for the provider (id must be added for patches)."""
    return {
        "name": provider.provider_name,
        "description": provider.description,
        "provider": provider.provider_string,
        "clientConfig": _client_config(provider),
    }


def _ensure_provider(client: httpx.Client, provider: ArizeAIProviderConfig) -> None:
    """Create or update the provider so it matches the config.

    The client config is stored encrypted and never returned, so it cannot be
    diffed: existing providers are always re-synced, so config changes (base
    URL, key rotation) take effect on re-run.
    """
    existing = _find_provider(client, provider.provider_name)
    if existing is None:
        _graphql_request(
            client,
            CREATE_PROVIDER,
            {"input": _provider_input(provider)},
            context=f"create provider '{provider.provider_name}' in Arize",
        )
        logger.info("Created provider '%s' in Arize.", provider.provider_name)
        return
    input_ = {"id": existing["id"], **_provider_input(provider)}
    _graphql_request(
        client,
        PATCH_PROVIDER,
        {"input": input_},
        context=f"update provider '{provider.provider_name}' in Arize",
    )
    logger.info("Updated provider '%s' in Arize.", provider.provider_name)


def _model_input(model: ArizeModelConfig) -> dict:
    """Phoenix create/update input for the model (id must be added for updates)."""
    return {
        "name": model.model_name,
        "provider": model.provider,
        "namePattern": model.name_pattern,
        "costs": ZERO_COSTS,
    }


def _ensure_model(client: httpx.Client, model: ArizeModelConfig) -> None:
    """Create or update the generative model so it matches the config."""
    existing = _find_model(client, model.model_name, model.provider)
    if existing is None:
        _graphql_request(
            client,
            CREATE_MODEL,
            {"input": _model_input(model)},
            context=f"create model '{model.model_name}' in Arize",
        )
        logger.info("Created model '%s' in Arize.", model.model_name)
        return
    if existing["namePattern"] == model.name_pattern:
        logger.info("Model '%s' is up to date in Arize - skipping.", model.model_name)
        return
    input_ = {"id": existing["id"], **_model_input(model)}
    _graphql_request(
        client,
        UPDATE_MODEL,
        {"input": input_},
        context=f"update model '{model.model_name}' in Arize",
    )
    logger.info("Updated model '%s' in Arize (pattern: %s).", model.model_name, model.name_pattern)


def bootstrap_arize_settings(config: BootstrapConfig) -> None:
    """Provision Arize (Phoenix) custom LLM providers and models from the config.

    Providers are created first: models reference their provider by name.
    """
    base_url = os.environ["ARIZE_SERVER_URI"].rstrip("/")
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        for provider in config.arize.ai_providers:
            _ensure_provider(client, provider)
        for model in config.arize.models:
            _ensure_model(client, model)
