"""Helpers for reading the optional record fields the evaluators and scorers receive."""

from collections.abc import Mapping
from typing import Any


def get_field(mapping: Mapping[str, Any] | None, name: str, default: Any = None) -> Any:
    """Return the named field from the mapping, or the default when the mapping or field is missing."""
    return (mapping or {}).get(name, default)


def get_field_list(mapping: Mapping[str, Any] | None, name: str) -> list[Any]:
    """Return the named list field from the mapping, or an empty list when the mapping or field is missing."""
    return get_field(mapping, name) or []
