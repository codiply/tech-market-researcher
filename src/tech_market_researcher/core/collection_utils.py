"""Helpers for working with collections."""

from collections.abc import Sequence


def find_duplicates[T](items: Sequence[T]) -> list[T]:
    """Return the items that appear more than once, in order of first appearance."""
    seen: set[T] = set()
    duplicates: list[T] = []
    for item in items:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    return duplicates
