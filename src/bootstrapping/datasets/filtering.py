"""Filtering dataset records loaded from assets/datasets/."""

from collections.abc import Callable

from bootstrapping.common.configuration import DatasetFiltersConfig
from bootstrapping.datasets.model import DatasetRecord


def apply_filters(records: list[DatasetRecord], filters: DatasetFiltersConfig) -> list[DatasetRecord]:
    """Return the records matching all the configured filters.

    An unset filter (None) matches every record; the filters combine with a
    logical AND.
    """
    checks: list[tuple[bool | None, Callable[[DatasetRecord], bool]]] = [
        (filters.has_expected_response, lambda r: r.expected_response is not None),
        (filters.has_expected_facts, lambda r: r.expected_facts is not None),
        (filters.has_expected_tool_calls, lambda r: r.expected_tool_calls is not None),
        (filters.has_guidelines, lambda r: len(r.guidelines) > 0),
        (filters.has_source_documents, lambda r: len(r.source_documents) > 0),
    ]

    def matches(record: DatasetRecord) -> bool:
        return all(actual(record) is configured for configured, actual in checks if configured is not None)

    return [record for record in records if matches(record)]
