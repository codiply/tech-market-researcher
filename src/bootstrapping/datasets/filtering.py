"""Filtering dataset records loaded from assets/datasets/."""

from bootstrapping.common.configuration import DatasetFiltersConfig
from bootstrapping.datasets.model import DatasetRecord


def apply_filters(records: list[DatasetRecord], filters: DatasetFiltersConfig) -> list[DatasetRecord]:
    """Return the records matching all the configured filters.

    An unset filter (None) matches every record; the filters combine with a
    logical AND.
    """

    def matches(record: DatasetRecord) -> bool:
        if filters.has_expected_response is not None:
            has_expected_response = record.expected_response is not None
            if has_expected_response != filters.has_expected_response:
                return False
        if filters.has_expected_facts is not None:
            has_expected_facts = record.expected_facts is not None
            if has_expected_facts != filters.has_expected_facts:
                return False
        if filters.has_expected_tool_calls is not None:
            has_expected_tool_calls = record.expected_tool_calls is not None
            if has_expected_tool_calls != filters.has_expected_tool_calls:
                return False
        if filters.has_guidelines is not None:
            has_guidelines = len(record.guidelines) > 0
            if has_guidelines != filters.has_guidelines:
                return False
        return True

    return [record for record in records if matches(record)]
