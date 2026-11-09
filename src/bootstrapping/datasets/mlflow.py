"""Pydantic model for MLflow evaluation dataset records."""

import hashlib
import json

from pydantic import BaseModel

from bootstrapping.datasets.model import DatasetRecord


def record_input_hash(inputs: dict) -> str:
    """Stable hash of a record's inputs; MLflow uses it as its record dedup key."""
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


class MlflowExpectations(BaseModel):
    """The expectations attached to an MLflow dataset record."""

    guidelines: list[str] | None = None
    expected_response: object | None = None
    expected_facts: list[str] | None = None
    expected_tool_calls: list[dict[str, object]] | None = None


class MlflowDatasetRecord(BaseModel):
    """One record as stored in an MLflow evaluation dataset."""

    inputs: dict[str, object]
    expectations: MlflowExpectations | None = None
    tags: dict[str, str] | None = None

    def input_hash(self) -> str:
        """Stable hash of the record's inputs; MLflow uses it as its dedup key."""
        return record_input_hash(self.inputs)


def to_mlflow_record(record: DatasetRecord) -> MlflowDatasetRecord:
    """Map a local dataset record to its MLflow representation."""
    expectations = (
        MlflowExpectations(
            guidelines=record.guidelines or None,
            expected_response=record.expected_response or None,
            expected_facts=record.expected_facts or None,
            expected_tool_calls=(
                [call.model_dump(exclude_none=True) for call in record.expected_tool_calls]
                if record.expected_tool_calls
                else None
            ),
        )
        if (record.guidelines or record.expected_response or record.expected_facts or record.expected_tool_calls)
        else None
    )
    return MlflowDatasetRecord(
        inputs=record.input.model_dump(),
        expectations=expectations,
        tags=record.tags or None,
    )
