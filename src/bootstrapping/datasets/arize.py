"""Pydantic model for Arize (Phoenix) dataset examples."""

from pydantic import BaseModel

from bootstrapping.datasets.model import DatasetRecord


class ArizeExample(BaseModel):
    """One example in an Arize (Phoenix) dataset, as sent by the upload API."""

    example_id: str
    input: dict[str, object]
    output: dict[str, object] = {}
    metadata: dict[str, object] = {}

    @staticmethod
    def to_payload(name: str, examples: list[ArizeExample]) -> dict:
        """Build the /v1/datasets/upload payload for the given examples."""
        return {
            "name": name,
            "inputs": [example.input for example in examples],
            "outputs": [example.output for example in examples],
            "metadata": [example.metadata for example in examples],
            "example_ids": [example.example_id for example in examples],
        }


def to_arize_example(record: DatasetRecord) -> ArizeExample:
    """Map a local dataset record to its Arize representation.

    guidelines and tags (when present) are attached under example metadata.
    """
    metadata: dict[str, object] = {}
    if record.guidelines:
        metadata["guidelines"] = record.guidelines
    if record.tags:
        metadata["tags"] = record.tags
    output: dict[str, object] = {"expected_response": record.expected_response} if record.expected_response else {}
    return ArizeExample(
        example_id=record.unique_id,
        input=record.input.model_dump(),
        output=output,
        metadata=metadata,
    )
