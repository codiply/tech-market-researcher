"""Pydantic models for dataset records stored in assets/datasets/."""

from pydantic import BaseModel


class Input(BaseModel):
    """What the agent receives for a record."""

    messages: list[dict[str, str]]
    context: dict[str, str] = {}


class ExpectedToolCall(BaseModel):
    """A tool call expected for a record."""

    name: str
    arguments: dict[str, object] | None = None


class DatasetRecord(BaseModel):
    """One record in a dataset YAML file."""

    unique_id: str
    input: Input
    disabled: bool = False
    expected_response: object | None = None
    expected_facts: list[str] | None = None
    expected_tool_calls: list[ExpectedToolCall] | None = None
    guidelines: list[str] = []
    tags: dict[str, str] = {}
