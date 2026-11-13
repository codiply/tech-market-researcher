"""Pydantic models for dataset records stored in assets/datasets/."""

from tech_market_researcher.core.common import BasePydanticModel


class Input(BasePydanticModel):
    """What the agent receives for a record."""

    messages: list[dict[str, str]]
    context: dict[str, str] = {}


class ExpectedToolCall(BasePydanticModel):
    """A tool call expected for a record."""

    name: str
    arguments: dict[str, object] | None = None


class DatasetRecord(BasePydanticModel):
    """One record in a dataset YAML file."""

    unique_id: str
    input: Input
    disabled: bool = False
    expected_response: object | None = None
    expected_facts: list[str] | None = None
    expected_tool_calls: list[ExpectedToolCall] | None = None
    guidelines: list[str] = []
    tags: dict[str, str] = {}
    source_documents: list[str] = []
