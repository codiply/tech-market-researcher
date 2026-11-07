"""Unit tests for the generic config loader."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ConfigDict

from tech_market_researcher.core.config_loader import ConfigLoader, ConfigLoaderParams, deep_merge


@pytest.mark.parametrize(
    ("configs", "expected"),
    [
        pytest.param([], {}, id="no configs"),
        pytest.param([{"a": 1}], {"a": 1}, id="single config"),
        pytest.param([{"a": 1}, {"b": 2}], {"a": 1, "b": 2}, id="disjoint keys are combined"),
        pytest.param([{"a": 1}, {"a": 2}], {"a": 2}, id="later scalar wins"),
        pytest.param(
            [{"a": {"x": 1, "y": 2}}, {"a": {"y": 3, "z": 4}}],
            {"a": {"x": 1, "y": 3, "z": 4}},
            id="nested dicts merge",
        ),
        pytest.param(
            [{"a": {"b": {"c": 1}}}, {"a": {"b": {"d": 2}}}],
            {"a": {"b": {"c": 1, "d": 2}}},
            id="deeply nested dicts merge",
        ),
        pytest.param([{"a": [1, 2]}, {"a": [3]}], {"a": [3]}, id="lists replace instead of merging"),
        pytest.param([{"a": {"b": 1}}, {"a": 2}], {"a": 2}, id="dict replaced by scalar"),
        pytest.param([{"a": 1}, {"a": {"b": 2}}], {"a": {"b": 2}}, id="scalar replaced by dict"),
        pytest.param([{"a": 1}, {"a": 2}, {"a": 3}], {"a": 3}, id="last of three entries wins"),
    ],
)
def test_deep_merge(configs: list[dict], expected: dict) -> None:
    """Deep-merge combines configs in order, with later entries winning."""
    assert deep_merge(configs) == expected


class SampleConfig(BaseModel):
    """Test config that accepts any keys, so scenarios can stay plain YAML."""

    model_config = ConfigDict(extra="allow")


@dataclass(frozen=True)
class Scenario:
    """Expected outcome for one scenario folder."""

    expected: dict | None = None
    error: type[Exception] | None = None
    params: ConfigLoaderParams = field(default_factory=ConfigLoaderParams)
    config_name: str = "dev"


SCENARIOS_DIR = Path(__file__).parent / "config_loader"

SCENARIOS: dict[str, Scenario] = {
    "named_only": Scenario(expected={"a": 1, "name": "named"}),
    "common_and_named": Scenario(expected={"a": 1, "b": 3, "nested": {"x": 1, "y": 9, "z": 3}}),
    "local_override": Scenario(expected={"a": 1, "b": 9, "c": "local"}),
    "full_stack": Scenario(expected={"a": 1, "b": 20, "c": 300}),
    "mixins": Scenario(expected={"a": 1, "provider": "other", "enabled": True}),
    "local_with_mixins": Scenario(expected={"model": {"name": "local-model"}, "search": {"provider": "beta"}}),
    "lists_replaced": Scenario(expected={"tags": ["c"]}),
    "missing_named_config": Scenario(error=FileNotFoundError),
    "mixins_all_levels": Scenario(
        expected={"app": "local-app", "search": {"provider": "gamma", "timeout": 30}},
        config_name="prd",
    ),
    "custom_local_infix": Scenario(
        expected={"a": 2}, params=ConfigLoaderParams(local_override_filename_infix="machine")
    ),
    "yml_extension": Scenario(expected={"a": 1, "b": 2}, params=ConfigLoaderParams(config_file_extension="yml")),
}


@pytest.mark.parametrize("folder", sorted(p for p in SCENARIOS_DIR.iterdir() if p.is_dir()), ids=lambda p: p.name)
def test_load_scenario(folder: Path) -> None:
    """Each scenario folder in config_loader/ loads into its expected merged dict."""
    scenario = SCENARIOS.get(folder.name)
    assert scenario is not None, f"Scenario folder '{folder.name}' has no expectations defined."
    loader = ConfigLoader(SampleConfig, folder, params=scenario.params)
    if scenario.error:
        with pytest.raises(scenario.error):
            loader.load(scenario.config_name)
    else:
        assert loader.load(scenario.config_name).model_dump() == scenario.expected


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


@pytest.mark.parametrize(
    ("params", "files", "expected"),
    [
        pytest.param(
            ConfigLoaderParams(common_config_name="shared"),
            {"shared.yaml": {"a": 1, "b": 2}, "dev.yaml": {"b": 3}},
            {"a": 1, "b": 3},
            id="common_config_name",
        ),
        pytest.param(
            ConfigLoaderParams(mixins_folder_name="extra_mixins"),
            {"dev.yaml": {"a": 1, "_mixins": ["m"]}, "extra_mixins/m.yaml": {"a": 2}},
            {"a": 2},
            id="mixins_folder_name",
        ),
        pytest.param(
            ConfigLoaderParams(mixins_field="extra_configs"),
            {"dev.yaml": {"a": 1, "extra_configs": ["m"]}, "mixins/m.yaml": {"a": 2}},
            {"a": 2},
            id="mixins_field",
        ),
    ],
)
def test_constructor_parameters(tmp_path: Path, params: ConfigLoaderParams, files: dict, expected: dict) -> None:
    """Non-default constructor parameters (common name, mixins folder, mixins field) take effect."""
    for rel_path, data in files.items():
        _write_yaml(tmp_path / rel_path, data)
    loader = ConfigLoader(SampleConfig, tmp_path, params=params)
    assert loader.load("dev").model_dump() == expected


def test_custom_common_config_name_is_reserved(tmp_path: Path) -> None:
    """Loading the (custom) common config by name raises ValueError."""
    loader = ConfigLoader(SampleConfig, tmp_path, params=ConfigLoaderParams(common_config_name="shared"))
    with pytest.raises(ValueError):
        loader.load("shared")
