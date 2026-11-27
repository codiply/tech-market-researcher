"""Unit tests for the generic config loader."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ConfigDict

from tech_market_researcher.core.config_loader import (
    ConfigLoader,
    ConfigLoaderParams,
    ConfigSafeLoader,
    ListMergeBehaviour,
    MergeDirective,
    MergeOperation,
    deep_merge,
)


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
    assert deep_merge(configs, ListMergeBehaviour.OVERWRITE) == expected


@pytest.mark.parametrize(
    ("configs", "expected"),
    [
        pytest.param([], {}, id="no configs"),
        pytest.param([{"a": 1}], {"a": 1}, id="single config"),
        pytest.param([{"a": [1, 2]}, {"a": [3]}], {"a": [1, 2, 3]}, id="lists are extended"),
        pytest.param([{"a": [1, 2]}, {"a": []}], {"a": [1, 2]}, id="empty list appends nothing"),
        pytest.param([{"a": []}, {"a": [1]}], {"a": [1]}, id="items appended to empty list"),
        pytest.param(
            [{"a": [1]}, {"a": [2]}, {"a": [3]}],
            {"a": [1, 2, 3]},
            id="three layers extend in order",
        ),
        pytest.param(
            [{"nested": {"a": [1, 2]}}, {"nested": {"a": [3]}}],
            {"nested": {"a": [1, 2, 3]}},
            id="lists inside nested dicts are extended",
        ),
        pytest.param([{"a": [1, 2]}, {"a": 3}], {"a": 3}, id="list replaced by scalar"),
        pytest.param([{"a": 1}, {"a": [2]}], {"a": [2]}, id="scalar replaced by list"),
        pytest.param([{"a": 1}, {"a": 2}], {"a": 2}, id="later scalar still wins"),
        pytest.param(
            [{"a": {"x": 1}}, {"a": {"y": 2}}],
            {"a": {"x": 1, "y": 2}},
            id="dicts still merge",
        ),
    ],
)
def test_deep_merge_extend(configs: list[dict], expected: dict) -> None:
    """Deep-merge with EXTEND appends later list items onto earlier ones; all other values still use the later entry."""
    assert deep_merge(configs, ListMergeBehaviour.EXTEND) == expected


@pytest.mark.parametrize(
    ("configs", "behaviour", "expected"),
    [
        pytest.param(
            [{"a": {"x": 1, "y": 2}}, {"a": MergeDirective({"y": 3}, MergeOperation.OVERWRITE)}],
            ListMergeBehaviour.OVERWRITE,
            {"a": {"y": 3}},
            id="dict overwrite replaces whole section",
        ),
        pytest.param(
            [{"a": {"x": 1}}, {"a": MergeDirective({"y": 2}, MergeOperation.OVERWRITE)}],
            ListMergeBehaviour.EXTEND,
            {"a": {"y": 2}},
            id="dict overwrite still replaces section under extend behaviour",
        ),
        pytest.param(
            [{"a": {"x": 1}}, {"a": MergeDirective({"y": 2}, MergeOperation.EXTEND)}],
            ListMergeBehaviour.OVERWRITE,
            {"a": {"x": 1, "y": 2}},
            id="extend on a dict falls back to merging",
        ),
        pytest.param(
            [{"a": [1, 2]}, {"a": MergeDirective([3], MergeOperation.EXTEND)}],
            ListMergeBehaviour.OVERWRITE,
            {"a": [1, 2, 3]},
            id="list extend appends under overwrite behaviour",
        ),
        pytest.param(
            [{"a": [1, 2]}, {"a": MergeDirective([3], MergeOperation.OVERWRITE)}],
            ListMergeBehaviour.EXTEND,
            {"a": [3]},
            id="list overwrite replaces under extend behaviour",
        ),
        pytest.param(
            [{"a": MergeDirective([3], MergeOperation.EXTEND)}],
            ListMergeBehaviour.OVERWRITE,
            {"a": [3]},
            id="extend with no earlier list sets the value",
        ),
        pytest.param(
            [{"a": 1}, {"a": MergeDirective([3], MergeOperation.EXTEND)}],
            ListMergeBehaviour.OVERWRITE,
            {"a": [3]},
            id="extend over a non-list earlier value sets the value",
        ),
        pytest.param(
            [{"a": [1]}, {"a": MergeDirective(3, MergeOperation.EXTEND)}],
            ListMergeBehaviour.OVERWRITE,
            {"a": 3},
            id="extend with a non-list value replaces",
        ),
        pytest.param(
            [{"a": 1}, {"a": MergeDirective(2, MergeOperation.OVERWRITE)}],
            ListMergeBehaviour.OVERWRITE,
            {"a": 2},
            id="scalar overwrite",
        ),
        pytest.param(
            [{"nested": {"a": [1, 2]}}, {"nested": {"a": MergeDirective([3], MergeOperation.EXTEND)}}],
            ListMergeBehaviour.OVERWRITE,
            {"nested": {"a": [1, 2, 3]}},
            id="lists inside nested dicts extend via tag",
        ),
        pytest.param(
            [
                {"a": [1]},
                {"a": MergeDirective([2], MergeOperation.EXTEND)},
                {"a": MergeDirective([3], MergeOperation.EXTEND)},
            ],
            ListMergeBehaviour.OVERWRITE,
            {"a": [1, 2, 3]},
            id="three layers extend in order",
        ),
        pytest.param(
            [{"a": 1}, {"b": 2}, {"a": MergeDirective(None, MergeOperation.DELETE)}],
            ListMergeBehaviour.OVERWRITE,
            {"b": 2},
            id="delete removes the key",
        ),
        pytest.param(
            [{"a": MergeDirective(None, MergeOperation.DELETE)}],
            ListMergeBehaviour.OVERWRITE,
            {},
            id="delete of a missing key is a no-op",
        ),
        pytest.param(
            [{"a": {"x": 1, "y": 2}}, {"a": MergeDirective(None, MergeOperation.DELETE)}],
            ListMergeBehaviour.EXTEND,
            {},
            id="delete removes a whole section under extend behaviour",
        ),
        pytest.param(
            [{"nested": {"a": 1, "b": 2}}, {"nested": {"b": MergeDirective(None, MergeOperation.DELETE)}}],
            ListMergeBehaviour.OVERWRITE,
            {"nested": {"a": 1}},
            id="delete of a nested key",
        ),
        pytest.param(
            [{"a": {"x": 1, "y": 2}}, {"a": MergeDirective({"y": 3, "z": 4}, MergeOperation.OVERRIDE_IF_PRESENT)}],
            ListMergeBehaviour.OVERWRITE,
            {"a": {"x": 1, "y": 3, "z": 4}},
            id="override-if-present over an existing section merges",
        ),
        pytest.param(
            [{"b": 1}, {"a": MergeDirective({"x": 1}, MergeOperation.OVERRIDE_IF_PRESENT)}],
            ListMergeBehaviour.OVERWRITE,
            {"b": 1},
            id="override-if-present over a missing key is skipped",
        ),
        pytest.param(
            [{"b": 1}, {"a": {"x": MergeDirective({"p": 1}, MergeOperation.OVERRIDE_IF_PRESENT)}}],
            ListMergeBehaviour.OVERWRITE,
            {"b": 1, "a": {}},
            id="override-if-present in a new subtree skips the key, leaving the empty parent",
        ),
        pytest.param(
            [
                {"a": {"x": 1}},
                {"a": MergeDirective({"y": 2}, MergeOperation.OVERRIDE_IF_PRESENT)},
                {"a": MergeDirective({"z": 3}, MergeOperation.OVERRIDE_IF_PRESENT)},
            ],
            ListMergeBehaviour.OVERWRITE,
            {"a": {"x": 1, "y": 2, "z": 3}},
            id="three layers merge into a present section in order",
        ),
        pytest.param(
            [{"a": 1}, {"a": MergeDirective({"x": 2}, MergeOperation.OVERRIDE_IF_PRESENT)}],
            ListMergeBehaviour.OVERWRITE,
            {"a": {"x": 2}},
            id="override-if-present over a non-dict earlier value replaces",
        ),
        pytest.param(
            [
                {"a": {"x": 1, "y": 2}},
                {
                    "a": MergeDirective(
                        {"y": MergeDirective(3, MergeOperation.OVERWRITE)}, MergeOperation.OVERRIDE_IF_PRESENT
                    )
                },
            ],
            ListMergeBehaviour.OVERWRITE,
            {"a": {"x": 1, "y": 3}},
            id="inner overwrite resolves inside a present override-if-present subtree",
        ),
        pytest.param(
            [
                {"a": {"x": [1, 2]}},
                {
                    "a": MergeDirective(
                        {"x": MergeDirective([3], MergeOperation.EXTEND)}, MergeOperation.OVERRIDE_IF_PRESENT
                    )
                },
            ],
            ListMergeBehaviour.OVERWRITE,
            {"a": {"x": [1, 2, 3]}},
            id="inner extend resolves inside a present override-if-present subtree",
        ),
        pytest.param(
            [
                {"a": {"x": 1, "y": 2}},
                {
                    "a": MergeDirective(
                        {"x": MergeDirective(None, MergeOperation.DELETE)}, MergeOperation.OVERRIDE_IF_PRESENT
                    )
                },
            ],
            ListMergeBehaviour.OVERWRITE,
            {"a": {"y": 2}},
            id="inner delete resolves inside a present override-if-present subtree",
        ),
        pytest.param(
            [
                {"a": {"x": {"p": 1}}},
                {
                    "a": MergeDirective(
                        {"x": MergeDirective({"p": 2}, MergeOperation.OVERRIDE_IF_PRESENT)},
                        MergeOperation.OVERRIDE_IF_PRESENT,
                    )
                },
            ],
            ListMergeBehaviour.OVERWRITE,
            {"a": {"x": {"p": 2}}},
            id="nested override-if-present over present keys merges",
        ),
        pytest.param(
            [
                {"a": {"x": {"p": 1}}},
                {
                    "a": MergeDirective(
                        {"z": MergeDirective({"p": 1}, MergeOperation.OVERRIDE_IF_PRESENT)},
                        MergeOperation.OVERRIDE_IF_PRESENT,
                    )
                },
            ],
            ListMergeBehaviour.OVERWRITE,
            {"a": {"x": {"p": 1}}},
            id="nested override-if-present skips a missing inner key",
        ),
    ],
)
def test_deep_merge_directives(configs: list[dict], behaviour: ListMergeBehaviour, expected: dict) -> None:
    """Merge directive tags force per-node merge behaviour, overriding the loader's list behaviour."""
    assert deep_merge(configs, behaviour) == expected


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
    extra_mixins: list[str] | None = None
    expected_dump: dict | None = None


SCENARIOS_DIR = Path(__file__).parent / "config_loader"

SCENARIOS: dict[str, Scenario] = {
    "named_only": Scenario(
        expected={"a": 1, "name": "named"},
        expected_dump={
            "1_raw/dev.yaml": {"a": 1, "name": "named"},
            "2_configs_with_mixins/dev.yaml": {"a": 1, "name": "named"},
            "3_merged/dev.yaml": {"a": 1, "name": "named"},
            "4_final/dev.yaml": {"a": 1, "name": "named"},
            "extra_mixins_list.yaml": [],
        },
    ),
    "common_and_named": Scenario(
        expected={"a": 1, "b": 3, "nested": {"x": 1, "y": 9, "z": 3}},
        expected_dump={
            "1_raw/_common.yaml": {"a": 1, "b": 2, "nested": {"x": 1, "y": 2}},
            "1_raw/dev.yaml": {"b": 3, "nested": {"y": 9, "z": 3}},
            "2_configs_with_mixins/_common.yaml": {"a": 1, "b": 2, "nested": {"x": 1, "y": 2}},
            "2_configs_with_mixins/dev.yaml": {"b": 3, "nested": {"y": 9, "z": 3}},
            "3_merged/dev.yaml": {"a": 1, "b": 3, "nested": {"x": 1, "y": 9, "z": 3}},
            "4_final/dev.yaml": {"a": 1, "b": 3, "nested": {"x": 1, "y": 9, "z": 3}},
            "extra_mixins_list.yaml": [],
        },
    ),
    "local_override": Scenario(
        expected={"a": 1, "b": 9, "c": "local"},
        expected_dump={
            "1_raw/dev.yaml": {"a": 1, "b": 2},
            "1_raw/dev.local.yaml": {"b": 9, "c": "local"},
            "2_configs_with_mixins/dev.yaml": {"a": 1, "b": 2},
            "2_configs_with_mixins/dev.local.yaml": {"b": 9, "c": "local"},
            "3_merged/dev.yaml": {"a": 1, "b": 9, "c": "local"},
            "4_final/dev.yaml": {"a": 1, "b": 9, "c": "local"},
            "extra_mixins_list.yaml": [],
        },
    ),
    "full_stack": Scenario(
        expected={"a": 1, "b": 20, "c": 300},
        expected_dump={
            "1_raw/_common.yaml": {"a": 1, "b": 2, "c": 3},
            "1_raw/dev.yaml": {"b": 20},
            "1_raw/dev.local.yaml": {"c": 300},
            "2_configs_with_mixins/_common.yaml": {"a": 1, "b": 2, "c": 3},
            "2_configs_with_mixins/dev.yaml": {"b": 20},
            "2_configs_with_mixins/dev.local.yaml": {"c": 300},
            "3_merged/dev.yaml": {"a": 1, "b": 20, "c": 300},
            "4_final/dev.yaml": {"a": 1, "b": 20, "c": 300},
            "extra_mixins_list.yaml": [],
        },
    ),
    "mixins": Scenario(
        expected={"a": 1, "provider": "other", "enabled": True},
        expected_dump={
            "1_raw/dev.yaml": {"a": 1, "provider": "alpha", "_mixins": ["theme"]},
            "2_configs_with_mixins/dev.yaml": {"a": 1, "provider": "other", "enabled": True},
            "0_mixins/theme.yaml": {"provider": "other", "enabled": True},
            "3_merged/dev.yaml": {"a": 1, "provider": "other", "enabled": True},
            "4_final/dev.yaml": {"a": 1, "provider": "other", "enabled": True},
            "extra_mixins_list.yaml": [],
        },
    ),
    "local_with_mixins": Scenario(
        expected={"model": {"name": "local-model"}, "search": {"provider": "beta"}},
        expected_dump={
            "1_raw/dev.yaml": {"model": {"name": "base"}, "search": {"provider": "alpha"}},
            "1_raw/dev.local.yaml": {"_mixins": ["search_override"], "model": {"name": "local-model"}},
            "2_configs_with_mixins/dev.yaml": {"model": {"name": "base"}, "search": {"provider": "alpha"}},
            "2_configs_with_mixins/dev.local.yaml": {"model": {"name": "local-model"}, "search": {"provider": "beta"}},
            "0_mixins/search_override.yaml": {"search": {"provider": "beta"}},
            "3_merged/dev.yaml": {"model": {"name": "local-model"}, "search": {"provider": "beta"}},
            "4_final/dev.yaml": {"model": {"name": "local-model"}, "search": {"provider": "beta"}},
            "extra_mixins_list.yaml": [],
        },
    ),
    "extra_mixins": Scenario(
        expected={"a": 1, "provider": "beta", "theme": "light"},
        extra_mixins=["theme_mix", "override_mix"],
        expected_dump={
            "1_raw/dev.yaml": {"a": 1, "provider": "alpha", "_mixins": ["base_mix"]},
            "2_configs_with_mixins/dev.yaml": {"a": 1, "provider": "beta"},
            "0_mixins/base_mix.yaml": {"provider": "beta"},
            "0_mixins/theme_mix.yaml": {"theme": "dark"},
            "0_mixins/override_mix.yaml": {"theme": "light"},
            "extra_mixins_list.yaml": ["theme_mix", "override_mix"],
            "3_merged/dev.yaml": {"a": 1, "provider": "beta"},
            "4_final/dev.yaml": {"a": 1, "provider": "beta", "theme": "light"},
        },
    ),
    "extra_mixins_duplicate": Scenario(error=ValueError, extra_mixins=["m", "m"]),
    "extra_mixins_missing": Scenario(error=FileNotFoundError, extra_mixins=["missing"]),
    "lists_replaced": Scenario(
        expected={"tags": ["c"]},
        expected_dump={
            "1_raw/_common.yaml": {"tags": ["a", "b"]},
            "1_raw/dev.yaml": {"tags": ["c"]},
            "2_configs_with_mixins/_common.yaml": {"tags": ["a", "b"]},
            "2_configs_with_mixins/dev.yaml": {"tags": ["c"]},
            "3_merged/dev.yaml": {"tags": ["c"]},
            "4_final/dev.yaml": {"tags": ["c"]},
            "extra_mixins_list.yaml": [],
        },
    ),
    "lists_extended": Scenario(
        expected={"tags": ["a", "b", "c"]},
        params=ConfigLoaderParams(list_merge_behaviour=ListMergeBehaviour.EXTEND),
        expected_dump={
            "1_raw/_common.yaml": {"tags": ["a", "b"]},
            "1_raw/dev.yaml": {"tags": ["c"]},
            "2_configs_with_mixins/_common.yaml": {"tags": ["a", "b"]},
            "2_configs_with_mixins/dev.yaml": {"tags": ["c"]},
            "3_merged/dev.yaml": {"tags": ["a", "b", "c"]},
            "4_final/dev.yaml": {"tags": ["a", "b", "c"]},
            "extra_mixins_list.yaml": [],
        },
    ),
    "missing_named_config": Scenario(error=FileNotFoundError),
    "dict_section_overwrite": Scenario(
        expected={"search": {"provider": "beta", "retries": [1]}},
        expected_dump={
            "1_raw/_common.yaml": {"search": {"provider": "alpha", "timeout": 10}},
            "1_raw/dev.yaml": {
                "search": MergeDirective({"provider": "beta", "retries": [1]}, MergeOperation.OVERWRITE)
            },
            "2_configs_with_mixins/_common.yaml": {"search": {"provider": "alpha", "timeout": 10}},
            "2_configs_with_mixins/dev.yaml": {"search": {"provider": "beta", "retries": [1]}},
            "3_merged/dev.yaml": {"search": {"provider": "beta", "retries": [1]}},
            "4_final/dev.yaml": {"search": {"provider": "beta", "retries": [1]}},
            "extra_mixins_list.yaml": [],
        },
    ),
    "list_extend_tag": Scenario(
        expected={"tags": ["a", "b", "c"]},
        expected_dump={
            "1_raw/_common.yaml": {"tags": ["a", "b"]},
            "1_raw/dev.yaml": {"tags": MergeDirective(["c"], MergeOperation.EXTEND)},
            "2_configs_with_mixins/_common.yaml": {"tags": ["a", "b"]},
            "2_configs_with_mixins/dev.yaml": {"tags": ["c"]},
            "3_merged/dev.yaml": {"tags": ["a", "b", "c"]},
            "4_final/dev.yaml": {"tags": ["a", "b", "c"]},
            "extra_mixins_list.yaml": [],
        },
    ),
    "list_overwrite_tag": Scenario(
        expected={"tags": ["c"]},
        params=ConfigLoaderParams(list_merge_behaviour=ListMergeBehaviour.EXTEND),
        expected_dump={
            "1_raw/_common.yaml": {"tags": ["a", "b"]},
            "1_raw/dev.yaml": {"tags": MergeDirective(["c"], MergeOperation.OVERWRITE)},
            "2_configs_with_mixins/_common.yaml": {"tags": ["a", "b"]},
            "2_configs_with_mixins/dev.yaml": {"tags": ["c"]},
            "3_merged/dev.yaml": {"tags": ["c"]},
            "4_final/dev.yaml": {"tags": ["c"]},
            "extra_mixins_list.yaml": [],
        },
    ),
    "mixin_list_extend": Scenario(
        expected={"tags": ["x", "y"]},
        expected_dump={
            "1_raw/dev.yaml": {"tags": ["x"], "_mixins": ["tags_mix"]},
            "2_configs_with_mixins/dev.yaml": {"tags": ["x", "y"]},
            "0_mixins/tags_mix.yaml": {"tags": MergeDirective(["y"], MergeOperation.EXTEND)},
            "3_merged/dev.yaml": {"tags": ["x", "y"]},
            "4_final/dev.yaml": {"tags": ["x", "y"]},
            "extra_mixins_list.yaml": [],
        },
    ),
    "delete_tag": Scenario(
        expected={"search": {"provider": "alpha"}},
        expected_dump={
            "1_raw/_common.yaml": {"search": {"provider": "alpha", "timeout": 10}},
            "1_raw/dev.yaml": {"search": {"timeout": MergeDirective(None, MergeOperation.DELETE)}},
            "2_configs_with_mixins/_common.yaml": {"search": {"provider": "alpha", "timeout": 10}},
            "2_configs_with_mixins/dev.yaml": {"search": {}},
            "3_merged/dev.yaml": {"search": {"provider": "alpha"}},
            "4_final/dev.yaml": {"search": {"provider": "alpha"}},
            "extra_mixins_list.yaml": [],
        },
    ),
    "delete_new_subtree": Scenario(
        expected={"name": "dev", "search": {"provider": "beta", "timeout": 20}},
        extra_mixins=["new_subtree_mix"],
        expected_dump={
            "1_raw/dev.yaml": {"name": "dev"},
            "2_configs_with_mixins/dev.yaml": {"name": "dev"},
            "0_mixins/new_subtree_mix.yaml": {
                "search": {
                    "provider": "beta",
                    "timeout": 20,
                    "tags": MergeDirective(None, MergeOperation.DELETE),
                }
            },
            "3_merged/dev.yaml": {"name": "dev"},
            "4_final/dev.yaml": {"name": "dev", "search": {"provider": "beta", "timeout": 20}},
            "extra_mixins_list.yaml": ["new_subtree_mix"],
        },
    ),
    "delete_section": Scenario(
        expected={},
        expected_dump={
            "1_raw/_common.yaml": {"search": {"provider": "alpha", "timeout": 10}},
            "1_raw/dev.yaml": {"search": MergeDirective(None, MergeOperation.DELETE)},
            "2_configs_with_mixins/_common.yaml": {"search": {"provider": "alpha", "timeout": 10}},
            "2_configs_with_mixins/dev.yaml": {},
            "3_merged/dev.yaml": {},
            "4_final/dev.yaml": {},
            "extra_mixins_list.yaml": [],
        },
    ),
    "override_if_present_present": Scenario(
        expected={"name": "dev", "search": {"provider": "alpha", "timeout": 20}},
        expected_dump={
            "1_raw/_common.yaml": {"search": {"provider": "alpha", "timeout": 10}},
            "1_raw/dev.yaml": {"name": "dev", "_mixins": ["timeout_mix"]},
            "0_mixins/timeout_mix.yaml": {
                "search": MergeDirective({"timeout": 20}, MergeOperation.OVERRIDE_IF_PRESENT)
            },
            "2_configs_with_mixins/_common.yaml": {"search": {"provider": "alpha", "timeout": 10}},
            "2_configs_with_mixins/dev.yaml": {"name": "dev"},
            "3_merged/dev.yaml": {"name": "dev", "search": {"provider": "alpha", "timeout": 20}},
            "4_final/dev.yaml": {"name": "dev", "search": {"provider": "alpha", "timeout": 20}},
            "extra_mixins_list.yaml": [],
        },
    ),
    "override_if_present_absent": Scenario(
        expected={"name": "dev", "cache": {"enabled": True}},
        expected_dump={
            "1_raw/dev.yaml": {"name": "dev", "_mixins": ["cache_mix"]},
            "0_mixins/cache_mix.yaml": {
                "cache": {
                    "enabled": True,
                    "eviction": MergeDirective({"strategy": "lru"}, MergeOperation.OVERRIDE_IF_PRESENT),
                }
            },
            "2_configs_with_mixins/dev.yaml": {"name": "dev", "cache": {"enabled": True}},
            "3_merged/dev.yaml": {"name": "dev", "cache": {"enabled": True}},
            "4_final/dev.yaml": {"name": "dev", "cache": {"enabled": True}},
            "extra_mixins_list.yaml": [],
        },
    ),
    "override_if_present_scalar": Scenario(error=ValueError),
    "mixins_all_levels": Scenario(
        expected={"app": "local-app", "search": {"provider": "gamma", "timeout": 30}},
        config_name="prd",
        expected_dump={
            "1_raw/_common.yaml": {"app": "base-app", "_mixins": ["common_mix"]},
            "1_raw/prd.yaml": {"app": "prd-app", "_mixins": ["prd_mix"]},
            "1_raw/prd.local.yaml": {"app": "local-app", "_mixins": ["local_mix"]},
            "2_configs_with_mixins/_common.yaml": {"app": "base-app", "search": {"provider": "alpha", "timeout": 30}},
            "2_configs_with_mixins/prd.yaml": {"app": "prd-app", "search": {"provider": "other"}},
            "2_configs_with_mixins/prd.local.yaml": {"app": "local-app", "search": {"provider": "gamma"}},
            "0_mixins/common_mix.yaml": {"search": {"provider": "alpha", "timeout": 30}},
            "0_mixins/prd_mix.yaml": {"search": {"provider": "other"}},
            "0_mixins/local_mix.yaml": {"search": {"provider": "gamma"}},
            "3_merged/prd.yaml": {"app": "local-app", "search": {"provider": "gamma", "timeout": 30}},
            "4_final/prd.yaml": {"app": "local-app", "search": {"provider": "gamma", "timeout": 30}},
            "extra_mixins_list.yaml": [],
        },
    ),
    "custom_local_infix": Scenario(
        expected={"a": 2},
        params=ConfigLoaderParams(local_override_filename_infix="machine"),
        expected_dump={
            "1_raw/dev.yaml": {"a": 1},
            "1_raw/dev.machine.yaml": {"a": 2},
            "2_configs_with_mixins/dev.yaml": {"a": 1},
            "2_configs_with_mixins/dev.machine.yaml": {"a": 2},
            "3_merged/dev.yaml": {"a": 2},
            "4_final/dev.yaml": {"a": 2},
            "extra_mixins_list.yaml": [],
        },
    ),
    "yml_extension": Scenario(
        expected={"a": 1, "b": 2},
        params=ConfigLoaderParams(config_file_extension="yml"),
        expected_dump={
            "1_raw/_common.yml": {"a": 1},
            "1_raw/dev.yml": {"b": 2},
            "2_configs_with_mixins/_common.yml": {"a": 1},
            "2_configs_with_mixins/dev.yml": {"b": 2},
            "3_merged/dev.yml": {"a": 1, "b": 2},
            "4_final/dev.yml": {"a": 1, "b": 2},
            "extra_mixins_list.yaml": [],
        },
    ),
}


@pytest.mark.parametrize("folder", sorted(p for p in SCENARIOS_DIR.iterdir() if p.is_dir()), ids=lambda p: p.name)
def test_load_scenario(folder: Path) -> None:
    """Each scenario folder in config_loader/ loads into its expected merged dict."""
    scenario = SCENARIOS.get(folder.name)
    assert scenario is not None, f"Scenario folder '{folder.name}' has no expectations defined."
    loader = ConfigLoader(SampleConfig, folder, params=scenario.params)
    if scenario.error:
        with pytest.raises(scenario.error):
            loader.load(scenario.config_name, extra_mixins=scenario.extra_mixins)
    else:
        assert loader.load(scenario.config_name, extra_mixins=scenario.extra_mixins).model_dump() == scenario.expected


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


@pytest.mark.parametrize(
    "folder",
    sorted(name for name, scenario in SCENARIOS.items() if scenario.error is None),
    ids=lambda name: name,
)
def test_debug_dump(tmp_path: Path, folder: str) -> None:
    """Each scenario folder dumps its rendered configs under the configured debug dir."""
    scenario = SCENARIOS[folder]
    params = scenario.params.model_copy(update={"debug_dir_path": tmp_path / "debug"})
    loader = ConfigLoader(SampleConfig, SCENARIOS_DIR / folder, params=params)
    assert loader.load(scenario.config_name, extra_mixins=scenario.extra_mixins).model_dump() == scenario.expected
    debug_dir = tmp_path / "debug"
    dumped = {
        p.relative_to(tmp_path / "debug").as_posix(): yaml.load(p.read_text(), Loader=ConfigSafeLoader)
        for p in debug_dir.rglob("*")
        if p.suffix in (".yaml", ".yml")
    }
    assert dumped == scenario.expected_dump


@pytest.mark.parametrize(
    "folder",
    sorted(name for name, scenario in SCENARIOS.items() if scenario.error is None),
    ids=lambda name: name,
)
def test_no_debug_dir_dumps_nothing(folder: str) -> None:
    """Without a debug dir, loading writes no files into the config folder."""
    scenario = SCENARIOS[folder]
    config_dir = SCENARIOS_DIR / folder
    before = {p.relative_to(config_dir).as_posix() for p in config_dir.rglob("*")}
    loader = ConfigLoader(SampleConfig, config_dir, params=scenario.params)
    assert loader.load(scenario.config_name, extra_mixins=scenario.extra_mixins).model_dump() == scenario.expected
    after = {p.relative_to(config_dir).as_posix() for p in config_dir.rglob("*")}
    assert after == before
