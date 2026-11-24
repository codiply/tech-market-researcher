"""Filtering of Phoenix dataset examples by their record unique id."""

import fnmatch

from phoenix.client.resources.datasets import Dataset


def filter_dataset_examples(dataset: Dataset, patterns: list[str]) -> Dataset:
    """Return a copy of the dataset with only the examples whose id matches a pattern.

    The patterns use fnmatch glob syntax and are matched against each example's id,
    which is the record's unique id ``<dataset>/<file>/<number>`` assigned at
    bootstrap; e.g. ``benchmark/news/001`` selects one record and ``benchmark/news/*``
    every record of a file.
    """
    kept = [
        example for example in dataset.examples if any(fnmatch.fnmatch(example["id"], pattern) for pattern in patterns)
    ]
    data = dataset.to_dict()
    data["examples"] = kept
    return Dataset.from_dict(data)
