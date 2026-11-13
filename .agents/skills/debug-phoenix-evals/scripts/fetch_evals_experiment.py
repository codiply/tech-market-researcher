#!/usr/bin/env python3
"""Fetch a Phoenix evaluation experiment's data: task runs, dataset
examples (inputs + expected outputs), and the per-run evaluator evaluations,
so an agent can identify issues with the evaluation dataset or the evaluators
(configuration or implementation).

Usage:
    fetch_evals_experiment.py [experiment_name_or_id]

With no argument, the most recently created experiment whose name starts with
the evaluation run prefix (``evals-``) is fetched. The argument may be a full
experiment ID or an experiment name (exact match, then prefix match).
"""

import json
import os
import sys

import httpx
from phoenix.client import Client

PHOENIX_SERVER_URI = os.environ.get("PHOENIX_SERVER_URI", "http://localhost:6006").rstrip("/")
RUN_NAME_PREFIX = "evals-"
EXPLANATION_MAX = 300
RESPONSE_MAX = 500


def make_client() -> Client:
    """Return a client for the Phoenix server."""
    return Client(base_url=PHOENIX_SERVER_URI)


def all_experiments(client: Client):
    """Yield all experiments across all datasets."""
    for dataset in client.datasets.list():
        yield from client.experiments.list(dataset_id=dataset["id"])


def find_experiment(client: Client, arg: str | None):
    """Resolve the argument (experiment name or ID) to an experiment; default to the most recent evals experiment."""
    if arg:
        for experiment in all_experiments(client):
            if experiment["name"] == arg:
                return experiment
        for experiment in all_experiments(client):
            if experiment["name"].startswith(arg):
                return experiment
        try:
            return client.experiments.get(experiment_id=arg)
        except httpx.HTTPStatusError:
            sys.exit(f"No experiment found matching '{arg}'.")
    candidates = [e for e in all_experiments(client) if e["name"].startswith(RUN_NAME_PREFIX)]
    if not candidates:
        sys.exit(f"No experiment with name prefix '{RUN_NAME_PREFIX}' found.")
    return max(candidates, key=lambda e: e["created_at"])


def final_output(run) -> str:
    """Return the run's output text, truncated."""
    output = run.get("output")
    if output is None:
        return "<none>"
    return str(output)[:RESPONSE_MAX]


def print_dataset_examples(client: Client, experiment) -> dict[str, int]:
    """Print the dataset version used by the experiment with its example inputs and expected outputs."""
    # Use the version the experiment ran against, not the latest one, since re-bootstrapping
    # re-creates the dataset and the run's example ids point at the older version.
    dataset = client.datasets.get_dataset(dataset=experiment["dataset_id"], version_id=experiment["dataset_version_id"])
    print(f"\nDATASET '{dataset.name}' ({len(dataset.examples)} examples):")
    index_by_example: dict[str, int] = {}
    for i, example in enumerate(dataset.examples):
        input_preview = json.dumps(example["input"], default=str)[:300]
        expected_preview = json.dumps(example["output"], default=str)[:300]
        print(f"  example {i}: input={input_preview}")
        print(f"             expected={expected_preview}")
        # Task runs reference the example's node_id, not its (possibly external) id.
        index_by_example[example["id"]] = i
        index_by_example[example["node_id"]] = i
    return index_by_example


def print_evaluations(evaluations: list, summary: dict[str, dict[str, int]], errors: dict[str, dict[str, int]]) -> None:
    """Print one run's evaluator evaluations and tally their outcomes in the given dicts."""
    for evaluation in sorted(evaluations, key=lambda e: e.name):
        name = evaluation.name
        result = evaluation.result
        error = evaluation.error
        if error:
            bucket = "error"
            errors.setdefault(name, {}).setdefault(str(error)[:200], 0)
            errors[name][str(error)[:200]] += 1
            line = f"  {name}: ERROR :: {str(error)[:160]}"
        else:
            label = result.get("label") if result else None
            score = result.get("score") if result else None
            bucket = (
                f"label:{label}"
                if label is not None
                else (f"score:{round(score, 3)}" if score is not None else "<no result>")
            )
            summary.setdefault(name, {}).setdefault(bucket, 0)
            summary[name][bucket] += 1
            line = f"  {name}: {label if label is not None else score}"
            if result and result.get("explanation"):
                line += f" :: {str(result['explanation'])[:EXPLANATION_MAX]}"
        print(line)


def print_summary(summary: dict[str, dict[str, int]], errors: dict[str, dict[str, int]], n_runs: int) -> None:
    """Print the evaluator outcome summary, flagging evaluators that failed on every run."""
    print("\nEVALUATOR SUMMARY (across all runs; error=failed invocation)")
    for name in sorted(summary):
        counts = ", ".join(f"{k}={v}" for k, v in sorted(summary[name].items()))
        print(f"  {name}: {counts}")
    failed = {n: s for n, s in summary.items() if s.get("error", 0) == n_runs}
    if failed:
        print("\nEvaluators that failed on EVERY run (no result is recorded for these):")
        for name, errs in errors.items():
            if name in failed:
                for msg, count in sorted(errs.items(), key=lambda x: -x[1]):
                    print(f"  {name}: [{count}x] {msg}")


def main() -> None:
    """Fetch and print the evaluation experiment's data, then exit."""
    arg = sys.argv[1] if len(sys.argv) == 2 else None
    client = make_client()
    experiment = find_experiment(client, arg)

    print(
        f"EXPERIMENT {experiment['name']} id={experiment['id']} "
        f"created={experiment['created_at']} project={experiment.get('project_name')}"
    )
    print(
        f"  runs: {experiment['successful_run_count']} successful, "
        f"{experiment['failed_run_count']} failed, {experiment['missing_run_count']} missing "
        f"of {experiment['example_count']} examples"
    )

    examples = print_dataset_examples(client, experiment)

    ran = client.experiments.get_experiment(experiment_id=experiment["id"])
    evaluations_by_run: dict[str, list] = {}
    for evaluation in ran["evaluation_runs"]:
        evaluations_by_run.setdefault(evaluation.experiment_run_id, []).append(evaluation)

    print(f"\nRUNS ({len(ran['task_runs'])})")
    summary: dict[str, dict[str, int]] = {}
    errors: dict[str, dict[str, int]] = {}
    for i, run in enumerate(ran["task_runs"]):
        example = examples.get(run["dataset_example_id"], -1)
        print(f"\nRUN {i} (example {example}, repetition {run['repetition_number']}):")
        if run.get("trace_id"):
            print(f"  trace id: {run['trace_id']}")
        print(f"  output: {final_output(run)}")
        if run.get("error"):
            print(f"  TASK ERROR: {str(run['error'])[:300]}")
        if not evaluations_by_run.get(run["id"]):
            print("  (no evaluations recorded)")
        else:
            print_evaluations(evaluations_by_run[run["id"]], summary, errors)

    print_summary(summary, errors, len(ran["task_runs"]))


if __name__ == "__main__":
    main()
