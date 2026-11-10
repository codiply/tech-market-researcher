#!/usr/bin/env python3
"""Fetch an MLflow evaluation run's data: metrics, dataset records, and the
per-record scorer matrix, so an agent can identify issues with the evaluation
dataset or the scorers (configuration or implementation).

Usage:
    fetch_evals_run.py [run_id_or_name]

With no argument, the most recent finished run whose name starts with the
evaluation run prefix (``evals-``) is fetched. The argument may be a full run ID
or a run name (exact match, then prefix match).
"""

import json
import os
import sys

# Set before importing mlflow so its agent hints stay disabled.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# pylint: disable=wrong-import-position
import mlflow  # noqa: E402

# pylint: enable=wrong-import-position

TRACKING_URI = os.environ.get("MLFLOW_SERVER_URI", "http://localhost:5000").rstrip("/")
# Set the active tracking URI so lazy dataset-record loading resolves against this
# server instead of the local default store.
mlflow.set_tracking_uri(TRACKING_URI)
RUN_NAME_PREFIX = "evals-"
RATIONALE_MAX = 300
RESPONSE_MAX = 500


def make_client() -> mlflow.MlflowClient:
    """Return a client for the MLflow tracking server."""
    return mlflow.MlflowClient(tracking_uri=TRACKING_URI)


def all_runs(client: mlflow.MlflowClient):
    """Yield all runs across all experiments."""
    for experiment in client.search_experiments():
        yield from client.search_runs(experiment_ids=[experiment.experiment_id], max_results=100)


def find_run(client: mlflow.MlflowClient, arg: str | None):
    """Resolve the argument (run ID or run name) to a run; default to the most recent finished evals run."""
    if arg and len(arg) == 32 and all(c in "0123456789abcdef" for c in arg):
        return client.get_run(arg)
    runs = list(all_runs(client))
    if arg:
        for run in runs:
            if run.data.tags.get("mlflow.runName") == arg:
                return run
        for run in runs:
            if run.data.tags.get("mlflow.runName", "").startswith(arg):
                return run
        sys.exit(f"No run found matching '{arg}'.")
    candidates = [
        r
        for r in runs
        if r.data.tags.get("mlflow.runName", "").startswith(RUN_NAME_PREFIX) and r.info.status == "FINISHED"
    ]
    if not candidates:
        sys.exit(f"No finished run with name prefix '{RUN_NAME_PREFIX}' found.")
    return max(candidates, key=lambda r: r.info.start_time or 0)


def final_response(trace) -> str:
    """Return the text of the last AI message of the trace's root span."""
    try:
        outputs = trace.search_spans()[0].outputs
        if isinstance(outputs, str):
            outputs = json.loads(outputs)
        last = outputs["messages"][-1]
        content = last["content"]
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        return str(content)[:RESPONSE_MAX]
    except json.JSONDecodeError, IndexError, KeyError, TypeError:
        return "<unavailable>"


def run_id_of(trace) -> str | None:
    """Return the evaluation run ID associated with the trace, if any."""
    metadata = trace.info.to_dict().get("trace_metadata") or {}
    value = metadata.get("mlflow.runId") or metadata.get("mlflow.sourceRun") or ""
    return value.rsplit("/", 1)[-1] or None


def print_dataset_records(client: mlflow.MlflowClient, experiment_id: str) -> None:
    """Print all datasets of the experiment with their record inputs and expectations."""
    print("\nDATASETS IN EXPERIMENT")
    for dataset in client.search_datasets(experiment_ids=[experiment_id]):
        data = client.get_dataset(dataset.dataset_id)
        print(f"\nDataset '{dataset.name}' ({len(data.records)} records):")
        for i, record in enumerate(data.records):
            print(f"  record {i}: inputs={json.dumps(record.inputs, default=str)[:300]}")
            print(f"            expectations={json.dumps(record.expectations, default=str)[:300]}")


def print_trace(index: int, trace) -> None:
    """Print one trace with its span types, final response, and scorer assessments."""
    preview = trace.info.to_dict().get("request_preview")
    span_types = sorted({s.attributes.get("mlflow.spanType") for s in trace.search_spans()})
    print(f"\nTRACE {index}: {str(preview)[:100]}")
    print(f"  span types: {span_types}")
    print(f"  final response: {final_response(trace)}")


def print_assessments(trace, summary: dict[str, dict[str, int]], errors: dict[str, dict[str, int]]) -> None:
    """Print the trace's scorer assessments and tally their outcomes in the given dicts."""
    for assessment in sorted(trace.search_assessments(), key=lambda a: a.name):
        name = assessment.name
        if type(assessment).__name__ == "Expectation":
            print(f"  expectation {name}: {str(assessment.value)[:200]}")
            continue
        value = assessment.value
        error = assessment.error_message
        bucket = "error" if error else str(value)
        summary.setdefault(name, {}).setdefault(bucket, 0)
        summary[name][bucket] += 1
        line = f"  {name}: {value if not error else 'ERROR'}"
        if error:
            line += f" :: {error[:160]}"
            errors.setdefault(name, {}).setdefault(error[:200], 0)
            errors[name][error[:200]] += 1
        elif value == "no" and assessment.rationale:
            line += f" :: {assessment.rationale[:RATIONALE_MAX]}"
        print(line)


def print_summary(summary: dict[str, dict[str, int]], errors: dict[str, dict[str, int]], n_traces: int) -> None:
    """Print the scorer outcome summary, flagging scorers that failed on every record."""
    print("\nSCORER SUMMARY (across all records; error=failed invocation)")
    for name in sorted(summary):
        counts = ", ".join(f"{k}={v}" for k, v in sorted(summary[name].items()))
        print(f"  {name}: {counts}")
    failed = {n: s for n, s in summary.items() if s.get("error", 0) == n_traces}
    if failed:
        print("\nScorers that failed on EVERY record (no metric is logged for these):")
        for name, errs in errors.items():
            if name in failed:
                for msg, count in sorted(errs.items(), key=lambda x: -x[1]):
                    print(f"  {name}: [{count}x] {msg}")


def main() -> None:
    """Fetch and print the evaluation run's data, then exit."""
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    client = make_client()
    run = find_run(client, arg)
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

    print(f"RUN {run.data.tags.get('mlflow.runName')} id={run_id} experiment={experiment_id} status={run.info.status}")

    print("\nRUN METRICS")
    for key, value in run.data.metrics.items():
        print(f"  {key} = {value}")

    print_dataset_records(client, experiment_id)

    traces = [t for t in client.search_traces(locations=[experiment_id], max_results=500) if run_id_of(t) == run_id]
    print(f"\nTRACES ({len(traces)})")
    summary: dict[str, dict[str, int]] = {}
    errors: dict[str, dict[str, int]] = {}
    for i, trace in enumerate(traces):
        print_trace(i, trace)
        print_assessments(trace, summary, errors)

    print_summary(summary, errors, len(traces))


if __name__ == "__main__":
    main()
