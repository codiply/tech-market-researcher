---
name: debug-mlflow-evals
description: "Inspect evaluation runs on the MLflow platform: run a fresh evaluation (or take an existing run id/name from the user) and identify issues with the evaluation dataset, the test cases, or the scorers (configuration or implementation). Use when the user asks to inspect or diagnose an MLflow evaluation run. The agent implementation under evaluation is assumed to be fixed; this skill is about the evaluation, not the agent."
---

# Debug MLflow Evals

Inspect an MLflow evaluation run and find issues with the **evaluation dataset** or the **scorers** (configuration or implementation) — not with the agent. The agent implementation is assumed constant and out of scope; do not try to improve it.

## Steps

1. Determine which run to inspect:
   - **User provided a run id or name** → inspect that run.
   - **Otherwise, run a fresh evaluation first**:
     - Make sure the docker compose stack is up: `./stack.sh ps`; if not, `./stack.sh up -d`.
     - Confirm the evaluation dataset has been bootstrapped (fresh stacks run the bootstrap automatically; re-run with `./stack.sh bootstrap` if a dataset is missing).
     - Run the evaluation from the repo root: `uv run evals --platform mlflow`. Honor flags the user gave, e.g. `--evals-config-name`, `--agent-config-name`; otherwise use the defaults. This blocks until the run finishes; the run is named `evals-<timestamp>` and lives in the experiment named by the evaluation config (`config/evaluations/mlflow/<name>.yaml`).
2. Fetch the run data (from the repo root):

   ```bash
   uv run python .agents/skills/debug-mlflow-evals/scripts/fetch_evals_run.py [run_id_or_name]
   ```

   - Pass the run id/name the user gave; with no argument the script fetches the most recent finished run whose name starts with `evals-` — which is the run created in step 1.
   - The output contains the run's metrics, the dataset's records (inputs + expectations), the per-record scorer assessments (with error messages and rationales for failing "no" verdicts), and a summary of scorer outcomes across all records, flagging scorers that failed on every record (those log no metric at all).
3. Read the full output and analyze it with your judgment. You are looking for problems in the evaluation itself — in the dataset/records or in the scorers — while assuming the agent is fixed. Cross-check the pieces against each other: the dataset records, the per-record assessments, the rationales, and the run-level metrics, and use your own judgment to decide what is wrong and why.
4. Report your findings to the user: a list of issues, each with evidence from the output (record indices, error messages, metric values) and which part of the evaluation it affects (dataset or scorers). If nothing is wrong, say so, noting anything suspicious you could not conclusively attribute.

## Rules

- Do not modify the agent implementation or its config to "fix" findings; this skill diagnoses the evaluation.
- Do not modify any datasets or dataset records on the MLflow server — no upserting, editing, or deleting records through the API or directly. Changes to the dataset files in this repo are the only allowed form of a dataset fix.
- If the user did not provide a run id and does not object to running a fresh evaluation, just run one — do not ask.
- The script is read-only; it never mutates MLflow state.
