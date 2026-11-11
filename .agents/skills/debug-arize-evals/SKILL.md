---
name: debug-arize-evals
description: "Inspect evaluation experiments on the Arize (Phoenix) platform: run a fresh evaluation (or take an existing experiment name/id from the user) and identify issues with the evaluation dataset, the test cases, or the evaluators (configuration or implementation). Use when the user asks to inspect or diagnose an Arize evaluation experiment. The agent implementation under evaluation is assumed to be fixed; this skill is about the evaluation, not the agent."
---

# Debug Arize Evals

Inspect an Arize (Phoenix) evaluation experiment and find issues with the **evaluation dataset** or the **evaluators** (configuration or implementation) — not with the agent. The agent implementation is assumed constant and out of scope; do not try to improve it.

All access is through the Phoenix client / REST API only. Do not connect to or query the Arize Postgres database in any way.

## Steps

1. Determine which experiment to inspect:
   - **User provided an experiment name or id** → inspect that experiment.
   - **Otherwise, run a fresh evaluation first**:
     - Make sure the docker compose stack is up: `./stack.sh ps`; if not, `./stack.sh up -d`.
     - Confirm the evaluation dataset has been bootstrapped (fresh stacks run the bootstrap automatically; re-run with `./stack.sh bootstrap` if a dataset is missing).
     - Run the evaluation from the repo root: `uv run evals --platform arize`. Honor flags the user gave, e.g. `--evals-config-name`, `--agent-config-name`; otherwise use the defaults. This blocks until the experiment finishes; the experiment is named `evals-<timestamp>` and is attached to the dataset named in the evaluation config (`config/evaluations/arize/<name>.yaml`).
2. Fetch the experiment data (from the repo root):

   ```bash
   uv run python .agents/skills/debug-arize-evals/scripts/fetch_evals_experiment.py [experiment_name_or_id]
   ```

   - Pass the experiment name/id the user gave; with no argument the script fetches the most recently created experiment whose name starts with `evals-` — which is the experiment created in step 1.
   - The output contains the experiment's run counts, the dataset's examples (inputs + expected outputs), the per-run agent output, the per-run evaluator evaluations (with errors and explanations), and a summary of evaluator outcomes across all runs, flagging evaluators that failed on every run (those record no result at all).
   - The summary only covers evaluators that logged results; an evaluator that logged nothing is absent from it, so cross-check the evaluators listed in the evaluation config (`config/evaluations/arize/*.yaml`) against the summary and treat any missing one as a likely failure.
3. Read the full output and analyze it with your judgment. You are looking for problems in the evaluation itself — in the dataset/examples or in the evaluators — while assuming the agent is fixed. Cross-check the pieces against each other: the dataset examples, the agent outputs, the evaluator results and explanations, and the run-level counts, and use your own judgment to decide what is wrong and why. Check that every configured evaluator appears in the summary — a whole evaluator missing from the results (e.g. all its judge invocations failed) is itself a finding, distinct from per-record errors.
4. Report your findings to the user: a list of issues, each with evidence from the output (example indices, error messages, result values) and which part of the evaluation it affects (dataset or evaluators). If nothing is wrong, say so, noting anything suspicious you could not conclusively attribute.

## Rules

- Do not modify the agent implementation or its config to "fix" findings; this skill diagnoses the evaluation.
- Do not modify any datasets or dataset examples on the Arize server — no adding, editing, or deleting examples through the API or directly. Changes to the dataset files in this repo are the only allowed form of a dataset fix.
- Do not access the Arize database (Postgres) directly; the Phoenix client / REST API is the only allowed access path.
- If the user did not provide an experiment name and does not object to running a fresh evaluation, just run one — do not ask.
- The script is read-only; it never mutates Arize state.
