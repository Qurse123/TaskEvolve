# Experiment: Cost-Aware Co-Optimization of Task-Specific LLM Agents

## 1. One-Sentence Summary

This experiment studies whether an autonomous iterator agent can reduce the cost of a task-oriented LLM agent while preserving task success by jointly optimizing two surfaces:

1. The agent harness.
2. The model specialization pipeline for an open-weight task model.

The experiment uses a frozen τ-bench-family evaluation environment as the source of truth.

## 2. Core Research Question

Given a fixed task environment, fixed benchmark family, fixed evaluation protocol, and fixed optimization budget, what is the cheapest reliable way to build a high-performing task-oriented agent?

The main research question is:

Can an iterator agent find lower-cost τ-bench agent systems by jointly searching over agent harness design and open-weight model specialization, while maintaining task success and policy compliance above a fixed threshold?

The comparison question is:

Which optimization strategy produces the best cost-performance tradeoff?

1. Closed-weight model + static harness.
2. Closed-weight model + iterator-optimized harness.
3. Naive open-weight model + static harness.
4. Specialized open-weight model + static harness.
5. Specialized open-weight model + iterator-optimized harness.
6. Cost-aware joint optimization over harness + model specialization choices.

The goal is not only to maximize benchmark score. The goal is to map the cost-performance frontier and identify which systems deliver the highest task success per unit cost.

## 3. Primary Hypothesis

A cost-aware iterator agent can discover agent systems that preserve τ-bench task success while reducing inference cost by jointly optimizing:

1. Harness design.
2. Open-weight model specialization.
3. Routing and fallback decisions between cheaper and stronger models.

The expected result is that harness optimization gives fast performance gains, model specialization lowers cost, and the best hybrid system lies on a better cost-performance frontier than either harness optimization or model specialization alone.

## 4. Secondary Hypotheses

### 4.1 Harness Optimization Hypothesis

A fixed closed-weight model with an iterator-optimized harness will outperform the same closed-weight model with a static harness under the same evaluation budget.

### 4.2 Model Specialization Hypothesis

A specialized open-weight model will outperform a naive open-weight model on τ-bench-family tasks under the same static harness.

### 4.3 Hybrid Optimization Hypothesis

A specialized open-weight model inside an iterator-optimized harness will outperform either the specialized model alone or the optimized harness alone when measured by cost per successful task.

### 4.4 Cost-Aware Co-Optimization Hypothesis

The best deployable system may not be the highest-scoring system. It may be the cheapest system that satisfies minimum task-success and policy-compliance thresholds.

The experiment should therefore optimize for the Pareto frontier between task success and cost, not only absolute task score.

## 5. What This Experiment Is Not

This experiment is not claiming that autonomous harness optimization itself is novel.

This experiment is not claiming that the iterator agent improves its own intelligence.

This experiment is not claiming that the closed-weight base LLM is being trained.

This experiment is not claiming that benchmark performance equals general intelligence.

This experiment is not proposing a new benchmark.

This experiment is not prompt engineering alone.

This experiment is not a product clone of an adaptive-harness system.

This experiment is a controlled measurement study of where improvements come from in task-oriented LLM agents:

1. Harness optimization.
2. Model specialization.
3. Hybrid harness + model co-optimization.
4. Cost-aware model routing and fallback.

## 6. Contribution

The contribution is not:

"An iterator can improve an agent harness."

That has already been explored by prior adaptive harness and meta-harness work.

The intended contribution is:

"For τ-bench-style stateful tool agents, this experiment measures whether cost-effective performance comes primarily from harness optimization, model specialization, or joint optimization of both."

A stronger contribution is possible if the results show:

1. Harness optimization improves performance quickly but plateaus.
2. Open-weight model specialization reduces cost but may lose robustness.
3. Hybrid optimization recovers robustness at lower cost.
4. Cost-aware routing gives a better Pareto frontier than using one model everywhere.
5. Some harness/model changes transfer across τ-bench-family domains while others overfit.

## 7. Mental Model

There are four separate systems:

1. Base model or model pool.
2. Task-performing agent with an agent harness.
3. Iterator agent.
4. Frozen τ-bench-family evaluator.

The base model or model pool provides the LLM calls used by the task-performing agent.

The task-performing agent is the system that actually attempts the benchmark task.

The agent harness is the surrounding control system that determines how the task-performing agent uses prompts, tools, memory, state tracking, verification, retries, output formatting, model routing, and fallback.

The iterator agent is a separate research agent that proposes modifications to the task-performing system. It does not perform the benchmark task directly.

The iterator agent is the optimizer. The task-performing agent is the system being optimized.

## 8. System Components

### 8.1 Model Pool

The model pool may contain:

1. A strong closed-weight model.
2. A naive open-weight model.
3. A specialized open-weight model.
4. A smaller or quantized version of the specialized model.
5. A fallback model for hard cases.

Examples:

- Claude
- GPT
- Gemini
- Llama
- Qwen
- Mistral
- DeepSeek

In the first milestone, the closed-weight model is fixed.

In later milestones, the open-weight model can be fine-tuned, distilled, quantized, or routed.

### 8.2 Task-Performing Agent

The task-performing agent is the system that attempts the benchmark task.

For τ-bench-family tasks, this means the agent must interact with a simulated user, follow a domain policy, use backend tools correctly, and complete the user's request.

The task-performing agent receives a task input, reasons through the task, optionally calls tools, responds to the user, and attempts to complete the goal.

### 8.3 Agent Harness

The agent harness is the surrounding system that controls how the task-performing agent operates.

The harness may include:

- System prompt
- Task prompt
- Policy instructions
- Policy compression
- Few-shot examples
- Tool schema
- Tool selection policy
- Retrieval policy
- Memory policy
- State tracking
- Context compression
- Planning step
- Reflection step
- Verification step
- Clarification strategy
- Retry policy
- Error recovery
- Output formatting
- Cost controls
- Latency controls
- Logging
- Model routing
- Confidence estimation
- Fallback rules

### 8.4 Model Specialization Pipeline

The model specialization pipeline controls how an open-weight model is adapted to τ-bench-family tasks.

It may include:

- Fine-tuning data selection
- Successful task traces
- Failed task traces with corrections
- Distillation from stronger model trajectories
- Synthetic task generation
- Tool-call supervision
- LoRA configuration
- Quantization configuration
- Training hyperparameters
- Evaluation checkpoints
- Model selection criteria

This pipeline is not part of the MVP milestone, but it is required for the full research paper.

### 8.5 Iterator Agent

The iterator agent is a separate research agent.

Its job is not to solve the benchmark task directly.

Its job is to:

1. Read experiment logs.
2. Inspect failure cases.
3. Inspect cost, latency, turn count, and tool-call count.
4. Propose exactly one change to an allowed surface.
5. Apply the change only to allowed files.
6. Run the benchmark or a benchmark subset.
7. Compare the result to the previous version.
8. Keep the change if it improves the objective.
9. Revert the change if it regresses the objective.
10. Write an experiment report.
11. Repeat under a fixed optimization budget.

The iterator agent may run on a strong closed-weight model.

This does not defeat the purpose of the experiment because the iterator agent is acting as the optimizer/search process, not as the system being evaluated.

### 8.6 Frozen Evaluator

The evaluator scores the task-performing agent.

The evaluator must be external, frozen, and not editable by the iterator agent.

The evaluator should come from the τ-bench benchmark family.

The iterator agent may not modify:

- Benchmark tasks
- User simulator
- Domain database
- Ground truth labels
- Evaluation code
- Scoring logic

## 9. Benchmark Decision

We will use the τ-bench benchmark family as the primary evaluation environment.

The reason is that τ-bench evaluates tool-agent-user interaction in realistic domains. The benchmark is not just testing whether a model knows an answer. It tests whether an agent can complete a task across a dynamic conversation with a simulated user while following domain-specific policies and using API tools correctly.

This makes τ-bench a strong fit for this experiment because the variables we care about are harness behavior, model specialization, tool-use reliability, state tracking, and cost.

The original τ-bench repository includes airline and retail domains, but the repository warns that these task versions are outdated and recommends using newer τ-bench-family versions for the latest fixed tasks and new domains. Therefore, the experiment should target the latest compatible τ-bench-family version if setup is compatible.

If the latest version is too difficult to integrate quickly, the original τ-bench retail domain can be used for early development, but final claims should be made against the latest available τ-bench-family benchmark.

Primary benchmark family:

- τ-bench / newer τ-bench-family versions

Preferred first domain:

- Retail

Secondary domain for transfer:

- Airline or banking

Reason for choosing retail first:

- It is easy to understand.
- It has clear customer-service workflows.
- It requires policy following, tool use, state tracking, and multi-turn recovery.
- Harness changes are likely to matter.
- Cost-sensitive deployment is realistic in customer-service-style agents.

## 10. Why τ-bench Instead of SWE-bench First

SWE-bench is valuable, but it is not the best first benchmark for this hypothesis.

This experiment is about task-oriented agents that must use tools, track state, follow policies, and interact with users.

A good first benchmark should therefore make the harness and the model's tool-use behavior first-class variables.

τ-bench-family tasks require:

- Multi-turn interaction
- Tool calls
- Policy compliance
- User clarification
- State tracking
- Error recovery
- Correct final task completion

These are exactly the surfaces controlled by the agent harness and affected by model specialization.

SWE-bench can be added later as a coding-agent transfer benchmark, but τ-bench is the cleaner first choice.

## 11. Objective Function

### 11.1 Primary Objective

The primary objective is:

Minimize cost per successful task while maintaining task success and policy compliance above fixed thresholds.

Written plainly:

Find the cheapest agent system that still works.

### 11.2 Constraint-Based Objective

The main optimization objective should be:

- Minimize cost per successful task.
- Subject to task success rate >= target threshold.
- Subject to policy compliance >= target threshold.
- Subject to no unacceptable regression on held-out tasks.
- Subject to latency <= maximum allowed latency.

Example threshold:

- Task success rate must be at least 95% of the best observed closed-weight baseline.
- Policy compliance must not decrease by more than an agreed tolerance.
- Cost per successful task should decrease as much as possible.

The exact thresholds should be set before final evaluation.

### 11.3 Secondary Objective

The secondary objective is:

Maximize task success under a fixed cost budget.

This gives two views:

1. Cheapest acceptable system.
2. Best system for a fixed budget.

### 11.4 Pareto Frontier

The experiment should report a Pareto frontier.

A system is Pareto-optimal if no other system is both cheaper and better.

The final result should not be one single winner only. It should show the tradeoff curve between:

- Task success
- Policy compliance
- Cost
- Latency
- Number of turns
- Number of tool calls

## 12. Core Architecture

```text
External τ-bench-family Environment
        ↓
Target Agent Harness
        ↓
Task-Performing Agent
        ↓
Model Pool
        ↓
Model Routing / Fallback
        ↓
Tools + Policy + State Tracking
        ↓
Agent/User Conversation + Tool Calls
        ↓
Frozen Evaluator
        ↓
Score + Cost + Failure Logs
        ↓
Iterator Agent
        ↓
Proposed System Change
        ↓
Allowed Change Check
        ↓
Apply Change or Reject Change
        ↓
Rerun Evaluation
        ↓
Keep or Revert
        ↓
Experiment Log
        ↓
Repeat
```

## 13. Experimental Arms

### 13.1 Arm A: Closed-Weight Model + Static Harness

A fixed closed-weight model is used with the default or minimally modified τ-bench tool-calling harness.

The model, task set, evaluator, user simulator, domain database, and scoring logic remain fixed.

Purpose:

Establish the strong closed-weight baseline.

### 13.2 Arm B: Closed-Weight Model + Iterator-Optimized Harness

The same closed-weight model from Arm A is used.

An iterator agent is allowed to modify only the target agent harness.

The iterator agent may edit:

- System prompt
- Task prompt
- Policy compression
- Tool-selection strategy
- Retry strategy
- Clarification strategy
- Verification strategy
- Memory/state tracking strategy
- Error recovery strategy
- Output formatting
- Cost controls
- Latency controls

The iterator agent may not edit:

- Benchmark tasks
- Evaluator
- User simulator
- Ground truth
- Domain database
- Scoring logic
- Closed-weight model weights

Purpose:

Measure the marginal value of autonomous harness optimization.

### 13.3 Arm C: Naive Open-Weight Model + Static Harness

A naive open-weight model is used inside the same static harness family.

Purpose:

Establish the open-weight baseline before task specialization.

Possible models:

- Qwen
- Llama
- Mistral
- DeepSeek

This arm is required to measure the gain from model specialization.

### 13.4 Arm D: Specialized Open-Weight Model + Static Harness

An open-weight model is fine-tuned, distilled, or adapted for the τ-bench-family task.

The harness remains static.

Purpose:

Measure the marginal value of model specialization alone.

Possible specialization methods:

- Fine-tuning on successful task traces
- Distillation from high-performing closed-model trajectories
- Synthetic data generation
- Adapter training
- Supervised learning on conversation/tool-use traces
- Fine-tuning on corrected failure trajectories

### 13.5 Arm E: Specialized Open-Weight Model + Iterator-Optimized Harness

A specialized open-weight model is placed inside an iterator-optimized harness.

Purpose:

Test whether model specialization and harness optimization compound.

This arm measures whether a specialized model becomes substantially better when paired with a harness optimized for its weaknesses.

### 13.6 Arm F: Cost-Aware Joint Optimization

The iterator agent is allowed to search over both:

1. Harness design.
2. Model specialization and routing decisions.

The iterator agent may edit allowed harness files and allowed model-specialization config files.

Possible edits:

- Compress policy prompts.
- Turn verification on only for uncertain cases.
- Route easy tasks to a cheaper model.
- Route hard tasks to a stronger model.
- Adjust confidence thresholds.
- Modify fine-tuning data selection.
- Modify LoRA or adapter settings.
- Quantize the specialized model.
- Distill trajectories from the best harness.
- Remove expensive reflection steps if they do not improve success.
- Add fallback to a stronger model only after failure signals.

Purpose:

Find Pareto-optimal systems that reduce cost while preserving task success.

This is the main research contribution of the full paper.

## 14. Milestone Plan

### 14.1 Milestone 1: Arm A Baseline

Goal:

Build the τ-bench adapter, static harness baseline, split generation, logging, and repeated baseline runner.

Question:

What is the closed-weight static-harness baseline distribution under the fixed evaluation protocol?

Arm B starts only after the Arm A baseline is reproducible and reviewed.

### 14.2 Milestone 2: Arm C and Arm D

Goal:

Add naive open-weight model and specialized open-weight model.

Question:

How much performance or cost improvement comes from model specialization alone?

### 14.3 Milestone 3: Arm E

Goal:

Place the specialized open-weight model inside an iterator-optimized harness.

Question:

Do harness optimization and model specialization compound?

### 14.4 Milestone 4: Arm F

Goal:

Allow cost-aware joint optimization over harness, model specialization, and routing.

Question:

Can the iterator find a cheaper system that preserves task success and policy compliance?

## 15. Metrics

### 15.1 Primary Metrics

Primary metrics:

- Cost per successful task
- Task success rate / pass rate
- Policy compliance rate

The same primary metrics must be used across all experimental arms.

### 15.2 Secondary Metrics

Secondary metrics:

- Cost per task
- Total token cost
- Input token count
- Output token count
- Tool-call cost
- Training cost
- Amortized training cost per task
- Latency
- Number of turns
- Number of tool calls
- Invalid action rate
- Error rate
- Retry count
- Regression rate
- Performance by task category
- Performance on hard cases
- Robustness on held-out tasks
- Transfer performance across domains

### 15.3 Cost Accounting

The experiment must clearly separate:

1. Inference cost.
2. Tool-call cost.
3. Training/fine-tuning cost.
4. Iterator search cost.
5. Amortized deployment cost.

For deployment comparisons, report both:

- Runtime cost per task.
- Total optimization cost required to discover the system.

A specialized model may be cheap at runtime but expensive to train. The paper must not hide that tradeoff.

### 15.4 Metrics Not to Use

Do not use `val_bpb` unless the experiment is literally training a language model on next-token prediction.

`val_bpb` is appropriate for Karpathy's autoresearch because that experiment optimizes a tiny GPT training loop.

For this experiment, task-level and cost-level metrics are required.

### 15.5 Statistical Reporting Protocol

The benchmark is stochastic because the user simulator and task-performing agent use LLM calls. A single run is not a valid estimate of baseline quality.

For Milestone 1, the precommitted repeat protocol is:

1. Smoke split: run once. It is a wiring check only.
2. Proxy baseline: run 5 repeats with seeds `1001..1005`.
3. Validation baseline: run one blind post-hoc validation event with 5 repeats and seeds `2001..2005`.
4. TAU2 official hidden test: run once at milestone end.

Report proxy and validation numbers as mean ± standard deviation across repeated runs, and keep the raw per-run rows in `experiments/results.csv`. Do not choose `N`, seeds, or task subsets after looking at results.

## 16. Dataset Splits

The benchmark should be split into four evaluation groups.

### 16.1 Train Set

The iterator agent may optimize against this set. In this repo, the operational train subset is called the **proxy** split.

The train set is used to discover candidate harness and model-specialization improvements.

### 16.2 Validation Set

The validation set is used to check whether the final candidate system generalizes beyond the proxy split.

For the cost-controlled TaskEvolve protocol, validation is not exposed to the iterator during optimization and is not run after every candidate change. It runs once as a blind post-hoc validation event after the optimization budget is exhausted. That event may contain repeated stochastic runs for measurement precision, but it remains one validation event for access-control purposes.

A final system should not be claimed as improved solely because it improves the proxy split.

### 16.3 Hidden Test Set

The hidden test set is used only once at the end.

The iterator agent must not see hidden test results during development.

### 16.4 OOD Transfer Set

An out-of-distribution transfer set should be used to measure whether improvements generalize.

For τ-bench-family tasks, this can mean:

- Optimize on retail and test on airline.
- Optimize on retail and test on banking.
- Optimize on one version of the task set and test on perturbed tasks.
- Optimize on original τ-bench and test on a newer τ-bench-family version, if appropriate.

## 17. Overfitting Controls

The experiment must reduce overfitting through the following controls:

1. The iterator agent cannot edit the evaluator.
2. The iterator agent cannot edit the benchmark dataset.
3. The iterator agent cannot edit ground truth labels.
4. The iterator agent cannot edit the user simulator.
5. The iterator agent cannot edit the domain database.
6. The iterator agent cannot see hidden test results.
7. The iterator agent must optimize only on the proxy/train subset; the final candidate is checked once on blind validation after the optimization budget is exhausted.
8. The final test must be run once.
9. The optimization budget must be fixed before the experiment starts.
10. Each experimental arm must receive the same budget class.
11. Improvements must be checked across multiple task subsets.
12. Results must be reported by task category.
13. Complexity should be tracked.
14. Cost should be tracked.
15. Latency should be tracked.
16. The final system should be tested on an OOD benchmark, transfer domain, or perturbed task set.
17. The iterator may not create task-ID-specific hacks.
18. The iterator may not use memorized task IDs, labels, or known benchmark shortcuts.

## 18. Public Benchmark Contamination Concern

A public benchmark may already be partially known to closed-weight models.

This creates a contamination risk.

The experiment reduces this risk by choosing τ-bench-family tasks because task success depends heavily on interactive tool use, policy compliance, and stateful multi-turn behavior rather than static factual knowledge.

However, this does not fully eliminate contamination risk.

Additional protections:

1. Keep each base model fixed within its experimental arm.
2. Compare static harness vs optimized harness using the same base model.
3. Use perturbation sets that change names, IDs, inventory values, dates, order details, and policy wording while preserving task structure.
4. Evaluate transfer from one domain to another.
5. Run hidden test only once.
6. Report results conservatively.

The claim should not be:

"The model became smarter."

The claim should be:

"Under a fixed τ-bench-family evaluation protocol, cost-aware co-optimization improved the cost-performance tradeoff of task-oriented agents."

## 19. Editable and Forbidden Surfaces

### 19.1 Allowed Harness Editable Files

The iterator agent may edit. The list below is the broad conceptual design space
across arms; for **Milestone 2 (Arm B)** the enforced, frozen editable surface is
the narrower set declared in `iterator_agent/allowed_edits.yaml` — exactly five
files: `prompts/system_prompt.j2`, `prompts/policy_summary.j2`,
`prompts/few_shot_examples.j2`, `harness.py`, `model_routing.py`. Prompts are
Jinja2 templates (`.j2`), not Markdown.

```text
target_agent/prompts/system_prompt.j2
target_agent/prompts/policy_summary.j2
target_agent/prompts/few_shot_examples.j2
target_agent/harness.py
target_agent/tool_selection.py
target_agent/context_compression.py
target_agent/memory_policy.py
target_agent/state_tracking.py
target_agent/reflection_policy.py
target_agent/verification_policy.py
target_agent/clarification_policy.py
target_agent/retry_policy.py
target_agent/error_recovery.py
target_agent/output_format.py
target_agent/model_routing.py
target_agent/confidence_estimation.py
target_agent/fallback_policy.py
target_agent/cost_controls.py
```

### 19.2 Allowed Model-Specialization Editable Files

For model-specialization arms, the iterator may edit:

```text
model_specialization/training_data_manifest.yaml
model_specialization/distillation_config.yaml
model_specialization/lora_config.yaml
model_specialization/finetune_config.yaml
model_specialization/quantization_config.yaml
model_specialization/checkpoint_selection.py
model_specialization/synthetic_data_generator.py
model_specialization/trajectory_filter.py
model_specialization/model_registry.yaml
```

### 19.3 Forbidden Files

The iterator agent may not edit:

```text
benchmark/tasks.jsonl
benchmark/train_split.jsonl
benchmark/validation_split.jsonl
benchmark/test_split.jsonl
benchmark/user_simulator.py
benchmark/evaluator.py
benchmark/scoring.py
benchmark/ground_truth.jsonl
benchmark/domain_database.json
benchmark/tools.py
experiment_runner.py
allowed_edits.yaml
cost_accounting.py
```

If τ-bench-family file names differ in the actual repository, these should be mapped to the equivalent files.

## 20. Change Acceptance Rule

Each iteration must make exactly one meaningful change.

A change is accepted only if it improves the objective without violating constraints.

For performance-first arms, a change may be accepted if:

1. It improves the proxy objective under the precommitted repeated-run rule.
2. It does not cause unacceptable regression in policy compliance.
3. It does not cause unacceptable cost or latency inflation.

For cost-aware arms, a change may be accepted if:

1. It reduces cost per successful task.
2. It maintains task success above the required threshold.
3. It maintains policy compliance above the required threshold.
4. It does not increase hidden-risk indicators such as invalid actions, tool misuse, or task-category collapse.

A change is rejected if:

1. The proxy double-run does not improve against the current best proxy distribution.
2. Apparent improvement comes only through unacceptable cost increase.
3. Cost decreases only by sacrificing required task success.
4. The change increases invalid output rate.
5. The change breaks the harness.
6. The change touches forbidden files.
7. The change is too broad to attribute causality.
8. The change creates benchmark-specific hacks.
9. The change depends on memorized task IDs or task-specific shortcuts.

After optimization ends, the final candidate is checked on the blind validation event. If validation fails to preserve the proxy gain within the precommitted guardrails, the final report must describe the system as proxy-overfit rather than as a validated improvement.

## 21. Iteration Protocol

Each iteration follows this loop:

1. Run or load the current best proxy distribution.
2. Iterator agent reads only proxy logs (per-task verdicts plus the TAU2 `SimulationRun` transcript persisted for every task) and allowed source files. The transcripts are the primary cost evidence: the editor sees a per-task cost table (cost, turns, tool calls across all tasks) and turn-by-turn digests of the costliest tasks, to cut cost while holding success.
3. Iterator agent proposes exactly one meaningful change.
4. System checks whether the change touches only allowed files.
5. If the change is invalid, reject immediately.
6. If valid, apply the change.
7. Run proxy eval once with a new logged seed.
8. If the first proxy run improves, run proxy eval again with a different logged seed.
9. Keep the change only if both proxy runs improve against the current best proxy distribution without crossing task success, policy compliance, invalid-action, latency, or cost guardrails.
10. Revert the change if either proxy run fails the rule.
11. Write an experiment note.
12. Commit accepted changes.
13. Log rejected changes.
14. Continue until the fixed optimization budget is exhausted.
15. Run validation once as a blind post-hoc event with the precommitted repeated seed set.
16. Run the hidden test once at milestone end, and report it conservatively even if validation exposes overfitting.

## 22. Experiment Logging

Each iteration should produce a log entry with:

```text
iteration_id
timestamp
benchmark_name
domain
task_subset
experimental_arm
base_model
open_weight_model
specialization_method
harness_version
model_version
changed_surface
changed_file
change_summary
reason_for_change
proxy_repeat_count
proxy_seeds
proxy_task_success_before_mean
proxy_task_success_before_std
proxy_task_success_after_mean
proxy_task_success_after_std
validation_event_id
validation_repeat_count
validation_seeds
validation_task_success_mean
validation_task_success_std
policy_compliance_before
policy_compliance_after
tool_call_correctness_before
tool_call_correctness_after
cost_per_task_before
cost_per_task_after
cost_per_successful_task_before
cost_per_successful_task_after
latency_before
latency_after
turn_count_before
turn_count_after
tool_call_count_before
tool_call_count_after
accepted_or_rejected
reason_accepted_or_rejected
failure_cases_improved
failure_cases_regressed
notes
```

## 23. Repository Structure

A possible repository structure:

```text
cost-aware-agent-cooptimization/
  README.md
  experiment.md

  benchmark/
    tau_bench_adapter/
    splits/
      proxy.json
      validation.json
      tau_test.json
      transfer.json

  target_agent/
    agent.py
    harness.py
    prompts/
      system_prompt.j2
      policy_summary.j2
      few_shot_examples.j2
    policies/
      tool_selection.py
      memory_policy.py
      state_tracking.py
      clarification_policy.py
      verification_policy.py
      retry_policy.py
      error_recovery.py
      model_routing.py
      confidence_estimation.py
      fallback_policy.py
      cost_controls.py
    output_format.py

  model_specialization/
    training_data_manifest.yaml
    distillation_config.yaml
    lora_config.yaml
    finetune_config.yaml
    quantization_config.yaml
    checkpoint_selection.py
    synthetic_data_generator.py
    trajectory_filter.py
    model_registry.yaml

  iterator_agent/
    allowed_edits.yaml      # frozen edit surface + guardrails + double-run protocol
    edit_guard.py           # allowed-change check
    baseline.py             # current-best proxy distribution (comparator input)
    feedback.py             # cost-centric summary + transcript digests (editor input)
    researcher.py           # the editor (only LLM role) + prompts/diagnose.j2, propose_edit.j2
    acceptance.py           # deterministic accept/revert rule
    iteration_log.py        # per-iteration §22 record + accepted/rejected changelogs
    run_iteration.py        # orchestrator: one propose->test->keep/revert cycle

  experiments/
    logs/                   # per-run folders: task_<id>.json + task_<id>_messages.json
    iterations/             # per-iteration records (iteration.json + debug artifacts)
    accepted_changes.md
    rejected_changes.md
    results.csv
    plots/

  scripts/
    run_smoke.py
    run_train_eval.py
    run_iterator.py         # Arm B optimization driver (fixed budget; proxy only)
    run_tau_test.py
    run_transfer.py
    plot_results.py
```

## 24. Success Criteria

### 24.1 Milestone 1 Success Criteria

Milestone 1 is successful if:

1. Frozen split JSON files exist for smoke, proxy, and validation.
2. Smoke passes once and produces logs/traces without paid benchmark spend.
3. Arm A proxy baseline runs `N=5` with seeds `1001..1005`.
4. Arm A validation baseline runs as one blind post-hoc event with `N=5` and seeds `2001..2005`.
5. Results are reported as mean ± standard deviation with raw per-run rows.
6. Logs include task success, policy compliance, cost, latency, turns, tool calls, invalid actions, model, split, seed, and run ID.

### 24.2 Arm B Success Criteria

Arm B is successful relative to Arm A if:

1. Arm B improves validation task success.
2. Arm B does not regress heavily on hidden test.
3. Improvements are not explained purely by increased cost, latency, turns, or tool calls.
4. The iterator agent discovers at least one reusable harness improvement.
5. The system produces interpretable experiment logs.

### 24.3 Milestone 2 Success Criteria

Arm D is successful relative to Arm C if:

1. The specialized open-weight model improves task success over the naive open-weight model.
2. The specialized model reduces cost relative to the closed-weight baseline or improves the cost-performance tradeoff.
3. The gains are not limited to the train set.

### 24.4 Milestone 3 Success Criteria

Arm E is successful if:

1. The specialized model plus optimized harness outperforms the specialized model with static harness.
2. The hybrid system improves cost per successful task compared to the closed-weight baseline.
3. The hybrid system does not collapse on held-out or transfer tasks.

### 24.5 Milestone 4 Success Criteria

Arm F is successful if:

1. The iterator discovers at least one Pareto-optimal system that is cheaper than the closed-weight baseline.
2. The system preserves task success above the chosen threshold.
3. The system preserves policy compliance above the chosen threshold.
4. The cost-performance frontier improves over non-joint baselines.
5. The final report explains which changes produced cost savings and which caused regressions.

## 25. Expected Graphs

### 25.1 Task Success Over Iterations

X-axis:

Iteration number.

Y-axis:

Task success rate.

Lines:

- Closed-weight static harness.
- Closed-weight iterator-optimized harness.
- Naive open-weight static harness.
- Specialized open-weight static harness.
- Specialized open-weight optimized harness.
- Cost-aware joint optimization.

### 25.2 Cost per Successful Task Over Iterations

X-axis:

Iteration number.

Y-axis:

Cost per successful task.

Purpose:

Show whether systems become cheaper while maintaining task success.

### 25.3 Cost vs Task Success Pareto Frontier

X-axis:

Cost per task or cost per successful task.

Y-axis:

Task success rate.

Purpose:

Identify Pareto-optimal systems.

A system is better if it is up and to the left: higher success, lower cost.

Plotting convention: each run is one point and each arm a cloud of its repeated
seeds, with the arm mean drawn as a marker carrying x/y std error bars. Axes
auto-zoom to the data range (with margin) so tightly-clustered arms stay legible;
origin-anchoring is an explicit option (`--from-zero`) for iso-cost-per-success
reading. A Pareto frontier requires ≥2 arms — there is nothing to trade against
otherwise — so a single arm (e.g. the Arm A baseline alone) is instead shown as a
**per-metric distribution**: one dot-plot panel per headline metric (task success
rate, cost per successful task), each panel showing the raw seeds plus mean ± std.
That spread is the noise floor a later arm must beat on the proxy double-run.

### 25.4 Policy Compliance vs Cost

X-axis:

Cost per successful task.

Y-axis:

Policy compliance rate.

Purpose:

Detect systems that become cheap by ignoring policy constraints.

### 25.5 Validation vs Hidden Test Performance

X-axis:

Validation score.

Y-axis:

Hidden test score.

Purpose:

Detect overfitting.

If validation improves but hidden test does not, the system is likely overfitting to the validation distribution.

### 25.6 Domain Transfer Performance

X-axis:

Domain.

Y-axis:

Task success rate and cost per successful task.

Purpose:

Determine whether improvements transfer across retail, airline, banking, or other τ-bench-family domains.

## 26. Final Report Format

The final report should include:

1. Research question.
2. Hypotheses.
3. Related work positioning.
4. Benchmark used.
5. Domain used.
6. Dataset split.
7. Base models.
8. Agent harness design.
9. Model-specialization pipeline.
10. Iterator agent design.
11. Allowed edit surface.
12. Experimental arms.
13. Objective function.
14. Metrics.
15. Cost accounting method.
16. Overfitting controls.
17. Contamination controls.
18. Results table.
19. Pareto frontier graphs.
20. Accepted changes.
21. Rejected changes.
22. Failure analysis.
23. Generalization test.
24. Limitations.
25. Future work.

## 27. Key Risks

### 27.1 Existing Work Overlap

Prior work has already explored autonomous harness optimization.

Mitigation:

Position this project as a controlled cost-performance comparison of harness optimization, model specialization, and joint co-optimization on τ-bench-style task agents.

### 27.2 Benchmark Overfitting

The iterator agent may discover benchmark-specific tricks.

Mitigation:

Use train / validation / hidden test split, domain transfer, and perturbation evaluation.

### 27.3 Eval Exploitation

The iterator agent may optimize for the scorer rather than real task success.

Mitigation:

Freeze evaluator, inspect failure cases, use external benchmark scoring, and report qualitative examples.

### 27.4 Public Benchmark Contamination

The closed-weight model may already know aspects of the benchmark.

Mitigation:

Keep each model fixed within its arm, use task perturbations, and test domain transfer.

### 27.5 Cost Accounting Error

The system may appear cheaper because search cost, training cost, or tool-call cost is ignored.

Mitigation:

Report runtime cost, training cost, iterator search cost, and amortized deployment cost separately.

### 27.6 Cost-Driven Quality Collapse

The iterator may reduce cost by making the system less careful.

Mitigation:

Use hard thresholds for task success, policy compliance, invalid actions, and held-out performance.

### 27.7 Loss of Interpretability

The system may become complex and hard to understand.

Mitigation:

Allow only one meaningful change per iteration and require written experiment notes.

### 27.8 Non-Generalization

The optimized system may work only on one benchmark or one domain.

Mitigation:

Run final transfer tests on a second τ-bench-family domain and perturbed tasks.

## 28. Final Claim Standard

The final claim should be conservative.

Acceptable claim:

"Using a fixed τ-bench-family evaluation environment, we compare harness optimization, model specialization, and joint co-optimization for task-oriented agents. The best systems are selected by cost per successful task subject to task success and policy-compliance thresholds."

Stronger claim only if supported:

"Cost-aware joint optimization over harness design and open-weight model specialization produced a better cost-performance frontier than either harness optimization or model specialization alone."

Strongest claim only if supported:

"The discovered cost-performance improvements transferred across held-out tasks and at least one additional τ-bench-family domain."

Do not claim:

"The model became smarter."

Do not claim:

"The agent is generally better across all tasks."

Do not claim:

"This proves autonomous AI research."

Do not claim:

"We invented autonomous harness optimization."

## 29. Paper Title Candidates

Possible titles:

1. Cost-Aware Co-Optimization of LLM Agent Harnesses and Specialized Models on τ-bench.
2. Harness or Model? A Cost-Performance Study of Task-Oriented LLM Agents.
3. Finding Cheaper Reliable Agents: Joint Harness and Model Specialization on τ-bench.
4. When Should Agents Be Prompted, Routed, or Fine-Tuned? A τ-bench Study.
5. Pareto-Optimal Task Agents Through Harness and Model Co-Optimization.
