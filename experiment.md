# Experiment: Autonomous Optimization of Task-Specific Agent Harnesses

## 1. One-Sentence Summary

This experiment tests whether a strong iterator agent can autonomously improve the task performance of a separate LLM-powered task agent by modifying only its agent harness, using a frozen τ-bench-family benchmark as the source of truth.

## 2. Core Research Question

Given a fixed task environment, fixed benchmark, fixed model, fixed evaluation protocol, and fixed optimization budget, can an iterator agent improve task performance by autonomously modifying the task-performing agent's harness?

The broader research question is:

Which optimization strategy produces the best task performance?

1. Closed-weight model + static harness
2. Closed-weight model + iterator-optimized harness
3. Specialized open-weight model + static harness
4. Specialized open-weight model + iterator-optimized harness

The MVP will focus only on the first two arms.

## 3. Working Hypothesis

A strong closed-weight iterator agent can improve the performance of a separate task-performing agent by modifying the task agent's harness while leaving the base model, benchmark tasks, user simulator, domain database, ground truth, and evaluator unchanged.

The expected MVP result is that an iterator-optimized harness around a fixed closed-weight model will outperform a static harness on τ-bench-family tasks under the same task budget.

The broader expected result is that a hybrid system, where a specialized open-weight model is placed inside an iterator-optimized harness, may outperform both harness optimization alone and model specialization alone.

## 4. What This Experiment Is Not

This experiment is not claiming that the iterator agent improves its own intelligence.

This experiment is not claiming that the closed-weight base LLM is being trained.

This experiment is not claiming that benchmark performance equals general intelligence.

This experiment is not proposing a new benchmark.

This experiment is not prompt engineering alone.

This experiment is an autonomous search process over agent-system design choices, measured against a frozen external benchmark.

## 5. Mental Model

There are three separate systems:

1. Base LLM
2. Task-performing agent with an agent harness
3. Iterator agent

The base LLM is the model called by the task-performing agent.

The task-performing agent is the system that actually attempts the benchmark task.

The agent harness is the surrounding control system that determines how the task-performing agent uses prompts, tools, memory, retrieval, verification, retries, and output formatting.

The iterator agent is a separate research agent that proposes modifications to the task-performing agent's harness. It does not perform the benchmark task directly.

The iterator agent is the optimizer. The task-performing agent is the system being optimized.

## 6. System Components

### 6.1 Base LLM

The base LLM is the model called by the task-performing agent.

Examples:

- Claude
- GPT
- Gemini
- Llama
- Qwen
- DeepSeek

In the MVP, the base LLM weights are frozen. The base LLM is not trained, fine-tuned, or modified during the harness optimization loop.

### 6.2 Task-Performing Agent

The task-performing agent is the system that attempts the benchmark task.

For τ-bench-family tasks, this means the agent must interact with a simulated user, follow a domain policy, use backend tools correctly, and complete the user's request.

The task-performing agent receives a task input, reasons through the task, optionally calls tools, responds to the user, and attempts to complete the goal.

### 6.3 Agent Harness

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

The harness is the main editable surface in the MVP.

### 6.4 Iterator Agent

The iterator agent is a separate research agent.

Its job is not to solve the benchmark task directly.

Its job is to:

1. Read experiment logs
2. Inspect failure cases
3. Propose exactly one change to the target agent harness
4. Apply the change only to allowed files
5. Run the benchmark
6. Compare the result to the previous version
7. Keep the change if it improves validation performance
8. Revert the change if it regresses performance
9. Write a short experiment report
10. Repeat under a fixed iteration budget

The iterator agent may run on a strong closed-weight model.

This does not defeat the purpose of the experiment because the iterator agent is acting as the optimizer/search process, not as the system being evaluated.

### 6.5 Frozen Evaluator

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

## 7. Benchmark Decision

We will use the τ-bench benchmark family as the primary evaluation environment.

The reason is that τ-bench evaluates tool-agent-user interaction in realistic domains. The benchmark is not just testing whether a model knows an answer. It tests whether an agent can complete a task across a dynamic conversation with a simulated user while following domain-specific policies and using API tools correctly.

This makes τ-bench a strong fit for this experiment because the variable we care about is the agent harness.

The original τ-bench repository includes airline and retail domains, but the repository warns that these task versions are outdated and recommends using τ³-bench for the latest fixed tasks and new domains. Therefore, the experiment should target τ³-bench if setup is compatible.

If τ³-bench is too difficult to integrate quickly, the original τ-bench retail domain can be used for MVP development, but final claims should be made against the latest available τ-bench-family benchmark.

Primary benchmark family:

- τ-bench / τ³-bench

Preferred first domain:

- Retail

Secondary domain for transfer:

- Airline or banking

Reason for choosing retail first:

- It is easy to understand.
- It has clear customer-service workflows.
- It requires policy following, tool use, state tracking, and multi-turn recovery.
- Harness changes are likely to matter.

Primary metric:

- Task success rate / pass rate

Secondary metrics:

- Policy compliance
- Tool-call correctness
- Cost per task
- Cost per successful task
- Latency
- Number of turns
- Number of tool calls
- Invalid action rate
- Regression rate by task type

## 8. Why τ-bench Instead of SWE-bench First

SWE-bench is a valuable benchmark, but it is not the best first benchmark for this hypothesis.

This experiment is about whether an iterator agent can improve an agent harness.

A good first benchmark should therefore make the harness a first-class variable.

τ-bench-family tasks require:

- Multi-turn interaction
- Tool calls
- Policy compliance
- User clarification
- State tracking
- Error recovery
- Correct final task completion

These are exactly the surfaces controlled by the agent harness.

SWE-bench can be added later as a coding-agent transfer benchmark, but τ-bench is the cleaner first choice.

## 9. Core Architecture

```text
External τ-bench / τ³-bench Environment
        ↓
Target Agent Harness
        ↓
Task-Performing Agent
        ↓
Base LLM + Tools + Policy + State Tracking
        ↓
Agent/User Conversation + Tool Calls
        ↓
Frozen Evaluator
        ↓
Score + Failure Logs
        ↓
Iterator Agent
        ↓
Proposed Harness Change
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

## 10. MVP Experimental Arms

### 10.1 Arm A: Static Harness Baseline

A fixed closed-weight model is used with the default or minimally modified τ-bench tool-calling harness.

The model, task set, evaluator, user simulator, domain database, and scoring logic remain fixed.

Purpose:

Establish baseline performance for the default task-performing agent.

### 10.2 Arm B: Iterator-Optimized Harness

The same closed-weight model is used.

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

The iterator agent may not edit:

- Benchmark tasks
- Evaluator
- User simulator
- Ground truth
- Domain database
- Scoring logic

The comparison between Arm A and Arm B isolates the value of autonomous harness optimization.

## 11. Future Experimental Arms

### 11.1 Arm C: Naive Open-Weight Model + Static Harness

A naive open-weight model is used inside a static harness.

Purpose:

Establish the baseline performance of an open-weight model before task specialization.

Possible models:

- Qwen
- Llama
- Mistral
- DeepSeek

This arm is required if the experiment later wants to compare harness optimization against model specialization.

### 11.2 Arm D: Specialized Open-Weight Model + Static Harness

An open-weight model is fine-tuned, distilled, or adapted for the τ-bench-family task.

The harness remains static.

Purpose:

Test whether model specialization alone improves task performance.

Possible specialization methods:

- Fine-tuning on successful task traces
- Distillation from high-performing closed model trajectories
- Synthetic data generation
- Adapter training
- Supervised learning on conversation/tool-use traces

### 11.3 Arm E: Specialized Open-Weight Model + Iterator-Optimized Harness

A specialized open-weight model is placed inside an iterator-optimized harness.

Purpose:

Test whether model specialization and harness optimization compound.

This is the most advanced arm and should come only after the MVP loop works.

## 12. Metrics

### 12.1 Primary Metric

The primary metric is task success rate / pass rate on τ-bench-family tasks.

The same primary metric must be used across all experimental arms.

### 12.2 Secondary Metrics

Secondary metrics:

- Policy compliance
- Tool-call correctness
- Cost per task
- Cost per successful task
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

### 12.3 Metrics Not to Use

Do not use `val_bpb` unless the experiment is literally training a language model on next-token prediction.

`val_bpb` is appropriate for Karpathy's autoresearch because that experiment optimizes a tiny GPT training loop.

For this experiment, task-level metrics are required.

## 13. Dataset Splits

The benchmark should be split into four evaluation groups.

### 13.1 Train Set

The iterator agent may optimize against this set.

The train set is used to discover candidate harness improvements.

### 13.2 Validation Set

The validation set is used to decide whether a candidate harness change should be kept.

A change should not be accepted solely because it improves the train set.

### 13.3 Hidden Test Set

The hidden test set is used only once at the end.

The iterator agent must not see hidden test results during development.

### 13.4 OOD Transfer Set

An out-of-distribution transfer set should be used to measure whether harness improvements generalize.

For τ-bench-family tasks, this can mean:

- Optimize on retail and test on airline
- Optimize on retail and test on banking
- Optimize on one version of the task set and test on perturbed tasks
- Optimize on original τ-bench and test on τ³-bench, if appropriate

## 14. Overfitting Controls

The experiment must reduce overfitting through the following controls:

1. The iterator agent cannot edit the evaluator.
2. The iterator agent cannot edit the benchmark dataset.
3. The iterator agent cannot edit ground truth labels.
4. The iterator agent cannot edit the user simulator.
5. The iterator agent cannot edit the domain database.
6. The iterator agent cannot see hidden test results.
7. The iterator agent must optimize on train and be selected on validation.
8. The final test must be run once.
9. The optimization budget must be fixed before the experiment starts.
10. Each experimental arm must receive the same budget.
11. Improvements must be checked across multiple task subsets.
12. Results must be reported by task category.
13. A complexity penalty should be tracked.
14. A cost penalty should be tracked.
15. A latency penalty should be tracked.
16. The final system should be tested on an OOD benchmark, transfer domain, or perturbed task set.

## 15. Public Benchmark Contamination Concern

A public benchmark may already be partially known to closed-weight models.

This creates a contamination risk.

The experiment reduces this risk by choosing τ-bench-family tasks because task success depends heavily on interactive tool use, policy compliance, and stateful multi-turn behavior rather than static factual knowledge.

However, this does not fully eliminate contamination risk.

Additional protections:

1. Keep the base closed-weight model fixed across Arm A and Arm B.
2. Compare only static harness vs iterator-optimized harness when testing the MVP.
3. Use perturbation sets that change names, IDs, inventory values, dates, order details, and policy wording while preserving task structure.
4. Evaluate transfer from one domain to another.
5. Run hidden test only once.
6. Report results conservatively.

The claim should not be:

"The model became better."

The claim should be:

"With the same fixed model and same benchmark family, autonomous harness optimization improved task-agent performance under a controlled evaluation protocol."

## 16. Editable and Forbidden Surfaces

### 16.1 Allowed Editable Files

The iterator agent may edit:

```text
target_agent/prompts/system_prompt.md
target_agent/prompts/task_prompt.md
target_agent/prompts/policy_summary.md
target_agent/prompts/few_shot_examples.md
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
```

### 16.2 Forbidden Files

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
```

If τ-bench-family file names differ in the actual repository, these should be mapped to the equivalent files.

## 17. Change Acceptance Rule

Each iteration must make exactly one meaningful change.

A change is accepted only if:

1. It improves validation performance, or
2. It improves the primary metric without causing unacceptable regression in secondary metrics.

A change is rejected if:

1. Train improves but validation regresses.
2. Validation improves only through higher cost with no meaningful quality gain.
3. The change increases invalid output rate.
4. The change breaks the harness.
5. The change touches forbidden files.
6. The change is too broad to attribute causality.
7. The change creates benchmark-specific hacks.
8. The change depends on memorized task IDs or task-specific shortcuts.

## 18. Iteration Protocol

Each iteration follows this loop:

1. Run current harness on train subset.
2. Run current harness on validation subset.
3. Store score, cost, latency, turn count, tool-call count, and failure logs.
4. Iterator agent reads only allowed logs and allowed source files.
5. Iterator agent proposes one harness change.
6. System checks whether the change touches only allowed files.
7. If the change is invalid, reject immediately.
8. If valid, apply the change.
9. Rerun train and validation evaluation.
10. Compare against previous best harness.
11. Keep or revert the change.
12. Write an experiment note.
13. Commit accepted changes.
14. Log rejected changes.
15. Continue until budget is exhausted.

## 19. Experiment Logging

Each iteration should produce a log entry with:

```text
iteration_id
timestamp
benchmark_name
domain
task_subset
base_model
harness_version
changed_file
change_summary
reason_for_change
train_score_before
train_score_after
validation_score_before
validation_score_after
policy_compliance_before
policy_compliance_after
tool_call_correctness_before
tool_call_correctness_after
cost_before
cost_after
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

## 20. Repository Structure

A possible repository structure:

```text
task-harness-autoresearch/
  README.md
  experiment.md

  benchmark/
    tau_bench_adapter/
    splits/
      train.jsonl
      validation.jsonl
      hidden_test.jsonl
      transfer.jsonl

  target_agent/
    agent.py
    harness.py
    prompts/
      system_prompt.md
      task_prompt.md
      policy_summary.md
      few_shot_examples.md
    policies/
      tool_selection.py
      memory_policy.py
      state_tracking.py
      clarification_policy.py
      verification_policy.py
      retry_policy.py
      error_recovery.py
    output_format.py

  iterator_agent/
    researcher.py
    experiment_policy.md
    allowed_edits.yaml
    run_iteration.py

  experiments/
    logs/
    accepted_changes.md
    rejected_changes.md
    results.csv
    graphs/

  scripts/
    run_baseline.py
    run_validation.py
    run_hidden_test.py
    plot_results.py
```

## 21. Success Criteria

### 21.1 MVP Success Criteria

The MVP is successful if:

1. Arm B outperforms Arm A on validation.
2. Arm B does not regress heavily on hidden test.
3. Improvements are not explained purely by increased cost, latency, turns, or tool calls.
4. The iterator agent discovers at least one reusable harness improvement.
5. The final report identifies which harness changes helped and which failed.
6. The system produces interpretable experiment logs.

### 21.2 Broader Success Criteria

The broader experiment is successful if:

1. Harness optimization improves task performance across multiple task subsets.
2. Improvements transfer to a held-out or OOD task set.
3. Improvements transfer across τ-bench-family domains.
4. Hybrid optimization eventually outperforms either model specialization or harness optimization alone.
5. The method reveals generalizable findings about agent behavior.

## 22. Expected Graphs

### 22.1 Performance Over Iterations

X-axis:

Iteration number.

Y-axis:

Task success rate.

Lines:

- Static baseline
- Iterator-optimized harness
- Specialized open-weight model
- Hybrid specialized model + optimized harness

Expected pattern:

The static baseline stays flat. The iterator-optimized harness improves unevenly over time. The specialized model may start higher or lower depending on task fit. The hybrid may eventually perform best.

### 22.2 Cost vs Performance

X-axis:

Cost per task.

Y-axis:

Task success rate.

Purpose:

Identify whether improvements are genuine or just caused by spending more tokens, more turns, or more tool calls.

### 22.3 Validation vs Hidden Test Performance

X-axis:

Validation score.

Y-axis:

Hidden test score.

Purpose:

Detect overfitting.

If validation improves but hidden test does not, the harness is likely overfitting to the validation distribution.

### 22.4 Domain Transfer Performance

X-axis:

Domain.

Y-axis:

Task success rate.

Purpose:

Determine whether the harness improvements transfer across retail, airline, banking, or other τ-bench-family domains.

## 23. Final Report Format

The final report should include:

1. Research question
2. Hypothesis
3. Benchmark used
4. Domain used
5. Dataset split
6. Base model
7. Agent harness design
8. Iterator agent design
9. Allowed edit surface
10. Experimental arms
11. Metrics
12. Overfitting controls
13. Contamination controls
14. Results table
15. Performance graphs
16. Accepted changes
17. Rejected changes
18. Failure analysis
19. Generalization test
20. Limitations
21. Future work

## 24. Key Risks

### 24.1 Benchmark Overfitting

The iterator agent may discover benchmark-specific tricks.

Mitigation:

Use train / validation / hidden test split, domain transfer, and perturbation evaluation.

### 24.2 Eval Exploitation

The iterator agent may optimize for the scorer rather than real task success.

Mitigation:

Freeze evaluator, inspect failure cases, use external benchmark scoring, and report qualitative examples.

### 24.3 Public Benchmark Contamination

The closed-weight model may already know aspects of the benchmark.

Mitigation:

Keep the closed-weight model fixed across baseline and optimized harness arms, use task perturbations, and test domain transfer.

### 24.4 Cost Inflation

The iterator agent may improve performance by using more tokens, more turns, more retries, or more tool calls.

Mitigation:

Track cost per successful task and impose a cost budget.

### 24.5 Loss of Interpretability

The harness may become complex and hard to understand.

Mitigation:

Allow only one meaningful change per iteration and require written experiment notes.

### 24.6 Non-Generalization

The optimized harness may work only on one benchmark or one domain.

Mitigation:

Run final transfer tests on a second τ-bench-family domain and perturbed tasks.

## 25. Future Extensions

After the MVP works, extend the system to test:

1. Naive open-weight model vs closed-weight model
2. Specialized open-weight model vs optimized harness
3. Distilled traces from successful harness runs
4. Fine-tuning on accepted task trajectories
5. Multi-agent iterator systems
6. Multiple τ-bench-family domains
7. Automatic failure taxonomy generation
8. Automatic benchmark subset selection
9. Model routing policies
10. Cost-aware harness optimization
11. Human-in-the-loop acceptance review
12. SWE-bench transfer for coding-agent evaluation
13. WebArena transfer for browser-agent evaluation

## 26. Final Claim Standard

The final claim should be conservative.

Acceptable claim:

"Using a fixed base model and frozen τ-bench-family evaluation environment, an autonomous iterator agent improved task-agent performance by modifying only the agent harness under a controlled optimization budget."

Stronger claim only if supported:

"The discovered harness improvements transferred across held-out tasks and at least one additional τ-bench-family domain."

Do not claim:

"The model became smarter."

Do not claim:

"The agent is generally better across all tasks."

Do not claim:

"This proves autonomous AI research."
