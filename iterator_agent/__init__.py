"""Iterator agent (Milestone 2, Arm B).

A separate process that reads proxy logs, proposes one harness change per
iteration, double-runs proxy to confirm it beats the current-best distribution
within guardrails, then keeps or reverts — under a fixed optimization budget.

The iterator never sees validation or test results during optimization
(see CLAUDE.md "Hard Constraints").
"""
