"""$0 unit tests for scripts/amortized_cost.py — synthetic costs only, no CSV,
no network, no matplotlib rendering.

Run:  python -m pytest tests/test_amortized_cost.py -q
"""
from scripts.amortized_cost import (
    Cell, breakeven_n, load_cells, compute_deltas,
)


def _row(model, split, num_tasks, num_passed, total_cost, cost_succ="x"):
    return {
        "agent_model": model, "split": split, "harness_version": "v0.1",
        "num_tasks": str(num_tasks), "num_passed": str(num_passed),
        "total_cost_usd": str(total_cost),
        "cost_per_successful_task": cost_succ,
    }


def test_breakeven_is_t_over_delta():
    assert breakeven_n(10.0, 0.02) == 500.0
    assert breakeven_n(50.0, 0.25) == 200.0
    assert breakeven_n(1.0, 0.5) == 2.0


def test_breakeven_zero_train_cost_is_immediate():
    # T_train=0 with a positive saving -> break-even at zero tasks.
    assert breakeven_n(0.0, 0.03) == 0.0


def test_breakeven_none_when_delta_not_positive():
    # tuned not cheaper -> never breaks even (delta <= 0 flagged as None).
    assert breakeven_n(10.0, 0.0) is None
    assert breakeven_n(10.0, -0.05) is None


def test_cell_task_weighted_metrics():
    c = Cell()
    c.add(total_cost=2.0, num_tasks=10, num_passed=5)
    c.add(total_cost=4.0, num_tasks=10, num_passed=7)
    assert c.per_task_cost == 6.0 / 20
    assert c.pass_rate == 12 / 20
    assert c.per_success_cost == 6.0 / 12


def test_delta_computed_correctly():
    # base $1.00/task, tuned $0.40/task over one domain -> delta = $0.60/task.
    rows = [
        _row("openai/thinkingmachines/Inkling-Small", "eval_airline", 10, 5, 10.0),
        _row("openai/armd-inkling-small-tuned", "eval_airline", 10, 5, 4.0),
    ]
    deltas = compute_deltas(load_cells(rows))
    airline = next(d for d in deltas if d.domain == "airline")
    assert abs(airline.base_per_task - 1.0) < 1e-9
    assert abs(airline.tuned_per_task - 0.4) < 1e-9
    assert abs(airline.delta_per_task - 0.6) < 1e-9
    # N*(T=$30) = 30 / 0.6 = 50 tasks
    assert breakeven_n(30.0, airline.delta_per_task) == 50.0


def test_negative_delta_domain_flagged_never():
    # tuned MORE expensive than base -> delta<=0 -> never breaks even.
    rows = [
        _row("openai/thinkingmachines/Inkling-Small", "validation", 35, 20, 1.0),
        _row("openai/armd-inkling-small-tuned", "validation", 35, 20, 2.0),
    ]
    deltas = compute_deltas(load_cells(rows))
    retail = next(d for d in deltas if d.domain == "retail")
    assert retail.delta_per_task < 0
    assert breakeven_n(10.0, retail.delta_per_task) is None


def test_per_successful_task_delta_uses_pass_rate():
    # base: $1.00/task @ 0.50 pass -> $2.00/succ
    # tuned: $0.50/task @ 0.25 pass -> $2.00/succ  => delta_per_success == 0
    # even though per-task delta is +0.50 (favorable), per-success is a wash.
    rows = [
        _row("openai/thinkingmachines/Inkling-Small", "eval_telecom", 100, 50, 100.0),
        _row("openai/armd-inkling-small-tuned", "eval_telecom", 100, 25, 50.0),
    ]
    deltas = compute_deltas(load_cells(rows))
    telecom = next(d for d in deltas if d.domain == "telecom")
    assert abs(telecom.delta_per_task - 0.5) < 1e-9
    assert abs(telecom.delta_per_success) < 1e-9
    assert breakeven_n(10.0, telecom.delta_per_success) is None


def test_stale_rows_without_cost_per_successful_task_dropped():
    rows = [
        _row("openai/thinkingmachines/Inkling-Small", "transfer_banking", 30, 0, 1.5, cost_succ=""),
        _row("openai/thinkingmachines/Inkling-Small", "transfer_banking", 30, 4, 17.0),
        _row("openai/armd-inkling-small-tuned", "transfer_banking", 30, 3, 7.0),
    ]
    cells = load_cells(rows)
    base = cells[("base", "banking")]
    assert base.num_tasks == 30  # stale empty-cost row excluded
    assert abs(base.per_task_cost - 17.0 / 30) < 1e-9


def test_aggregate_is_task_weighted_across_domains():
    rows = [
        _row("openai/thinkingmachines/Inkling-Small", "eval_airline", 10, 5, 10.0),
        _row("openai/armd-inkling-small-tuned", "eval_airline", 10, 5, 4.0),
        _row("openai/thinkingmachines/Inkling-Small", "eval_telecom", 20, 10, 20.0),
        _row("openai/armd-inkling-small-tuned", "eval_telecom", 20, 10, 6.0),
    ]
    deltas = compute_deltas(load_cells(rows))
    agg = next(d for d in deltas if d.domain == "aggregate")
    # base: (10+20)/(10+20) = 1.0/task ; tuned: (4+6)/30 = 0.3333/task
    assert abs(agg.base_per_task - 1.0) < 1e-9
    assert abs(agg.tuned_per_task - 10.0 / 30) < 1e-9
