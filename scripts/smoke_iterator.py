"""Iterator-loop smoke — dry-run the full Arm B cycle at $0 (CLAUDE.md M2 "Runs & verification").

This is the zero-cost wiring gate before any real proxy spend. It drives the
**real** orchestrator + driver (``run_iterator`` -> ``run_iteration``) end to end —
``feedback -> propose -> guard -> apply -> proxy x2 -> accept/revert -> note ->
commit`` — with every external call faked:

  * ``complete``   — returns a fixed proposal JSON (no LLM call),
  * ``eval_seed``  — returns deterministic per-seed metrics (no TAU2 run),
  * ``build_feedback`` reads a seeded fake proxy run dir (no real logs),
  * all writes go to a throwaway temp tree (no real ``experiments/`` mutation).

It asserts the three decision paths the loop must get right:

  1. **accept** — both proxy runs improve -> version bumps, changelog + record written,
     commit hook fires, and the driver promotes the candidate to the new current-best;
  2. **reject (hill-climb)** — the next iteration cannot beat that new best -> reverted,
     version held, edited file restored;
  3. **forbidden target** — a proposal aimed at a frozen path is rejected *before* any
     eval (the eval fn is wired to raise if ever called).

A green run proves the iterator's control flow is sound before we spend the proxy budget.

Run from the repo root:
    python -m scripts.smoke_iterator
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from iterator_agent.acceptance import RunMetrics
from iterator_agent.baseline import Distribution
from iterator_agent.edit_guard import load_policy
from iterator_agent.feedback import build_feedback
from iterator_agent.hypothesis import Ticket
from iterator_agent.run_iteration import run_iteration
from results.logger import DEFAULT_LOGS_ROOT
from scripts.run_iterator import run_iterator
from settings import config

logger = logging.getLogger(__name__)

# The Arm A proxy distribution the smoke pretends to start from.
INITIAL_BEST = Distribution(
    split="proxy",
    harness_version="v0.1",
    n=5,
    pass_rate_mean=0.6,
    pass_rate_std=0.05,
    cost_per_successful_task_mean=0.10,
    cost_per_successful_task_std=0.01,
    run_ids=(),
    cost_per_task_mean=0.10,
    cost_per_task_std=0.01,
)

# Per-seed fake metrics. Iteration 1 (seeds 9001/9002) beats the $0.10 baseline at
# the same success -> accept. Iteration 2 (seeds 9003/9004) cannot beat the promoted
# $0.08 best -> reject on the first run (seed B short-circuited).
SEED_START = 9001
SEED_METRICS: Dict[int, Tuple[float, float]] = {
    9001: (0.6, 0.08),
    9002: (0.6, 0.08),
    9003: (0.6, 0.085),
    9004: (0.6, 0.085),
}

# A valid in-surface edit target and a frozen one, for the accept and forbidden cases.
ALLOWED_TARGET = "target_agent/prompts/system_prompt.j2"
FORBIDDEN_TARGET = "benchmark/splits/proxy.json"

# Number of _check() assertions in run_smoke_iterator — keeps the summary line honest.
_TOTAL_CHECKS = 16


def _proposal_json(target: str) -> str:
    """A fixed editor response (full-file replacement) for ``target``."""
    return json.dumps(
        {
            "target_file": target,
            "new_content": (
                "{# smoke: trimmed system prompt to cut tokens #}\n"
                "You are a concise retail support agent.\n"
            ),
            "change_summary": "Trim the system prompt to reduce per-turn token cost.",
            "reason_for_change": (
                "Costliest tasks spend tokens re-reading a verbose prompt; a leaner "
                "prompt holds success at lower cost."
            ),
        }
    )


def _make_complete(target: str):
    """An injected ``complete`` that returns the same proposal for both editor calls."""
    payload = _proposal_json(target)

    def complete(_prompt: str) -> str:
        return payload

    return complete


def _make_backlog(history_sink: List):
    """Injected backlog provider: one ticket pinned to ALLOWED_TARGET, records history.

    Keeps the smoke at $0 (no real generate_tickets LLM call) and lets us assert the
    editor's cross-iteration memory reaches the backlog generator.
    """

    def provide(current_version: str, history) -> Tuple[Tuple[Ticket, ...], float]:
        history_sink.append(list(history))
        ticket = Ticket(
            ticket_id="t1",
            hypothesis="Trim the system prompt to reduce per-turn token cost.",
            target_surface=ALLOWED_TARGET,
            rationale="verbose prompt re-read every turn",
            expected_effect="lower mean cost per task, success held",
            priority=1,
        )
        return (ticket,), 0.0

    return provide


def _make_eval_seed():
    """An injected per-seed eval returning the table above (no TAU2 run)."""

    def eval_seed(seed: int) -> RunMetrics:
        pass_rate, cost = SEED_METRICS[seed]
        return RunMetrics(
            pass_rate=pass_rate, cost_per_successful_task=cost, cost_per_task=cost
        )

    return eval_seed


def _eval_must_not_run(seed: int) -> RunMetrics:
    """Eval fn for the forbidden case — proves the guard rejects before any eval."""
    raise AssertionError(f"eval ran for a forbidden proposal (seed={seed})")


def _eval_no_improvement(seed: int) -> RunMetrics:
    """Eval fn whose cost exceeds the baseline — forces a reject (revert) for Scenario D."""
    _ = seed  # constant metrics regardless of seed
    return RunMetrics(pass_rate=0.6, cost_per_successful_task=0.2, cost_per_task=0.2)


def _seed_working_tree(repo_root: Path) -> None:
    """Seed the temp tree with every allowed file so the editor can read + revert them."""
    for rel in sorted(load_policy().allowed_paths):
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{{# original {rel} #}}\n", encoding="utf-8")


def _seed_fake_proxy_run(logs_root: Path) -> None:
    """Write a minimal proxy run dir so ``build_feedback`` has something to summarize."""
    run_dir = logs_root / "proxy_20260101_000000"
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        ("task_alpha", 1.0, True, 0.06, 18, 9, "user_stop"),
        ("task_beta", 0.0, False, 0.04, 12, 5, "max_turns"),
    ]
    for task_id, reward, passed, cost, turns, tools, reason in tasks:
        (run_dir / f"{task_id}.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "reward": reward,
                    "passed": passed,
                    "cost_usd": cost,
                    "turn_count": turns,
                    "tool_call_count": tools,
                    "termination_reason": reason,
                    "seed": 1001,
                }
            ),
            encoding="utf-8",
        )


def _check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    logger.info("  [%s] %s%s", mark, label, suffix)
    return condition


def run_smoke_iterator() -> int:
    """Run the three decision-path scenarios at $0. Returns the failed-check count."""
    saved_version = config.HARNESS_VERSION
    failures = 0
    with tempfile.TemporaryDirectory(prefix="iter_smoke_") as tmp:
        root = Path(tmp)
        repo_root = root / "repo"
        logs_root = repo_root / DEFAULT_LOGS_ROOT
        iterations_root = root / "iterations"
        experiments_dir = root / "experiments"

        _seed_working_tree(repo_root)
        _seed_fake_proxy_run(logs_root)
        target_path = repo_root / ALLOWED_TARGET

        committed: List[str] = []
        editor_prompts: List[str] = []
        backlog_histories: List = []
        base_complete = _make_complete(ALLOWED_TARGET)

        def editor_complete(prompt: str) -> str:
            editor_prompts.append(prompt)
            return base_complete(prompt)

        # --- Scenario A+B: drive the real loop for two iterations (accept, then reject) ---
        logger.info("Scenario A/B: run_iterator x2 (accept -> hill-climb -> reject)")
        results = run_iterator(
            max_iterations=2,
            seed_start=SEED_START,
            repo_root=repo_root,
            initial_best=INITIAL_BEST,
            initial_version="v0.1",
            complete=editor_complete,
            eval_seed=_make_eval_seed(),
            generate_backlog=_make_backlog(backlog_histories),
            commit=lambda r: committed.append(r.iteration_id),
            iterations_root=iterations_root,
            experiments_dir=experiments_dir,
            logs_root=logs_root,
        )

        accept, reject = results[0], results[1]
        failures += not _check(
            "iteration 1 accepted", accept.accepted, accept.decision.reason
        )
        failures += not _check(
            "iteration 1 bumped version v0.1 -> v0.2",
            accept.harness_version == "v0.2",
            accept.harness_version,
        )
        failures += not _check(
            "commit hook fired only for the accepted iteration",
            committed == ["iter_0001"],
            str(committed),
        )
        failures += not _check(
            "accepted record + changelog written",
            accept.record_path.exists() and accept.changelog_path.exists(),
        )
        failures += not _check(
            "iteration 2 rejected (cannot beat promoted best)",
            not reject.accepted,
            reject.decision.reason,
        )
        failures += not _check(
            "iteration 2 ran only seed A (short-circuit)",
            len(reject.record.proxy_seeds) == 1,
            str(reject.record.proxy_seeds),
        )
        failures += not _check(
            "version held at v0.2 after the reject",
            reject.harness_version == "v0.2",
            reject.harness_version,
        )
        failures += not _check(
            "edited file holds the accepted content (kept on accept; iter 2's revert is a no-op since it re-proposed the same edit)",
            target_path.read_text(encoding="utf-8") == accept.proposed_edit.new_content,
        )
        # Backlog memory: iteration 2's backlog generation must see iteration 1's record.
        failures += not _check(
            "backlog memory: iteration 2 backlog sees iteration 1's accepted change",
            len(backlog_histories) == 2
            and backlog_histories[0] == []
            and bool(backlog_histories[1])
            and backlog_histories[1][0].change_summary == accept.proposed_edit.change_summary,
        )
        # Ticket focus: the editor's (single) propose call carried the ticket hypothesis.
        failures += not _check(
            "editor propose prompt carried the ticket hypothesis (diagnose skipped)",
            any("Trim the system prompt" in p for p in editor_prompts),
        )
        # Change tracking: the exact edit diff is captured for the accepted iteration.
        accept_diff = iterations_root / accept.iteration_id / "change.diff"
        failures += not _check(
            "change.diff captured the accepted edit",
            accept_diff.exists()
            and "concise retail support agent" in accept_diff.read_text(encoding="utf-8"),
        )
        # Model provenance: the iteration record names the editor + agent models.
        accept_record = json.loads(accept.record_path.read_text(encoding="utf-8"))
        failures += not _check(
            "iteration record captures editor_model + agent_model",
            "editor_model" in accept_record and "agent_model" in accept_record,
        )

        # --- Scenario C: forbidden target is rejected before any eval ---
        logger.info("Scenario C: forbidden-target proposal rejected pre-eval")
        forbidden = run_iteration(
            iteration_id="iter_forbidden",
            seeds=(8001, 8002),
            repo_root=repo_root,
            best=INITIAL_BEST,
            feedback=build_feedback("proxy", logs_root=logs_root),
            complete=_make_complete(FORBIDDEN_TARGET),
            eval_seed=_eval_must_not_run,  # raises if the guard lets it through
            current_version="v0.2",
            iterations_root=iterations_root,
            experiments_dir=experiments_dir,
            logs_root=logs_root,
        )
        failures += not _check(
            "forbidden proposal rejected",
            not forbidden.accepted,
            forbidden.decision.reason,
        )
        failures += not _check(
            "no proxy seeds spent on the forbidden proposal",
            len(forbidden.record.proxy_seeds) == 0,
            str(forbidden.record.proxy_seeds),
        )

        # --- Scenario D: a rejected edit is reverted to its prior on-disk content ---
        # Distinct target (untouched harness.py) + a failing eval, so the revert is
        # observable: the proposed content differs from disk and must be rolled back.
        logger.info("Scenario D: rejected edit reverts the file to its prior content")
        harness_path = repo_root / "target_agent/harness.py"
        harness_before = harness_path.read_text(encoding="utf-8")
        reverted = run_iteration(
            iteration_id="iter_revert",
            seeds=(7001, 7002),
            repo_root=repo_root,
            best=INITIAL_BEST,
            feedback=build_feedback("proxy", logs_root=logs_root),
            complete=_make_complete("target_agent/harness.py"),
            eval_seed=_eval_no_improvement,
            current_version="v0.2",
            iterations_root=iterations_root,
            experiments_dir=experiments_dir,
            logs_root=logs_root,
        )
        failures += not _check(
            "edit rejected (eval did not improve)",
            not reverted.accepted,
            reverted.decision.reason,
        )
        failures += not _check(
            "file rolled back to its prior on-disk content",
            harness_path.read_text(encoding="utf-8") == harness_before,
        )

    config.HARNESS_VERSION = saved_version
    logger.info(
        "Iterator smoke complete: %d/%d checks passed",
        _TOTAL_CHECKS - failures,
        _TOTAL_CHECKS,
    )
    return failures


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    failures = run_smoke_iterator()
    if failures:
        logger.error("Iterator smoke FAILED: %d check(s) did not pass.", failures)
        return 1
    logger.info("Iterator smoke PASSED — the loop is wired correctly ($0 spent).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
