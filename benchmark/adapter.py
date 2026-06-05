"""Run a single TaskEvolveAgent evaluation against frozen TAU2 components.

Flow: build TAU2 environment and user -> inject TaskEvolveAgent into the
Orchestrator -> run the simulation -> return a normalized EvalResult.

This adapter does not write logs or CSV files; observability/logger.py owns
run-level persistence, while Langfuse captures per-LLM-call telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from tau2.config import (
    DEFAULT_LLM_ARGS_AGENT,
    DEFAULT_LLM_ARGS_USER,
    DEFAULT_LLM_USER,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
)
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.runner.build import build_environment, build_user
from tau2.runner.simulation import run_simulation

from settings import config
from target_agent.agent import TaskEvolveAgent

# TAU2's standard half-duplex user simulator (its own registered component).
USER_SIMULATOR = "user_simulator"


@dataclass(frozen=True)
class EvalResult:
    """The benchmark verdict for a single task simulation.

    Deliberately minimal: only what Langfuse does NOT capture — the evaluator's
    reward, pass/fail, and cost-per-success inputs. Per-call telemetry (tokens,
    latency, tool steps, prompt/response) lives in Langfuse and is correlated by
    ``task_id`` and run id, not duplicated here.
    """

    task_id: str
    domain: str
    split: str
    agent_model: str
    reward: float
    passed: bool
    agent_cost: Optional[float]
    termination_reason: str
    seed: Optional[int]
    timestamp: str


def _require_agent_model(agent_model: Optional[str]) -> str:
    """Return the agent model, failing fast with a clear message if unset."""
    model = agent_model or config.AGENT_MODEL
    if not model:
        raise RuntimeError(
            "AGENT_MODEL is not set. Define the agent model in .env "
            "(see .env.example), e.g. AGENT_MODEL=gpt-4.1"
        )
    return model


def _load_task(domain: str, task_id: str) -> Task:
    """Return the task with ``task_id`` from ``domain``'s full task set.

    Loads every task in the domain (``None`` split = no filtering) so lookup works
    regardless of which split the ID was sampled from.
    """
    loader = registry.get_tasks_loader(domain)
    for task in loader(None):
        if task.id == task_id:
            return task
    raise ValueError(f"Task {task_id!r} not found in domain {domain!r}.")


def _normalize(
    sim: SimulationRun, *, domain: str, split: str, agent_model: str
) -> EvalResult:
    """Convert a raw ``SimulationRun`` into the minimal :class:`EvalResult`."""
    reward = sim.reward_info.reward if sim.reward_info else 0.0
    termination = getattr(sim.termination_reason, "value", sim.termination_reason)
    return EvalResult(
        task_id=sim.task_id,
        domain=domain,
        split=split,
        agent_model=agent_model,
        reward=reward,
        passed=reward >= config.PASS_THRESHOLD,
        agent_cost=sim.agent_cost,
        termination_reason=str(termination),
        seed=sim.seed,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_eval(
    task_id: str,
    *,
    split: str,
    domain: Optional[str] = None,
    seed: Optional[int] = None,
    agent_model: Optional[str] = None,
    llm_args: Optional[dict] = None,
) -> EvalResult:
    """Run one task through TAU2's frozen Orchestrator with our injected agent.

    Args:
        task_id: ID of the task to run (must exist in ``domain``).
        split: Our split label (``"smoke"``/``"proxy"``/``"validation"``) — recorded
            on the result for logging; does not affect execution.
        domain: TAU2 domain name. Defaults to ``settings.config.DEFAULT_DOMAIN``.
        seed: Random seed for this trial (the double-run rule uses two seeds).
        agent_model: Override the agent model; defaults to ``config.AGENT_MODEL``.
        llm_args: Override the agent LLM args; defaults to TAU2's agent defaults
            (``temperature=0.0``) for reproducibility.

    Returns:
        An :class:`EvalResult` with the reward, pass/fail, and cost verdict.
    """
    domain = domain or config.DEFAULT_DOMAIN
    model = _require_agent_model(agent_model)
    user_model = config.USER_MODEL or DEFAULT_LLM_USER
    max_steps = config.MAX_STEPS if config.MAX_STEPS is not None else DEFAULT_MAX_STEPS
    max_errors = (
        config.MAX_ERRORS if config.MAX_ERRORS is not None else DEFAULT_MAX_ERRORS
    )

    task = _load_task(domain, task_id)
    environment = build_environment(domain)

    # Our agent — constructed directly and injected (no registry name lookup).
    agent = TaskEvolveAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm=model,
        llm_args=dict(llm_args) if llm_args is not None else dict(DEFAULT_LLM_ARGS_AGENT),
    )

    # TAU2's frozen user simulator, built via TAU2's own helper.
    user = build_user(
        USER_SIMULATOR,
        environment,
        task,
        llm=user_model,
        llm_args=dict(DEFAULT_LLM_ARGS_USER),
    )

    orchestrator = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
    )

    sim = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
    return _normalize(sim, domain=domain, split=split, agent_model=model)
