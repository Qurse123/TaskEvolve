"""Subprocess payload for the candidate preflight — run INSIDE the candidate tree.

Invoked as ``python -m iterator_agent.preflight_check`` with cwd at the working
tree under test, so ``target_agent`` resolves to the *edited* files (fresh
imports, no stale modules from the parent process). Exercises every edit
surface the way the real agent uses it, at $0 (no LLM, no benchmark):

  1. Compile the editable Python surfaces (clear syntax-error messages).
  2. Import ``target_agent.harness``; call ``build_messages`` with a realistic
     long conversation (system + 30 messages incl. tool traffic) and verify it
     returns tau2 Message instances; call ``filter_tools``.
  3. Import ``target_agent.model_routing``; ``get_model`` must return an
     allowed agent model (the editor's contract: gpt-4.1 / gpt-4.1-mini).
  4. Render the prompt templates exactly as ``target_agent.agent`` does
     (StrictUndefined), so an undefined variable or Jinja error fails here.

Prints one JSON object ``{"passed": bool, "reason": str}`` on the last stdout
line. Exit code 0 whenever a verdict was produced (even a failing one).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The editor may only route the agent to these models (see prompts/*.j2).
ALLOWED_AGENT_MODELS = ("gpt-4.1", "gpt-4.1-mini")

PYTHON_SURFACES = (
    "target_agent/harness.py",
    "target_agent/model_routing.py",
)

_SAMPLE_POLICY = (
    "Agents must verify the user's identity before any account action. "
    "Refunds over $100 require a manager override. Never reveal internal notes."
)


def _check_compiles(root: Path) -> None:
    for rel in PYTHON_SURFACES:
        path = root / rel
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        try:
            compile(source, rel, "exec")
        except SyntaxError as exc:
            raise RuntimeError(f"{rel} does not compile: {exc}") from exc


def _synthetic_history():
    """A realistic ~30-message conversation, long enough to trigger any
    compression path an edited harness may add."""
    from tau2.data_model.message import (
        AssistantMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    history = []
    for i in range(12):
        history.append(UserMessage(role="user", content=f"User turn {i}: my order?"))
        if i % 3 == 0:
            call = ToolCall(id=f"call_{i}", name="get_order_details", arguments={"order_id": f"#W{i}"})
            history.append(
                AssistantMessage(role="assistant", content=None, tool_calls=[call])
            )
            history.append(
                ToolMessage(
                    id=f"call_{i}", role="tool",
                    content='{"status": "delivered"}', requestor="assistant",
                )
            )
        history.append(
            AssistantMessage(role="assistant", content=f"Assistant reply {i}.")
        )
    return history


def _check_harness() -> None:
    from tau2.data_model.message import (
        AssistantMessage,
        SystemMessage,
        ToolMessage,
        UserMessage,
    )

    import target_agent.harness as harness

    for name in ("build_messages", "filter_tools"):
        if not callable(getattr(harness, name, None)):
            raise RuntimeError(f"target_agent/harness.py must define {name}()")

    system = SystemMessage(role="system", content="You are a retail service agent.")
    try:
        messages = harness.build_messages([system], _synthetic_history())
    except Exception as exc:
        raise RuntimeError(
            f"harness build_messages crashed on a realistic conversation: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("harness build_messages must return a non-empty list")
    valid_types = (SystemMessage, UserMessage, AssistantMessage, ToolMessage)
    for i, msg in enumerate(messages):
        if not isinstance(msg, valid_types):
            raise RuntimeError(
                f"harness build_messages returned a non-Message element at index {i}: "
                f"{type(msg).__name__} — every element must be a tau2 message model"
            )

    tools = harness.filter_tools([])
    if not isinstance(tools, list):
        raise RuntimeError("harness filter_tools must return a list")


def _check_model_routing() -> None:
    import target_agent.model_routing as routing

    if not callable(getattr(routing, "get_model", None)):
        raise RuntimeError("target_agent/model_routing.py must define get_model()")
    try:
        model = routing.get_model("gpt-4.1")  # the exact call agent.py makes
    except Exception as exc:
        raise RuntimeError(
            f"model_routing get_model crashed: {type(exc).__name__}: {exc}"
        ) from exc
    if model not in ALLOWED_AGENT_MODELS:
        raise RuntimeError(
            f"model_routing returned disallowed model {model!r}; "
            f"allowed: {', '.join(ALLOWED_AGENT_MODELS)}"
        )


def _check_templates(root: Path) -> None:
    import jinja2

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(root / "target_agent" / "prompts")),
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    def render(name: str, **ctx: object) -> str:
        try:
            return env.get_template(name).render(**ctx).strip()
        except Exception as exc:
            raise RuntimeError(
                f"template {name} failed to render: {type(exc).__name__}: {exc}"
            ) from exc

    # Mirror target_agent/agent.py's system_prompt chain.
    policy = render("policy_summary.j2", domain_policy=_SAMPLE_POLICY)
    examples = render("few_shot_examples.j2")
    system_prompt = render(
        "system_prompt.j2",
        domain_policy=policy,
        example_conversations=examples,
    )
    if not system_prompt:
        raise RuntimeError("template system_prompt.j2 rendered to an empty prompt")


def main() -> int:
    root = Path.cwd()
    try:
        _check_compiles(root)
        _check_harness()
        _check_model_routing()
        _check_templates(root)
    except Exception as exc:  # noqa: BLE001 - the verdict IS the product
        print(json.dumps({"passed": False, "reason": str(exc)}))
        return 0
    print(json.dumps({"passed": True, "reason": "all preflight checks passed"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
