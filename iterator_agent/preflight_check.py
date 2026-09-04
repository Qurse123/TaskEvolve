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
     allowed agent model (the editor's contract: a single model from the
     model this study can serve and price).
  4. Render the prompt templates exactly as ``target_agent.agent`` does
     (StrictUndefined), so an undefined variable or Jinja error fails here.

Prints one JSON object ``{"passed": bool, "reason": str}`` on the last stdout
line. Exit code 0 whenever a verdict was produced (even a failing one).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# model_routing picks a single model for the WHOLE agent (no per-turn/keyword
# routing). We do not enumerate a pool here: doing so would hand the editor a menu
# of candidates and turn a discovered cost lever into a prompted one. Instead we
# check only that whatever it returns is an Anthropic Claude model LiteLLM can
# price, which is what "servable and billable" actually means for this study.
MODEL_PREFIXES = ("anthropic/claude", "claude")


def _is_servable_model(model: object, configured: str) -> bool:
    """True when *model* is a model this study can actually serve and bill.

    Identity routing always passes: every arm sets its own AGENT_MODEL, and the
    open-weight arms are not Anthropic. A deliberate swap away from the configured
    model must land on an Anthropic Claude id LiteLLM holds a price for.
    """
    if not isinstance(model, str) or not model.strip():
        return False
    if model == configured:
        return True
    if not model.startswith(MODEL_PREFIXES):
        return False
    import litellm

    bare = model.split("/", 1)[1] if "/" in model else model
    return bool(litellm.model_cost.get(model) or litellm.model_cost.get(bare))

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
    from settings import config

    configured = config.AGENT_MODEL
    # Exercise the exact call agent.py makes — get_model(configured, history) —
    # with both an empty and a realistic multi-turn history, so any code path that
    # returns an unservable model (on any input) is caught at $0.
    for history in ([], _synthetic_history()):
        try:
            model = routing.get_model(configured, history)
        except Exception as exc:
            raise RuntimeError(
                f"model_routing get_model crashed: {type(exc).__name__}: {exc}"
            ) from exc
        if not _is_servable_model(model, configured):
            raise RuntimeError(
                f"model_routing returned {model!r}, which this study cannot serve "
                "and price; a swap must target an Anthropic Claude model"
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
