"""TaskEvolve target agent — Arm A baseline.

Subclasses TAU2's LLMAgent and overrides the system prompt with a
Jinja2 template so the iterator can modify it programmatically.

The iterator (Milestone 2+) may modify:
    target_agent/prompts/system_prompt.j2
    target_agent/prompts/policy_summary.j2
    target_agent/prompts/few_shot_examples.j2
    target_agent/harness.py
    target_agent/model_routing.py

Do NOT modify this file (agent.py) to change agent behaviour — use
harness.py or model_routing.py instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import jinja2

from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import AssistantMessage, MultiToolMessage
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

from target_agent.harness import build_messages, filter_tools
from target_agent.model_routing import get_model

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_PROMPTS_DIR)),
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _render(template_name: str, **ctx: object) -> str:
    return _JINJA_ENV.get_template(template_name).render(**ctx).strip()


class TaskEvolveAgent(LLMAgent):
    """Half-duplex LLM agent with a Jinja2-rendered system prompt.

    Wires TAU2's generate() through harness.py and model_routing.py so
    the iterator can modify those surfaces without touching this class.
    """

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ) -> None:
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )

    @property
    def system_prompt(self) -> str:
        policy = _render("policy_summary.j2", domain_policy=self.domain_policy)
        example_conversations = _render("few_shot_examples.j2")
        return _render(
            "system_prompt.j2",
            domain_policy=policy,
            example_conversations=example_conversations,
        )

    def _generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: LLMAgentState,
    ) -> AssistantMessage:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = build_messages(state.system_messages, state.messages)
        tools = filter_tools(self.tools)
        # Pass the conversation so far (read-only) so a routing policy in
        # model_routing.get_model can branch on context (Exp 2 cost-aware routing).
        # The identity default ignores it, so this is behavior-preserving for
        # every non-routing arm.
        model = get_model(self.llm, state.messages)

        result = generate(
            model=model,
            tools=tools,
            messages=messages,
            call_name="agent_response",
            **self.llm_args,
        )
        if not isinstance(result, AssistantMessage):
            raise RuntimeError(f"generate() returned unexpected type: {type(result)}")
        return result
