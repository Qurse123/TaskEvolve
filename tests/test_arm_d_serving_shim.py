"""$0 tests for arm_d.serving_shim's pure request-handling core (`handle_chat`).

The sampler and renderer are both fakes, so this file never imports the real
`tinker` / `tinker_cookbook` packages, never calls Tinker's SamplingClient, and
never hits Together/HF. `arm_d.serving_shim` itself imports the real `tinker`
package at module level (needed to build a real `tinker.SamplingParams` inside
`handle_chat` — that's a plain object construction, $0, no network), but this
test module only exercises that through the injected fakes below.
"""
from __future__ import annotations

from arm_d.serving_shim import BASE_MODEL_ID, TUNED_MODEL_ID, handle_chat


class _FakeModelInput:
    """Stub standing in for tinker.ModelInput: only `.length` is read."""

    def __init__(self, length: int):
        self.length = length


class _FakeFunctionBody:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    """Stub standing in for tinker_cookbook.renderers.base.ToolCall."""

    def __init__(self, name: str, arguments: str, call_id: str = "call_1"):
        self.id = call_id
        self.function = _FakeFunctionBody(name, arguments)


class _FakeSequence:
    def __init__(self, tokens):
        self.tokens = tokens


class _FakeSampleResult:
    def __init__(self, tokens):
        self.sequences = [_FakeSequence(tokens)]


class _FakeFuture:
    def __init__(self, tokens):
        self._tokens = tokens

    def result(self, timeout=None):
        return _FakeSampleResult(self._tokens)


class _FakeSampler:
    """Stub standing in for tinker.SamplingClient. Records every sample() call."""

    def __init__(self, tokens):
        self._tokens = tokens
        self.calls = []

    def sample(self, prompt, num_samples, sampling_params):
        self.calls.append(
            {"prompt": prompt, "num_samples": num_samples, "sampling_params": sampling_params}
        )
        return _FakeFuture(self._tokens)


class _FakeRenderer:
    """Stub standing in for a tinker_cookbook Renderer."""

    def __init__(self, *, prompt_length, tool_calls=None, text=None, stop=None):
        self.prompt_length = prompt_length
        self._tool_calls = tool_calls
        self._text = text
        self._stop = stop if stop is not None else ["<|end|>"]
        self.calls = {}

    def create_conversation_prefix_with_tools(self, tools, system_prompt=""):
        self.calls["prefix_tools"] = tools
        self.calls["system_prompt"] = system_prompt
        prefix = []
        if system_prompt:
            prefix.append({"role": "system", "content": system_prompt})
        return prefix

    def build_generation_prompt(self, messages, role="assistant", prefill=None):
        self.calls["prompt_messages"] = messages
        return _FakeModelInput(self.prompt_length)

    def get_stop_sequences(self):
        return self._stop

    def parse_response(self, tokens):
        self.calls["parsed_tokens"] = tokens
        if self._tool_calls is not None:
            message = {"role": "assistant", "content": "", "tool_calls": self._tool_calls}
        else:
            message = {"role": "assistant", "content": self._text or ""}
        return message, "stop_sequence"

    def to_openai_message(self, message):
        result = {"role": message["role"], "content": message.get("content", "")}
        if message.get("tool_calls"):
            result["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message["tool_calls"]
            ]
        return result


def test_handle_chat_tool_call_response_passes_through_with_correct_usage():
    tokens = [1, 2, 3, 4]
    tool_calls = [_FakeToolCall("get_weather", '{"city": "SF"}')]
    renderer = _FakeRenderer(prompt_length=37, tool_calls=tool_calls)
    sampler = _FakeSampler(tokens)

    body = {
        "model": TUNED_MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a helpful agent."},
            {"role": "user", "content": "What's the weather?"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }

    resp = handle_chat(body, sampler=sampler, renderer=renderer, model_id=TUNED_MODEL_ID)

    assert resp["object"] == "chat.completion"
    assert resp["model"] == TUNED_MODEL_ID
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert choice["message"]["tool_calls"][0]["function"]["arguments"] == '{"city": "SF"}'
    assert resp["usage"] == {
        "prompt_tokens": 37,
        "completion_tokens": 4,
        "total_tokens": 41,
    }

    # system prompt + tool schema were split out and forwarded correctly
    assert renderer.calls["system_prompt"] == "You are a helpful agent."
    assert renderer.calls["prefix_tools"] == [
        {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    # the sampler received a real tinker.SamplingParams built from defaults
    sp = sampler.calls[0]["sampling_params"]
    assert sp.max_tokens == 1024
    assert sp.temperature == 0.0
    assert list(sp.stop) == ["<|end|>"]
    assert sampler.calls[0]["num_samples"] == 1


def test_handle_chat_text_only_response_has_stop_finish_reason():
    tokens = [5, 6]
    renderer = _FakeRenderer(prompt_length=12, text="Sure, here you go.")
    sampler = _FakeSampler(tokens)

    body = {
        "model": BASE_MODEL_ID,
        "messages": [{"role": "user", "content": "Hi"}],
    }

    resp = handle_chat(body, sampler=sampler, renderer=renderer, model_id=BASE_MODEL_ID)

    choice = resp["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "Sure, here you go."
    assert "tool_calls" not in choice["message"]
    assert resp["usage"] == {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14}

    # no tools on this request -> empty tool spec list, no system prompt
    assert renderer.calls["prefix_tools"] == []
    assert renderer.calls["system_prompt"] == ""


def test_handle_chat_forwards_tool_role_message_with_call_id_and_name():
    renderer = _FakeRenderer(prompt_length=20, text="ok")
    sampler = _FakeSampler([9])
    body = {
        "model": BASE_MODEL_ID,
        "max_tokens": 256,
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": "call it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "72F"},
        ],
    }

    handle_chat(body, sampler=sampler, renderer=renderer, model_id=BASE_MODEL_ID)

    prompt_messages = renderer.calls["prompt_messages"]
    tool_msg = next(m for m in prompt_messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["name"] == "get_weather"
    assert tool_msg["content"] == "72F"

    assistant_msg = next(m for m in prompt_messages if m["role"] == "assistant")
    assert assistant_msg["tool_calls"][0].function.name == "get_weather"
    assert assistant_msg["tool_calls"][0].function.arguments == "{}"

    # request-level max_tokens/temperature overrides flowed into SamplingParams
    sp = sampler.calls[0]["sampling_params"]
    assert sp.max_tokens == 256
    assert sp.temperature == 0.7


def test_handle_chat_treats_missing_content_as_empty_string():
    renderer = _FakeRenderer(prompt_length=5, text="")
    sampler = _FakeSampler([1])
    body = {
        "model": BASE_MODEL_ID,
        "messages": [{"role": "user", "content": None}],
    }

    handle_chat(body, sampler=sampler, renderer=renderer, model_id=BASE_MODEL_ID)

    prompt_messages = renderer.calls["prompt_messages"]
    assert prompt_messages[0]["content"] == ""
