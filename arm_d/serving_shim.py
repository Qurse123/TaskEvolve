
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import tinker
from tinker_cookbook.renderers.base import ToolCall

import settings.config  # noqa: F401  side effect: load .env, bridge TINKER_KEY->TINKER_API_KEY

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8100

# Routing keys — the request's `model` field selects which SamplingClient serves
# it. Both are also the pricing keys registered in settings/pricing.py.
TUNED_MODEL_ID = "armd-inkling-small-tuned"
BASE_MODEL_ID = "thinkingmachines/Inkling-Small"

DEFAULT_BASE_MODEL = "thinkingmachines/Inkling-Small"
DEFAULT_TUNED_PATH = (
    "tinker://fdc7bf81-d958-529c-9788-2332faf52c1d:train:0/sampler_weights/"
    "arm-d-inkling-small-lora-run_20260807_222817"
)

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
# Per-request sample() ceiling — a legit (even long banking) generation finishes
# well under this; exceeding it means a stuck/dead connection, so fail-fast.
SAMPLE_TIMEOUT_S = 240


# --------------------------------------------------------------------------
# Pure request-handling logic (unit-testable at $0 with fakes)
# --------------------------------------------------------------------------


def _split_system_prompt(messages: Sequence[dict]) -> Tuple[str, List[dict]]:
    """Pull every system-role message out of an OpenAI-shaped message list,
    joining their content (there's normally one), and return the remaining
    messages untouched (in order)."""
    system_parts: List[str] = []
    rest: List[dict] = []
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content") or ""
            if isinstance(content, list):  # OpenAI content-parts form
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            system_parts.append(content)
        else:
            rest.append(message)
    return "\n".join(system_parts), rest


def _to_tool_specs(tools: Optional[Sequence[dict]]) -> List[dict]:
    """Convert OpenAI ``tools`` (``[{"type": "function", "function": {...}}]``)
    into cookbook ``ToolSpec`` dicts (``{"name", "description", "parameters"}``).
    Tolerates a bare function dict (no ``type``/``function`` wrapper) too."""
    specs: List[dict] = []
    for tool in tools or []:
        fn = tool.get("function", tool)
        specs.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return specs


def _to_cookbook_tool_calls(tool_calls: Optional[Sequence[dict]]) -> List[ToolCall]:
    """OpenAI-shaped ``tool_calls`` (already the client's wire format) ->
    cookbook ``ToolCall`` objects the renderer expects on an assistant
    ``Message``."""
    out: List[ToolCall] = []
    for call in tool_calls or []:
        fn = call.get("function", {})
        out.append(
            ToolCall(
                id=call.get("id"),
                function=ToolCall.FunctionBody(
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", ""),
                ),
            )
        )
    return out


def _to_cookbook_messages(messages: Sequence[dict]) -> List[dict]:
    """Map OpenAI-shaped non-system messages to cookbook ``Message`` dicts
    (role/content always; tool_calls/tool_call_id/name carried through when
    present)."""
    convo: List[dict] = []
    for message in messages:
        role = message["role"]
        content = message.get("content")
        if content is None:
            content = ""
        cookbook_message: Dict[str, Any] = {"role": role, "content": content}
        if message.get("tool_calls"):
            cookbook_message["tool_calls"] = _to_cookbook_tool_calls(message["tool_calls"])
        if role == "tool":
            if "tool_call_id" in message:
                cookbook_message["tool_call_id"] = message["tool_call_id"]
            if "name" in message:
                cookbook_message["name"] = message["name"]
        convo.append(cookbook_message)
    return convo


def _strip_think_content(oai_message: dict) -> None:
    """Inkling is a reasoning model; parse_response leaves its chain-of-thought
    in ``<think>...</think>`` inside the message content. Strip it in place so
    the user-simulator sees only the reply. Applied identically to tuned + base,
    so the tuned-vs-base delta is unaffected."""
    import re

    content = oai_message.get("content")
    if not isinstance(content, str):
        return
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)  # unclosed (truncated)
    oai_message["content"] = cleaned.strip()


def handle_chat(body: dict, *, sampler: Any, renderer: Any, model_id: str) -> dict:
    """Serve one ``/v1/chat/completions`` request against an already-resolved
    ``sampler``/``renderer`` pair. Pure aside from the injected ``sampler.sample``
    call — no I/O, no routing, no caching — so tests inject fakes for both and
    never touch the real ``tinker``/``tinker_cookbook`` packages.
    """
    system_prompt, rest = _split_system_prompt(body.get("messages") or [])
    convo = _to_cookbook_messages(rest)
    tool_specs = _to_tool_specs(body.get("tools"))

    prefix = renderer.create_conversation_prefix_with_tools(
        tool_specs, system_prompt=system_prompt
    )
    prompt = renderer.build_generation_prompt(prefix + convo)

    sampling_params = tinker.SamplingParams(
        max_tokens=body.get("max_tokens", DEFAULT_MAX_TOKENS),
        temperature=body.get("temperature", DEFAULT_TEMPERATURE),
        stop=renderer.get_stop_sequences(),
    )

    # Bound the wait: a dead/stuck Tinker connection otherwise blocks .result()
    # forever and wedges the whole eval at one turn. On timeout this raises ->
    # 500 -> run_train_eval's retry path (fail-fast, not hang).
    response = sampler.sample(
        prompt, num_samples=1, sampling_params=sampling_params
    ).result(timeout=SAMPLE_TIMEOUT_S)
    tokens = list(response.sequences[0].tokens)

    message, _termination = renderer.parse_response(tokens)
    oai_message = renderer.to_openai_message(message)
    _strip_think_content(oai_message)

    finish_reason = "tool_calls" if oai_message.get("tool_calls") else "stop"
    prompt_tokens = prompt.length
    completion_tokens = len(tokens)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": oai_message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# --------------------------------------------------------------------------
# Real Tinker wiring (lazy-created, cached; never touched by unit tests)
# --------------------------------------------------------------------------


def _default_renderer_factory(model_name: str):
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model_name)
    return renderers.get_renderer("tml_v0", tokenizer, model_name=model_name)


class SamplerCache:
    """Lazily creates and caches a ``SamplingClient`` per routing model id, plus
    the one shared renderer (same base tokenizer/template for tuned and base —
    a LoRA adapter doesn't change the tokenizer). Factories are injectable so
    this can be smoke-tested without a live Tinker call if ever needed."""

    def __init__(
        self,
        *,
        tuned_path: str = DEFAULT_TUNED_PATH,
        base_model: str = DEFAULT_BASE_MODEL,
        service_client_factory: Optional[Callable[[], Any]] = None,
        renderer_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self._tuned_path = tuned_path
        self._base_model = base_model
        self._service_client_factory = service_client_factory or tinker.ServiceClient
        self._renderer_factory = renderer_factory or _default_renderer_factory
        self._service_client: Optional[Any] = None
        self._renderer: Optional[Any] = None
        self._samplers: Dict[str, Any] = {}

    def _client(self) -> Any:
        if self._service_client is None:
            self._service_client = self._service_client_factory()
        return self._service_client

    def renderer(self) -> Any:
        if self._renderer is None:
            self._renderer = self._renderer_factory(self._base_model)
        return self._renderer

    def sampler_for(self, model_id: str) -> Any:
        cached = self._samplers.get(model_id)
        if cached is not None:
            return cached
        client = self._client()
        if model_id == TUNED_MODEL_ID:
     
            sampler = client.create_sampling_client(
                base_model=self._base_model, model_path=self._tuned_path
            )
        elif model_id == BASE_MODEL_ID:
            sampler = client.create_sampling_client(base_model=self._base_model)
        else:
            raise KeyError(
                f"unknown model id for routing: {model_id!r} "
                f"(expected {TUNED_MODEL_ID!r} or {BASE_MODEL_ID!r})"
            )
        self._samplers[model_id] = sampler
        return sampler


def make_handler(cache: SamplerCache) -> type:
    """Build a ``BaseHTTPRequestHandler`` subclass closing over ``cache`` (the
    stdlib handler protocol takes no constructor args of its own)."""

    class ShimHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
            if self.path.rstrip("/") == "/v1/models":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"id": TUNED_MODEL_ID, "object": "model"},
                            {"id": BASE_MODEL_ID, "object": "model"},
                        ],
                    },
                )
            else:
                self._send_json(404, {"error": f"not found: {self.path}"})

        def do_POST(self) -> None:  # noqa: N802 (stdlib method name)
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send_json(404, {"error": f"not found: {self.path}"})
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError as exc:
                self._send_json(400, {"error": f"invalid JSON body: {exc}"})
                return

            model_id = body.get("model") or BASE_MODEL_ID
            try:
                sampler = cache.sampler_for(model_id)
            except KeyError as exc:
                self._send_json(400, {"error": str(exc)})
                return

            try:
                result = handle_chat(
                    body, sampler=sampler, renderer=cache.renderer(), model_id=model_id
                )
            except Exception as exc:  # pragma: no cover - surfaces as a client error
                logger.exception("handle_chat failed for model=%s", model_id)
                self._send_json(500, {"error": str(exc)})
                return

            self._send_json(200, result)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002,N802
            logger.info("%s - %s", self.address_string(), format % args)

    return ShimHandler


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tuned-path", default=DEFAULT_TUNED_PATH)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cache = SamplerCache(tuned_path=args.tuned_path, base_model=args.base_model)
    handler_cls = make_handler(cache)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler_cls)

    base_url = f"http://localhost:{args.port}/v1"
    print(f"arm_d serving shim listening on {base_url}")
    print(f"  tuned:  model={TUNED_MODEL_ID!r} -> {args.tuned_path}")
    print(f"  base:   model={BASE_MODEL_ID!r} -> base_model={args.base_model}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
