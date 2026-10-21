"""Serve an OpenAI Chat Completions request with a CLI backend.

Everything else (the HTTP server, the Anthropic and Responses APIs, the
in-process clients and the framework integrations) is built on this.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from . import runner
from .backends import Final, RunSpec, TextDelta, ThinkingDelta, ToolActivity
from .config import BridgeConfig, resolve_model
from .errors import BridgeError
from .prompting import (
    build_prompt,
    extract_json,
    format_instructions,
    function_tools,
    parse_tool_reply,
    tool_instructions,
    tool_schema,
)

log = logging.getLogger("clibridge")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON-encoded


@dataclass
class Thinking:
    """A streamed piece of the model's thinking."""

    text: str


@dataclass
class ChatResult:
    content: Optional[str]
    tool_calls: List[ToolCall]
    finish_reason: str  # "stop", "tool_calls" or "length"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning: str = ""  # the model's thinking, when the CLI shows it
    reasoning_tokens: int = 0
    streamed: bool = False  # content was already yielded as str deltas
    reasoning_streamed: bool = False  # reasoning was already yielded as Thinking items
    meta: Dict[str, Any] = field(default_factory=dict)


ChatItem = Union[str, Thinking, ChatResult]


def new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:24]


class ChatJob:
    """Plans the CLI run for one request and turns its final output into a ChatResult."""

    def __init__(self, request: Dict[str, Any], config: BridgeConfig):
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            raise BridgeError("`messages` must be a non-empty list", status=400)
        route = resolve_model(request.get("model"), config.default_backend)
        self.backend = route.backend

        tools = function_tools(request)
        choice = request.get("tool_choice", request.get("function_call")) or "auto"
        if choice == "none":
            tools = []
        parallel = request.get("parallel_tool_calls") is not False
        response_format = request.get("response_format")
        format_type = response_format.get("type") if isinstance(response_format, dict) else None

        prompt, system, images = build_prompt(messages, tool_mode=bool(tools))
        system_parts = [system] if system else []
        schema = None
        if tools:
            self.mode = "tools"
            codex = self.backend == "codex"
            system_parts.append(tool_instructions(tools, choice, parallel, args_as_string=codex))
            schema = tool_schema([t["name"] for t in tools], choice, parallel, args_as_string=codex, strict=codex)
        elif format_type == "json_schema":
            self.mode = "json"
            spec = response_format.get("json_schema") or {}
            user_schema = spec.get("schema")
            system_parts.append(format_instructions(user_schema))
            # Codex enforces schemas in OpenAI's strict mode, so only hand it strict ones.
            if user_schema and (self.backend == "claude" or spec.get("strict")):
                schema = user_schema
        elif format_type == "json_object":
            self.mode = "json"
            system_parts.append(format_instructions(None))
            if self.backend == "claude":
                schema = {"type": "object"}
        else:
            self.mode = "text"

        # Parameters the CLIs have no equivalent for (max_tokens, temperature, stop, ...) are ignored.
        self.deltas = bool(request.get("stream")) and self.mode == "text"  # yield text as it arrives
        self.final: Optional[Final] = None
        self.streamed = self.thought = False
        max_turns = request.get("max_turns")
        self.spec = RunSpec(
            backend=self.backend,
            prompt=prompt,
            system="\n\n".join(system_parts) or None,
            model=route.model,
            images=images,
            json_schema=schema,
            agent=bool(request.get("agent", route.agent)),
            cwd=request.get("cwd"),
            effort=request.get("reasoning_effort") or (request.get("reasoning") or {}).get("effort"),
            max_turns=int(max_turns) if max_turns else None,
            stream=self.deltas,
        )

    def handle(self, event: Any) -> List[ChatItem]:
        """Process one CLI event; returns the items to yield to the caller."""
        if isinstance(event, TextDelta):
            if self.deltas:
                self.streamed = True
                return [event.text]
        elif isinstance(event, ThinkingDelta):
            if self.deltas:
                self.thought = True
                return [Thinking(event.text)]
        elif isinstance(event, ToolActivity):
            log.info("%s ▸ %s %s", self.backend, event.name, event.detail)
        elif isinstance(event, Final):
            self.final = event
        return []

    def finish(self) -> List[ChatItem]:
        """The items to yield once the CLI is done."""
        result = self.result()
        result.reasoning_streamed = self.thought
        items: List[ChatItem] = []
        if self.deltas and not self.streamed and result.content:
            result.streamed = True
            items.append(result.content)
        items.append(result)
        return items

    def result(self) -> ChatResult:
        final = self.final
        if final is None:
            raise BridgeError(f"{self.backend} produced no result")
        content: Optional[str] = final.text
        calls: List[ToolCall] = []
        if self.mode == "tools":
            content, parsed = parse_tool_reply(final.structured, final.text)
            calls = [ToolCall(new_id("call_"), name, args) for name, args in parsed]
            if calls and not content:
                content = None
        elif self.mode == "json":
            data = final.structured if final.structured is not None else extract_json(final.text)
            if data is not None:
                content = json.dumps(data, ensure_ascii=False)

        if calls:
            finish = "tool_calls"
        elif final.truncated:
            finish = "length"
        else:
            finish = "stop"
        meta = {"backend": self.backend, "session_id": final.session_id, "num_turns": final.num_turns}
        if final.cost_usd is not None:
            meta["cost_usd"] = final.cost_usd
        return ChatResult(
            content=content,
            tool_calls=calls,
            finish_reason=finish,
            prompt_tokens=final.input_tokens,
            completion_tokens=final.output_tokens,
            cached_tokens=final.cached_tokens,
            reasoning=final.reasoning,
            reasoning_tokens=final.reasoning_tokens,
            streamed=self.streamed,
            meta=meta,
        )


def iter_chat(request: Dict[str, Any], config: Optional[BridgeConfig] = None) -> Iterator[ChatItem]:
    """Yield text deltas (str) and Thinking items when ``request["stream"]`` is set, then one ChatResult."""
    config = config or BridgeConfig()
    job = ChatJob(request, config)
    events = runner.run(job.spec, config)
    try:
        for event in events:
            yield from job.handle(event)
    finally:
        events.close()  # stops the CLI if the caller stops reading
    yield from job.finish()


async def aiter_chat(request: Dict[str, Any], config: Optional[BridgeConfig] = None) -> AsyncIterator[ChatItem]:
    """Async version of :func:`iter_chat`."""
    config = config or BridgeConfig()
    job = ChatJob(request, config)
    events = runner.arun(job.spec, config)
    try:
        async for event in events:
            for item in job.handle(event):
                yield item
    finally:
        await events.aclose()
    for item in job.finish():
        yield item


def complete_chat(request: Dict[str, Any], config: Optional[BridgeConfig] = None) -> ChatResult:
    result = None
    for item in iter_chat({**request, "stream": False}, config):
        result = item
    assert isinstance(result, ChatResult)
    return result


async def acomplete_chat(request: Dict[str, Any], config: Optional[BridgeConfig] = None) -> ChatResult:
    result = None
    async for item in aiter_chat({**request, "stream": False}, config):
        result = item
    assert isinstance(result, ChatResult)
    return result
