"""OpenAI Responses API wire format (used by the OpenAI Agents SDK, Vercel AI SDK, ...)."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from .core import ChatItem, ChatResult, Thinking, new_id
from .errors import BridgeError
from .prompting import content_text

Event = Tuple[str, Dict[str, Any]]


class ResponseStore:
    """Keeps recent conversations in memory so ``previous_response_id`` works."""

    def __init__(self, size: int = 500):
        self.size = size
        self._items: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, response_id: str) -> Optional[List[Dict[str, Any]]]:
        with self._lock:
            return self._items.get(response_id)

    def put(self, response_id: str, messages: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._items[response_id] = messages
            while len(self._items) > self.size:
                self._items.popitem(last=False)


def _input_messages(items: Any) -> List[Dict[str, Any]]:
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    out: List[Dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind in (None, "message"):
            out.append({"role": item.get("role", "user"), "content": item.get("content")})
        elif kind == "function_call":
            call = {
                "id": item.get("call_id") or item.get("id"),
                "type": "function",
                "function": {"name": item.get("name"), "arguments": item.get("arguments") or "{}"},
            }
            if out and out[-1].get("role") == "assistant":
                out[-1].setdefault("tool_calls", []).append(call)
            else:
                out.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif kind == "function_call_output":
            output = item.get("output")
            text = output if isinstance(output, str) else content_text(output)
            out.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": text})
    return out


def to_chat_request(body: Dict[str, Any], store: ResponseStore) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return ``(chat request, conversation without instructions)``."""
    conversation: List[Dict[str, Any]] = []
    previous = body.get("previous_response_id")
    if previous:
        history = store.get(previous)
        if history is None:
            raise BridgeError(f"Unknown previous_response_id {previous!r}", status=400)
        conversation.extend(history)
    conversation.extend(_input_messages(body.get("input")))

    messages = list(conversation)
    if body.get("instructions"):
        messages.insert(0, {"role": "system", "content": body["instructions"]})
    request: Dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        "stream": bool(body.get("stream")),
    }
    for key in ("tools", "tool_choice", "parallel_tool_calls", "agent", "cwd", "max_turns"):
        if key in body:
            request[key] = body[key]
    effort = (body.get("reasoning") or {}).get("effort")
    if effort:
        request["reasoning_effort"] = effort
    fmt = (body.get("text") or {}).get("format") or {}
    if fmt.get("type") == "json_schema":
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": fmt.get("name"), "schema": fmt.get("schema"), "strict": fmt.get("strict")},
        }
    elif fmt.get("type") == "json_object":
        request["response_format"] = {"type": "json_object"}
    return request, conversation


def assistant_message(result: ChatResult) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in result.tool_calls
        ]
    return message


def _usage(result: ChatResult) -> Dict[str, Any]:
    return {
        "input_tokens": result.prompt_tokens,
        "input_tokens_details": {"cached_tokens": result.cached_tokens},
        "output_tokens": result.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": result.reasoning_tokens},
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }


def _text_part(text: str) -> Dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}


def _message_item(item_id: str, text: str, status: str = "completed") -> Dict[str, Any]:
    content = [_text_part(text)] if status == "completed" else []
    return {"type": "message", "id": item_id, "status": status, "role": "assistant", "content": content}


def _reasoning_item(item_id: str, text: str, status: str = "completed") -> Dict[str, Any]:
    summary = [{"type": "summary_text", "text": text}] if status == "completed" else []
    return {"type": "reasoning", "id": item_id, "summary": summary, "status": status}


def _call_item(item_id: str, call: Any, status: str = "completed") -> Dict[str, Any]:
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call.id,
        "name": call.name,
        "arguments": call.arguments if status == "completed" else "",
        "status": status,
    }


class ResponseEncoder:
    """Builds Response objects and streaming events for one request."""

    def __init__(self, body: Dict[str, Any], response_id: Optional[str] = None):
        self.body = body
        self.id = response_id or new_id("resp_")
        self.created = int(time.time())
        # Like the real API, reasoning items only appear when the request asks about reasoning.
        self.show_reasoning = body.get("reasoning") is not None
        self.seq = 0
        self.output: List[Dict[str, Any]] = []
        self.open_kind: Optional[str] = None  # "message" or "reasoning" item being streamed
        self.item_id = ""
        self.text = ""

    def envelope(self, status: str, output: List[Dict[str, Any]], result: Optional[ChatResult] = None) -> Dict[str, Any]:
        truncated = result is not None and result.finish_reason == "length"
        return {
            "id": self.id,
            "object": "response",
            "created_at": self.created,
            "status": "incomplete" if truncated else status,
            "model": self.body.get("model") or "claude-code",
            "output": output,
            "usage": _usage(result) if result else None,
            "parallel_tool_calls": self.body.get("parallel_tool_calls", True),
            "tool_choice": self.body.get("tool_choice", "auto"),
            "tools": self.body.get("tools") or [],
            "instructions": self.body.get("instructions"),
            "previous_response_id": self.body.get("previous_response_id"),
            "metadata": self.body.get("metadata") or {},
            "text": self.body.get("text") or {"format": {"type": "text"}},
            "temperature": None,
            "top_p": None,
            "max_output_tokens": None,
            "reasoning": self.body.get("reasoning"),
            "store": True,
            "truncation": "disabled",
            "error": None,
            "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
            "user": None,
            "clibridge": result.meta if result else None,
        }

    def response(self, result: ChatResult) -> Dict[str, Any]:
        output = []
        if self.show_reasoning and result.reasoning:
            output.append(_reasoning_item(new_id("rs_"), result.reasoning))
        if result.content:
            output.append(_message_item(new_id("msg_"), result.content))
        output += [_call_item(new_id("fc_"), call) for call in result.tool_calls]
        return self.envelope("completed", output, result)

    # streaming

    def _event(self, kind: str, **data: Any) -> Event:
        self.seq += 1
        return kind, {"type": kind, "sequence_number": self.seq, **data}

    def start(self) -> List[Event]:
        snapshot = self.envelope("in_progress", [])
        return [self._event("response.created", response=snapshot), self._event("response.in_progress", response=snapshot)]

    def _close(self) -> List[Event]:
        if self.open_kind is None:
            return []
        index, item_id, text = len(self.output), self.item_id, self.text
        if self.open_kind == "message":
            done = _message_item(item_id, text)
            events = [
                self._event("response.output_text.done", item_id=item_id, output_index=index, content_index=0,
                            text=text, logprobs=[]),
                self._event("response.content_part.done", item_id=item_id, output_index=index, content_index=0,
                            part=_text_part(text)),
            ]
        else:
            done = _reasoning_item(item_id, text)
            events = [
                self._event("response.reasoning_summary_text.done", item_id=item_id, output_index=index,
                            summary_index=0, text=text),
                self._event("response.reasoning_summary_part.done", item_id=item_id, output_index=index,
                            summary_index=0, part={"type": "summary_text", "text": text}),
            ]
        events.append(self._event("response.output_item.done", output_index=index, item=done))
        self.output.append(done)
        self.open_kind = None
        return events

    def _delta(self, kind: str, text: str) -> List[Event]:
        events: List[Event] = []
        index = len(self.output)
        if self.open_kind != kind:
            events += self._close()
            index = len(self.output)
            self.open_kind, self.text = kind, ""
            if kind == "message":
                self.item_id = new_id("msg_")
                events.append(self._event("response.output_item.added", output_index=index,
                                          item=_message_item(self.item_id, "", "in_progress")))
                events.append(self._event("response.content_part.added", item_id=self.item_id, output_index=index,
                                          content_index=0, part=_text_part("")))
            else:
                self.item_id = new_id("rs_")
                events.append(self._event("response.output_item.added", output_index=index,
                                          item=_reasoning_item(self.item_id, "", "in_progress")))
                events.append(self._event("response.reasoning_summary_part.added", item_id=self.item_id,
                                          output_index=index, summary_index=0,
                                          part={"type": "summary_text", "text": ""}))
        self.text += text
        if kind == "message":
            events.append(self._event("response.output_text.delta", item_id=self.item_id, output_index=index,
                                      content_index=0, delta=text, logprobs=[]))
        else:
            events.append(self._event("response.reasoning_summary_text.delta", item_id=self.item_id,
                                      output_index=index, summary_index=0, delta=text))
        return events

    def encode(self, item: ChatItem) -> List[Event]:
        if isinstance(item, str):
            return self._delta("message", item)
        if isinstance(item, Thinking):
            return self._delta("reasoning", item.text) if self.show_reasoning else []
        events = []
        if self.show_reasoning and item.reasoning and not item.reasoning_streamed:
            events += self._delta("reasoning", item.reasoning)
        if item.content and not item.streamed:
            events += self._delta("message", item.content)
        events += self._close()
        for call in item.tool_calls:
            index = len(self.output)
            item_id = new_id("fc_")
            events.append(self._event("response.output_item.added", output_index=index,
                                      item=_call_item(item_id, call, "in_progress")))
            events.append(self._event("response.function_call_arguments.delta", item_id=item_id,
                                      output_index=index, delta=call.arguments))
            events.append(self._event("response.function_call_arguments.done", item_id=item_id,
                                      output_index=index, name=call.name, arguments=call.arguments))
            done = _call_item(item_id, call)
            events.append(self._event("response.output_item.done", output_index=index, item=done))
            self.output.append(done)
        events.append(self._event("response.completed", response=self.envelope("completed", self.output, item)))
        return events

    def error(self, message: str) -> Event:
        # The "error" key makes the OpenAI SDK raise instead of passing the event through.
        return self._event("error", code="clibridge_error", message=message, param=None,
                           error={"message": message, "type": "clibridge_error"})
