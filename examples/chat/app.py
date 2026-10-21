import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

import capabilities
from chat_tools import TOOLS, execute_tool

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

aclient = AsyncOpenAI()

SYSTEM_PROMPT = (
    "You are clibridge chat, a helpful assistant running locally on the user's machine through "
    "their Claude Code or Codex subscription. Answer clearly and use markdown. Use the provided "
    "tools when they genuinely help (arithmetic, the current time, workspace files) and base your "
    "answer on their results. Never invent tool output."
)

MODELS = [
    "claude-code",
    "claude-code/opus",
    "claude-code/sonnet",
    "claude-code/haiku",
    "codex",
]

Effort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


@asynccontextmanager
async def lifespan(app):
    yield
    capabilities.stop_server()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatIn(BaseModel):
    messages: List[ChatMessage]
    model: Literal[
        "claude-code",
        "claude-code/opus",
        "claude-code/sonnet",
        "claude-code/haiku",
        "codex",
    ] = "claude-code"
    effort: Optional[Effort] = None
    tools: bool = True


def _sse(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _extra(obj, name):
    value = getattr(obj, name, None)
    if value is None:
        extras = getattr(obj, "model_extra", None) or {}
        value = extras.get(name)
    return value


def _history(messages):
    cleaned = []
    for message in messages[-40:]:
        text = message.content.strip()
        if text:
            cleaned.append({"role": message.role, "content": text[:12000]})
    return cleaned


async def _chat_stream(payload: ChatIn, request: Request):
    history = [{"role": "system", "content": SYSTEM_PROMPT}, *_history(payload.messages)]
    tools = TOOLS if payload.tools else None
    started = time.time()
    yield _sse({"type": "start", "model": payload.model})

    for _round in range(6):
        request_args = {
            "model": payload.model,
            "messages": history,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            request_args["tools"] = tools
        if payload.effort:
            request_args["reasoning_effort"] = payload.effort

        try:
            stream = await aclient.chat.completions.create(**request_args)
        except Exception as exc:
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"[:1500]})
            return

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        calls = {}
        usage = None
        finish_reason = None

        try:
            async for chunk in stream:
                if await request.is_disconnected():
                    return
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                reasoning = _extra(delta, "reasoning_content")
                if reasoning:
                    reasoning_parts.append(reasoning)
                    yield _sse({"type": "reasoning", "text": reasoning})

                if getattr(delta, "content", None):
                    content_parts.append(delta.content)
                    yield _sse({"type": "text", "text": delta.content})

                for call in getattr(delta, "tool_calls", None) or []:
                    slot = calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                    if call.id:
                        slot["id"] = call.id
                    function = getattr(call, "function", None)
                    if function is not None:
                        if function.name:
                            slot["name"] = function.name
                        if function.arguments:
                            slot["arguments"] += function.arguments

                if choice.finish_reason:
                    finish_reason = choice.finish_reason
        except Exception as exc:
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"[:1500]})
            return

        if usage is not None:
            yield _sse(
                {
                    "type": "usage",
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            )

        if not calls:
            break

        ordered = [calls[index] for index in sorted(calls)]
        tool_calls = []
        for index, slot in enumerate(ordered):
            tool_calls.append(
                {
                    "id": slot["id"] or f"call_{index}",
                    "type": "function",
                    "function": {"name": slot["name"], "arguments": slot["arguments"] or "{}"},
                }
            )
        history.append(
            {
                "role": "assistant",
                "content": "".join(content_parts) or None,
                "tool_calls": tool_calls,
            }
        )

        for call in tool_calls:
            call_id = call["id"]
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]
            yield _sse({"type": "tool_call", "id": call_id, "name": name, "arguments": arguments})
            result = await asyncio.to_thread(execute_tool, name, arguments)
            ok = json.loads(result).get("ok") is not False
            yield _sse({"type": "tool_result", "id": call_id, "name": name, "ok": ok, "result": result})
            history.append({"role": "tool", "tool_call_id": call_id, "content": result})

    yield _sse({"type": "done", "seconds": round(time.time() - started, 2), "finish_reason": finish_reason})


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse(STATIC / "index.html", media_type="text/html")


@app.get("/api/models")
def models():
    return {"models": MODELS}


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatIn, request: Request):
    return StreamingResponse(
        _chat_stream(payload, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/capabilities")
def list_capabilities():
    return [
        {"id": cap_id, "label": label, "description": description}
        for cap_id, label, description, _is_async, _fn in capabilities.CAPABILITIES
    ]


@app.post("/api/run/{cap_id}")
async def run_capability(cap_id: str, model: str = "claude-code"):
    entry = capabilities.BY_ID.get(cap_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="unknown capability")
    _id, _label, _description, is_async, fn = entry
    start = time.time()
    try:
        result = await fn(model) if is_async else await asyncio.to_thread(fn, model)
        return {"ok": True, "seconds": round(time.time() - start, 2), "result": result}
    except Exception as exc:
        return {
            "ok": False,
            "seconds": round(time.time() - start, 2),
            "error": f"{type(exc).__name__}: {exc}"[:1200],
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8788)

