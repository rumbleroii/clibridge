import asyncio
import base64
import json
import os
import socket
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import httpx
import litellm
from agents import Agent, Runner
from anthropic import Anthropic
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
WORKDIR = HERE / "workdir"
os.environ.setdefault("CLIBRIDGE_WORKDIR", str(WORKDIR))

client = OpenAI()
anthropic_client = Anthropic()
SERVER_PORT = 8789
_serve_proc = None


def _png(left_rgb, right_rgb, size=96):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    half = size // 2
    raw = b"".join(
        b"\x00" + bytes(left_rgb) * half + bytes(right_rgb) * (size - half) for _ in range(size)
    )
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_chat(model):
    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "In one sentence, what is idempotency in HTTP?"}],
    )
    return {"reply": reply.choices[0].message.content}


def test_streaming(model):
    start = time.time()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Count from 1 to 5, comma-separated, nothing else."}],
        stream=True,
    )
    parts, chunks, ttft = [], 0, None
    for event in stream:
        delta = event.choices[0].delta.content if event.choices else None
        if delta:
            chunks += 1
            if ttft is None:
                ttft = time.time() - start
            parts.append(delta)
    return {
        "text": "".join(parts),
        "chunks": chunks,
        "time_to_first_chunk_s": round(ttft, 2) if ttft is not None else None,
    }


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def test_tools(model):
    messages = [{"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."}]
    first = client.chat.completions.create(
        model=model, messages=messages, tools=[WEATHER_TOOL], tool_choice="auto"
    )
    message = first.choices[0].message
    calls = message.tool_calls or []
    if not calls:
        return {"tool_called": False, "final": message.content}
    call = calls[0]
    messages.append(message.model_dump(exclude_none=True))
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps({"city": "Paris", "temp_c": 21, "condition": "sunny"}),
        }
    )
    second = client.chat.completions.create(
        model=model, messages=messages, tools=[WEATHER_TOOL], tool_choice="auto"
    )
    return {
        "tool_called": True,
        "tool": call.function.name,
        "arguments": call.function.arguments,
        "final": second.choices[0].message.content,
    }


def test_structured_json(model):
    reply = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract name, city and year from: 'Ada Lovelace was born in London in 1815.' "
                "Return only JSON with keys name, city, year.",
            }
        ],
        response_format={"type": "json_object"},
    )
    raw = reply.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    return {"raw": raw, "parsed": parsed, "valid_json": parsed is not None}


class Person(BaseModel):
    name: str
    city: str
    year: int


def test_structured_pydantic(model):
    reply = client.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Extract name, city and year from: 'Ada Lovelace was born in London in 1815.'",
            }
        ],
        response_format=Person,
    )
    parsed = reply.choices[0].message.parsed
    return {
        "parsed": parsed.model_dump() if parsed else None,
        "valid_schema": parsed is not None,
    }


def test_reasoning(model):
    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Is 391 prime? Work it out before answering."}],
        reasoning_effort="high",
    )
    message = reply.choices[0].message
    reasoning = getattr(message, "reasoning_content", None) or ""
    return {
        "answer": message.content,
        "reasoning_present": bool(reasoning),
        "reasoning_chars": len(reasoning),
        "reasoning_preview": reasoning[:400],
    }


def test_vision(model):
    url = "data:image/png;base64," + base64.b64encode(_png((255, 0, 0), (0, 0, 255))).decode()
    reply = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This image has two solid color halves. Name them left to right, comma-separated.",
                    },
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
    )
    return {"description": reply.choices[0].message.content}


async def test_async(model):
    async_client = AsyncOpenAI()

    async def one(prompt):
        reply = await async_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return reply.choices[0].message.content

    start = time.time()
    outputs = await asyncio.gather(
        one("Reply with only the word: alpha"),
        one("Reply with only the word: beta"),
    )
    return {"outputs": outputs, "wall_seconds": round(time.time() - start, 2)}


def test_anthropic(model):
    reply = anthropic_client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": "Give one vivid sentence describing the ocean at night."}],
    )
    text = "".join(block.text for block in reply.content if block.type == "text")
    return {"reply": text, "stop_reason": reply.stop_reason}


def test_anthropic_thinking(model):
    reply = anthropic_client.messages.create(
        model=model,
        max_tokens=1024,
        thinking={"type": "enabled", "budget_tokens": 8000},
        messages=[{"role": "user", "content": "Is 391 prime? Work it out."}],
    )
    text = "".join(block.text for block in reply.content if block.type == "text")
    thinking = "\n".join(
        getattr(block, "thinking", "") for block in reply.content if block.type == "thinking"
    )
    return {
        "answer": text,
        "block_types": [block.type for block in reply.content],
        "thinking_present": bool(thinking),
        "thinking_chars": len(thinking),
        "thinking_preview": thinking[:400],
    }


def test_litellm(model):
    reply = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: litellm-ok"}],
    )
    return {"reply": reply.choices[0].message.content}


def test_langchain(model):
    llm = ChatOpenAI(model=model, api_key="sk-dummy")
    reply = llm.invoke("Reply with exactly: langchain-ok")
    content = reply.content
    if isinstance(content, list):
        content = " ".join(str(part) for part in content)
    return {"reply": content}


def test_responses(model):
    reply = client.responses.create(
        model=model,
        input="Name the three laws of robotics, one short line each.",
    )
    return {"reply": reply.output_text}


def test_agents(model):
    agent = Agent(name="Helper", instructions="Be concise.", model=model)
    result = Runner.run_sync(agent, "What is 17 * 23? Reply with just the number.")
    return {"reply": result.final_output}


def test_agent_mode(model):
    agent_model = "codex-agent" if model.startswith("codex") else "claude-code-agent"
    reply = client.chat.completions.create(
        model=agent_model,
        messages=[
            {
                "role": "user",
                "content": "Read the file task.txt in the current folder and reply with exactly the codename it contains.",
            }
        ],
    )
    text = reply.choices[0].message.content or ""
    return {
        "model": agent_model,
        "reply": text,
        "found_codename": "NIGHTINGALE-7" in text,
    }


def _port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _start_server():
    global _serve_proc
    if _port_open(SERVER_PORT):
        return
    if _serve_proc is not None and _serve_proc.poll() is None:
        return
    _serve_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "clibridge",
            "serve",
            "--port",
            str(SERVER_PORT),
            "--workdir",
            str(WORKDIR),
        ],
        cwd=str(HERE),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )


def stop_server():
    global _serve_proc
    if _serve_proc is not None and _serve_proc.poll() is None:
        _serve_proc.terminate()
        try:
            _serve_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _serve_proc.kill()
    _serve_proc = None


def test_server_mode(model):
    _start_server()
    deadline = time.time() + 30
    while time.time() < deadline and not _port_open(SERVER_PORT):
        time.sleep(0.5)
    if not _port_open(SERVER_PORT):
        raise RuntimeError("clibridge serve did not start within 30s")
    reply = httpx.post(
        f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: server-ok"}],
        },
        timeout=300,
    )
    reply.raise_for_status()
    data = reply.json()
    return {
        "reply": data["choices"][0]["message"]["content"],
        "http_status": reply.status_code,
    }


CAPABILITIES = [
    ("chat", "OpenAI chat", "Basic chat completion through the OpenAI SDK.", False, test_chat),
    ("streaming", "Streaming", "SSE chunks, time to first chunk, chunk count.", False, test_streaming),
    ("tools", "Tool calling", "Function call round trip with a tool result.", False, test_tools),
    ("structured_json", "JSON mode", "response_format=json_object plus parsing.", False, test_structured_json),
    ("structured_pydantic", "Pydantic output", "beta.chat.completions.parse into a Pydantic model.", False, test_structured_pydantic),
    ("reasoning", "Reasoning effort", "reasoning_effort=high, checks reasoning_content.", False, test_reasoning),
    ("vision", "Vision", "Base64 PNG with two colors, asks the model to describe it.", False, test_vision),
    ("async", "Async", "Two concurrent AsyncOpenAI calls.", True, test_async),
    ("anthropic", "Anthropic SDK", "Messages API via the Anthropic SDK.", False, test_anthropic),
    ("anthropic_thinking", "Anthropic thinking", "thinking blocks via the Anthropic SDK.", False, test_anthropic_thinking),
    ("litellm", "LiteLLM", "litellm.completion with a bridge model.", False, test_litellm),
    ("langchain", "LangChain", "ChatOpenAI invoke with a bridge model.", False, test_langchain),
    ("responses", "Responses API", "OpenAI Responses API.", False, test_responses),
    ("agents", "Agents SDK", "OpenAI Agents SDK runner.", False, test_agents),
    ("agent_mode", "Agent mode", "Reads task.txt through claude-code-agent / codex-agent.", False, test_agent_mode),
    ("server_mode", "Server mode", "clibridge serve + HTTP OpenAI endpoint.", False, test_server_mode),
]

BY_ID = {item[0]: item for item in CAPABILITIES}
