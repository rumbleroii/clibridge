"""The real openai SDK against the in-process transport and the HTTP server."""

import asyncio
import json
import threading

import pytest

openai = pytest.importorskip("openai")
pydantic = pytest.importorskip("pydantic")

import clibridge  # noqa: E402
from clibridge.server import BridgeServer  # noqa: E402

from .conftest import WEATHER_TOOL  # noqa: E402


@pytest.fixture(params=["in-process", "server"])
def client(request, config):
    if request.param == "in-process":
        yield clibridge.OpenAI(config=config)
        return
    server = BridgeServer("127.0.0.1", 0, config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield openai.OpenAI(base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="anything", max_retries=0)
    server.shutdown()
    server.server_close()


def test_chat_completion(client):
    reply = client.chat.completions.create(model="claude-code", messages=[{"role": "user", "content": "hi"}])
    assert reply.choices[0].message.content == "pong from claude-default: hi"
    assert reply.choices[0].finish_reason == "stop"
    assert reply.usage.total_tokens == 122
    assert reply.model == "claude-code"


def test_streaming_with_usage(client):
    stream = client.chat.completions.create(
        model="codex", messages=[{"role": "user", "content": "hi"}], stream=True,
        stream_options={"include_usage": True},
    )
    chunks = list(stream)
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "pong from gpt-default: hi"
    assert chunks[-2].choices[0].finish_reason == "stop"
    assert chunks[-1].usage.prompt_tokens == 120


def test_tool_calling_round_trip(client):
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    first = client.chat.completions.create(model="claude-code", messages=messages, tools=[WEATHER_TOOL])
    message = first.choices[0].message
    assert first.choices[0].finish_reason == "tool_calls"
    call = message.tool_calls[0]
    assert call.function.name == "get_weather" and json.loads(call.function.arguments) == {"city": "Paris"}

    messages += [message.model_dump(exclude_none=True),
                 {"role": "tool", "tool_call_id": call.id, "content": "sunny, 24C"}]
    second = client.chat.completions.create(model="claude-code", messages=messages, tools=[WEATHER_TOOL])
    assert second.choices[0].message.content == "Final answer using: sunny, 24C"
    assert not second.choices[0].message.tool_calls


def test_streamed_tool_calls(client):
    stream = client.chat.completions.create(
        model="codex", messages=[{"role": "user", "content": "weather?"}], tools=[WEATHER_TOOL], stream=True)
    calls = [tc for chunk in stream for tc in (chunk.choices[0].delta.tool_calls or [])]
    assert calls[0].function.name == "get_weather"
    assert json.loads(calls[0].function.arguments) == {"city": "Paris"}


def test_structured_output_parse(client):
    class Person(pydantic.BaseModel):
        name: str
        age: int

    completion = client.chat.completions.parse(
        model="claude-code", messages=[{"role": "user", "content": "Ada Lovelace, 36"}], response_format=Person)
    assert completion.choices[0].message.parsed == Person(name="text", age=1)


def test_models_list(client):
    assert "claude-code" in [m.id for m in client.models.list()]


def test_errors_are_raised(client):
    with pytest.raises(openai.APIStatusError) as info:
        client.chat.completions.create(model="claude-code", messages=[{"role": "user", "content": "FAIL"}])
    assert info.value.status_code == 424
    assert "Invalid API key" in str(info.value)


def test_async_client(config):
    async def go():
        client = clibridge.AsyncOpenAI(config=config)
        reply = await client.chat.completions.create(model="claude-code", messages=[{"role": "user", "content": "a"}])
        stream = await client.chat.completions.create(
            model="claude-code", messages=[{"role": "user", "content": "b"}], stream=True)
        text = "".join([c.choices[0].delta.content or "" async for c in stream])
        return reply.choices[0].message.content, text

    assert asyncio.run(go()) == ("pong from claude-default: a", "pong from claude-default: b")


def test_server_requires_api_key_when_configured(config):
    config.api_key = "secret"
    server = BridgeServer("127.0.0.1", 0, config)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1"
        with pytest.raises(openai.AuthenticationError):
            openai.OpenAI(base_url=url, api_key="wrong", max_retries=0).models.list()
        assert openai.OpenAI(base_url=url, api_key="secret").models.list().data
    finally:
        server.shutdown()
        server.server_close()


def test_unexpected_errors_become_500(config, monkeypatch):
    import clibridge.app

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(clibridge.app, "complete_chat", boom)
    with pytest.raises(openai.InternalServerError, match="kaboom"):
        clibridge.OpenAI(config=config).chat.completions.create(
            model="claude-code", messages=[{"role": "user", "content": "x"}])
