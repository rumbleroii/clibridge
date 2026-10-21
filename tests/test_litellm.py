import asyncio
import json

import pytest

litellm = pytest.importorskip("litellm")

import clibridge.litellm  # noqa: E402

from .conftest import WEATHER_TOOL  # noqa: E402


@pytest.fixture(autouse=True)
def providers(config):
    clibridge.litellm.register(config)
    yield
    litellm.custom_provider_map = [p for p in litellm.custom_provider_map
                                   if p["provider"] not in clibridge.litellm.PROVIDERS]


def test_completion():
    reply = litellm.completion(model="claude-code/sonnet", messages=[{"role": "user", "content": "hi"}])
    assert reply.choices[0].message.content == "pong from sonnet: hi"
    assert reply.usage.total_tokens == 122


def test_default_model_and_codex():
    reply = litellm.completion(model="codex/default", messages=[{"role": "user", "content": "hi"}])
    assert reply.choices[0].message.content == "pong from gpt-default: hi"


def test_streaming():
    chunks = litellm.completion(model="claude-code/default", messages=[{"role": "user", "content": "s"}], stream=True)
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "pong from claude-default: s"


def test_tools():
    reply = litellm.completion(model="codex/default", messages=[{"role": "user", "content": "weather?"}],
                               tools=[WEATHER_TOOL])
    call = reply.choices[0].message.tool_calls[0]
    assert call.function.name == "get_weather" and json.loads(call.function.arguments) == {"city": "Paris"}


def test_streaming_tools():
    chunks = list(litellm.completion(model="claude-code/default", messages=[{"role": "user", "content": "weather?"}],
                                     tools=[WEATHER_TOOL], stream=True))
    calls = [tc for c in chunks for tc in (c.choices[0].delta.tool_calls or [])]
    assert calls and calls[0].function.name == "get_weather"


def test_async():
    async def go():
        reply = await litellm.acompletion(model="claude-code/default", messages=[{"role": "user", "content": "a"}])
        stream = await litellm.acompletion(model="claude-code/default", messages=[{"role": "user", "content": "b"}],
                                           stream=True)
        text = "".join([c.choices[0].delta.content or "" async for c in stream])
        return reply.choices[0].message.content, text

    assert asyncio.run(go()) == ("pong from claude-default: a", "pong from claude-default: b")


def test_errors_surface():
    with pytest.raises(Exception, match="Invalid API key"):
        litellm.completion(model="claude-code/default", messages=[{"role": "user", "content": "FAIL"}])
