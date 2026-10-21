"""The real anthropic SDK against the in-process transport."""

import asyncio

import pytest

anthropic = pytest.importorskip("anthropic")

import clibridge  # noqa: E402

TOOL = {"name": "get_weather", "description": "Weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}


def test_messages_create(config, calls):
    client = clibridge.Anthropic(config=config)
    message = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024, system="Be brief.",
                                     messages=[{"role": "user", "content": "hello"}])
    assert message.content[0].text == "pong from sonnet: hello"
    assert message.stop_reason == "end_turn"
    assert message.usage.output_tokens == 7
    argv = calls()[0]["argv"]
    assert argv[argv.index("--system-prompt") + 1] == "Be brief."


def test_messages_stream(config):
    client = clibridge.Anthropic(config=config)
    with client.messages.stream(model="claude-code", max_tokens=64000,
                                messages=[{"role": "user", "content": "hello"}]) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
    assert text == "pong from claude-default: hello"
    assert final.content[0].text == text


def test_tool_use_round_trip(config):
    client = clibridge.Anthropic(config=config)
    messages = [{"role": "user", "content": "weather in Paris?"}]
    first = client.messages.create(model="claude-code", max_tokens=1024, messages=messages, tools=[TOOL])
    assert first.stop_reason == "tool_use"
    tool_use = next(b for b in first.content if b.type == "tool_use")
    assert tool_use.input == {"city": "Paris"}

    messages += [{"role": "assistant", "content": first.content},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": "warm"}]}]
    second = client.messages.create(model="claude-code", max_tokens=1024, messages=messages, tools=[TOOL])
    assert second.content[0].text == "Final answer using: warm"


def test_streamed_tool_use(config):
    client = clibridge.Anthropic(config=config)
    with client.messages.stream(model="codex", max_tokens=1024, tools=[TOOL],
                                messages=[{"role": "user", "content": "weather?"}]) as stream:
        final = stream.get_final_message()
    assert final.content[0].type == "tool_use" and final.content[0].input == {"city": "Paris"}


def test_errors(config):
    client = clibridge.Anthropic(config=config)
    with pytest.raises(anthropic.RateLimitError):
        client.messages.create(model="codex", max_tokens=10, messages=[{"role": "user", "content": "FAIL"}])


def test_async(config):
    async def go():
        client = clibridge.AsyncAnthropic(config=config)
        msg = await client.messages.create(model="claude-code", max_tokens=10,
                                           messages=[{"role": "user", "content": "yo"}])
        return msg.content[0].text

    assert asyncio.run(go()) == "pong from claude-default: yo"


def test_count_tokens(config):
    count = clibridge.Anthropic(config=config).messages.count_tokens(
        model="claude-code", messages=[{"role": "user", "content": "hello world " * 10}])
    assert count.input_tokens > 10


def test_against_http_server(config):
    import threading

    from clibridge.server import BridgeServer

    server = BridgeServer("127.0.0.1", 0, config)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = anthropic.Anthropic(base_url=f"http://127.0.0.1:{server.server_port}", api_key="anything")
        with client.messages.stream(model="claude-code", max_tokens=100,
                                    messages=[{"role": "user", "content": "over http"}]) as stream:
            assert "".join(stream.text_stream) == "pong from claude-default: over http"
    finally:
        server.shutdown()
        server.server_close()
