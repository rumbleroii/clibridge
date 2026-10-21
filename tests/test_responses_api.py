"""OpenAI Responses API (client.responses.*)."""

import json

import pytest

pytest.importorskip("openai")

import clibridge  # noqa: E402

TOOL = {"type": "function", "name": "get_weather", "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}


def test_create(config):
    client = clibridge.OpenAI(config=config)
    response = client.responses.create(model="claude-code", input="hello", instructions="Be brief.")
    assert response.output_text == "pong from claude-default: hello"
    assert response.status == "completed"
    assert response.usage.input_tokens == 115


def test_stream(config):
    client = clibridge.OpenAI(config=config)
    events = list(client.responses.create(model="claude-code", input="hello", stream=True))
    types = [e.type for e in events]
    assert types[0] == "response.created" and types[-1] == "response.completed"
    text = "".join(e.delta for e in events if e.type == "response.output_text.delta")
    assert text == "pong from claude-default: hello"
    assert events[-1].response.output_text == text


def test_function_calls_and_previous_response_id(config, calls):
    client = clibridge.OpenAI(config=config)
    first = client.responses.create(model="codex", input="weather in Paris?", tools=[TOOL])
    call = next(item for item in first.output if item.type == "function_call")
    assert call.name == "get_weather" and json.loads(call.arguments) == {"city": "Paris"}

    second = client.responses.create(
        model="codex", previous_response_id=first.id, tools=[TOOL],
        input=[{"type": "function_call_output", "call_id": call.call_id, "output": "cloudy"}],
    )
    assert second.output_text == "Final answer using: cloudy"
    assert "weather in Paris?" in calls()[-1]["prompt"]


def test_stream_with_function_call(config):
    client = clibridge.OpenAI(config=config)
    with client.responses.stream(model="claude-code", input="weather?", tools=[TOOL]) as stream:
        final = stream.get_final_response()
    assert final.output[0].type == "function_call"


def test_unknown_previous_response_id(config):
    import openai

    with pytest.raises(openai.BadRequestError):
        clibridge.OpenAI(config=config).responses.create(model="claude-code", input="x", previous_response_id="nope")
