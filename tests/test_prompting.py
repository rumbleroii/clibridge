import pytest

from clibridge import resolve_model
from clibridge.config import is_bridge_model
from clibridge.prompting import build_prompt, extract_json, parse_tool_reply, tool_schema


@pytest.mark.parametrize(
    "name, backend, model, agent",
    [
        ("claude-code", "claude", None, False),
        ("claude-code/opus", "claude", "opus", False),
        ("claude-code/default", "claude", None, False),
        ("claude-code-agent/sonnet", "claude", "sonnet", True),
        ("claude", "claude", None, False),
        ("haiku", "claude", "haiku", False),
        ("claude-3-5-sonnet-latest", "claude", "sonnet", False),
        ("anthropic/claude-opus-4-1", "claude", "opus", False),
        ("codex", "codex", None, False),
        ("codex/gpt-5-codex", "codex", "gpt-5-codex", False),
        ("codex-agent", "codex", None, True),
        ("gpt-5.1", "codex", "gpt-5.1", False),
        ("gpt-4o-mini", "codex", None, False),
        ("openai/gpt-4o", "codex", None, False),
        ("llama3", "claude", None, False),
        ("", "claude", None, False),
    ],
)
def test_resolve_model(name, backend, model, agent):
    route = resolve_model(name)
    assert (route.backend, route.model, route.agent) == (backend, model, agent)


def test_is_bridge_model():
    for name in ("claude-code", "claude-code/fable", "claude-code-agent/opus", "codex", "codex/gpt-5.5", "codex-agent"):
        assert is_bridge_model(name)
    for name in ("gpt-4o", "claude-sonnet-4-5", "codex-mini-latest", "clibridge", "", None):
        assert not is_bridge_model(name)


def test_unknown_model_uses_default_backend():
    assert resolve_model("mistral-large", default_backend="codex").backend == "codex"


def test_single_user_message_is_passed_through():
    prompt, system, images = build_prompt(
        [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}]
    )
    assert (prompt, system, images) == ("Hi", "Be terse.", [])


def test_conversation_is_rendered_as_transcript():
    prompt, system, _ = build_prompt(
        [
            {"role": "developer", "content": "Rule one."},
            {"role": "user", "content": [{"type": "text", "text": "What's the weather?"}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "rainy"},
        ],
        tool_mode=True,
    )
    assert system == "Rule one."
    assert "<user>\nWhat's the weather?\n</user>" in prompt
    assert '<tool_call id="call_1" name="get_weather">{"city": "Oslo"}</tool_call>' in prompt
    assert '<tool_result id="call_1" name="get_weather">\nrainy\n</tool_result>' in prompt
    assert prompt.endswith("in the required JSON format.")


def test_images_are_collected_from_data_urls():
    pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    prompt, _, images = build_prompt([{"role": "user", "content": [
        {"type": "text", "text": "Describe"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pixel}"}},
    ]}])
    assert images[0][0] == "image/png" and images[0][1].startswith(b"\x89PNG")
    assert "[image 1]" in prompt


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure! Here it is: {"a": [1, 2]} hope that helps', {"a": [1, 2]}),
        ("no json here", None),
    ],
)
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_parse_tool_reply_normalizes_arguments():
    content, calls = parse_tool_reply(
        {"content": "", "tool_calls": [{"name": "f", "arguments": {"x": 1}}, {"name": "g", "arguments": '{"y": 2}'}]},
        "",
    )
    assert content == ""
    assert calls == [("f", '{"x": 1}'), ("g", '{"y": 2}')]


def test_parse_tool_reply_falls_back_to_text():
    assert parse_tool_reply(None, "just words") == ("just words", [])


def test_strict_tool_schema_has_no_unsupported_keywords():
    schema = tool_schema(["f"], "required", parallel=False, args_as_string=True, strict=True)
    calls = schema["properties"]["tool_calls"]
    assert "minItems" not in calls and "maxItems" not in calls
    assert calls["items"]["properties"]["arguments"]["type"] == "string"
    loose = tool_schema(["f", "g"], {"type": "function", "function": {"name": "g"}}, True, False, strict=False)
    assert loose["properties"]["tool_calls"]["minItems"] == 1
    assert loose["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"] == ["g"]
