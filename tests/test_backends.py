"""How requests turn into CLI invocations, and how CLI output and failures come back."""

import asyncio

import pytest

from clibridge import BridgeConfig, BridgeError, aiter_chat, complete_chat, iter_chat
from clibridge.backends import chat_dir

from .conftest import FAKE_CLI, WEATHER_TOOL


def user(text):
    return [{"role": "user", "content": text}]


def test_claude_chat_mode_runs_without_tools(config, calls):
    result = complete_chat({"model": "claude-code/opus", "messages": [
        {"role": "system", "content": "Be terse."}, {"role": "user", "content": "hello"}]}, config)
    assert result.content == "pong from opus: hello"
    assert result.finish_reason == "stop"
    assert (result.prompt_tokens, result.completion_tokens, result.cached_tokens) == (115, 7, 100)
    assert result.meta["cost_usd"] == 0.0012
    call = calls()[0]
    argv = call["argv"]
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--system-prompt") + 1] == "Be terse."
    for flag in ("-p", "--no-session-persistence", "--strict-mcp-config", "--safe-mode"):
        assert flag in argv
    assert "--include-partial-messages" not in argv
    assert call["cwd"] == chat_dir()


def test_claude_streams_deltas(config):
    items = list(iter_chat({"model": "claude-code", "messages": user("stream please"), "stream": True}, config))
    deltas, result = items[:-1], items[-1]
    assert len(deltas) > 1
    assert "".join(deltas) == "pong from claude-default: stream please"
    assert result.streamed


def test_claude_agent_mode(config, calls, tmp_path):
    config.permission_mode = "acceptEdits"
    request = {"model": "claude-code-agent", "messages": [
        {"role": "system", "content": "Extra rules."}, {"role": "user", "content": "AGENT list files"}],
        "stream": True, "cwd": str(tmp_path), "max_turns": 5, "reasoning_effort": "minimal"}
    items = list(iter_chat(request, config))
    text = "".join(i for i in items if isinstance(i, str))
    assert text.startswith("Let me look around.\n\npong from")
    assert "SUBAGENT" not in text
    call = calls()[0]
    argv = call["argv"]
    assert "--tools" not in argv and "--system-prompt" not in argv
    assert argv[argv.index("--append-system-prompt") + 1] == "Extra rules."
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
    assert argv[argv.index("--max-turns") + 1] == "5"
    assert argv[argv.index("--effort") + 1] == "low"
    assert call["cwd"] == str(tmp_path)


def test_codex_chat_mode(config, calls):
    result = complete_chat({"model": "codex/gpt-5-codex", "messages": [
        {"role": "system", "content": "Be terse."}, {"role": "user", "content": "hello"}],
        "reasoning_effort": "high"}, config)
    assert result.content == "pong from gpt-5-codex: hello"
    assert (result.prompt_tokens, result.completion_tokens, result.cached_tokens) == (120, 9, 20)
    call = calls()[0]
    argv = call["argv"]
    assert argv[:2] == ["exec", "--json"]
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert argv[argv.index("--config") + 1] == 'model_reasoning_effort="high"'
    assert "--ephemeral" in argv and "--skip-git-repo-check" in argv
    assert call["prompt"].startswith("<instructions>\nAnswer directly")
    assert "Be terse." in call["prompt"] and call["prompt"].endswith("hello")


def test_codex_agent_mode_reports_all_messages_when_streaming(config, calls):
    config.sandbox = "workspace-write"
    items = list(iter_chat({"model": "codex-agent", "messages": user("fix it"), "stream": True}, config))
    assert "".join(i for i in items if isinstance(i, str)).startswith("Checked the files.\n\npong from")
    assert items[-1].content.startswith("pong from")
    argv = calls()[0]["argv"]
    assert argv[argv.index("--sandbox") + 1] == "workspace-write"


@pytest.mark.parametrize("model", ["claude-code", "codex"])
def test_tool_calls(config, calls, model):
    result = complete_chat({"model": model, "messages": user("weather in Paris?"), "tools": [WEATHER_TOOL]}, config)
    assert result.finish_reason == "tool_calls"
    assert result.content is None
    assert [(c.name, c.arguments) for c in result.tool_calls] == [("get_weather", '{"city": "Paris"}')]
    assert result.tool_calls[0].id.startswith("call_")
    call = calls()[0]
    if model == "codex":
        assert call["schema"]["properties"]["tool_calls"]["items"]["properties"]["arguments"]["type"] == "string"
    else:
        assert "--json-schema" in call["argv"]


def test_tool_choice_none_disables_tools(config, calls):
    result = complete_chat({"model": "claude-code", "messages": user("hi"), "tools": [WEATHER_TOOL],
                            "tool_choice": "none"}, config)
    assert result.tool_calls == [] and result.content.startswith("pong")
    assert "--json-schema" not in calls()[0]["argv"]


def test_json_schema_response_format(config, calls):
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
    fmt = {"type": "json_schema", "json_schema": {"name": "person", "schema": schema}}
    result = complete_chat({"model": "claude-code", "messages": user("Ada, 36"), "response_format": fmt}, config)
    assert result.content == '{"name": "text", "age": 1}'
    # Codex only gets strict schemas.
    complete_chat({"model": "codex", "messages": user("Ada"), "response_format": fmt}, config)
    assert calls()[-1]["schema"] is None


def test_parameters_the_clis_lack_are_ignored(config, calls):
    request = {"model": "claude-code", "messages": user("abc"), "stop": [":"], "max_tokens": 1,
               "temperature": 0.1, "top_p": 0.5, "n": 3, "seed": 7}
    result = complete_chat(request, config)
    assert (result.content, result.finish_reason) == ("pong from claude-default: abc", "stop")
    assert not any(flag in calls()[0]["argv"] for flag in ("--temperature", "--max-tokens", "--stop"))


def test_images_are_forwarded(config, calls):
    pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    content = [{"type": "text", "text": "what is this"},
               {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pixel}"}}]
    for model in ("claude-code", "codex"):
        complete_chat({"model": model, "messages": [{"role": "user", "content": content}]}, config)
    claude_call, codex_call = calls()
    assert claude_call["images"] == 1 and "--input-format" in claude_call["argv"]
    assert codex_call["images"] == 1


def test_cli_error_result_raises(config):
    with pytest.raises(BridgeError, match="Invalid API key") as info:
        complete_chat({"model": "claude-code", "messages": user("FAIL")}, config)
    assert info.value.status == 424


def test_codex_failure_maps_to_rate_limit(config):
    with pytest.raises(BridgeError, match="usage limit") as info:
        complete_chat({"model": "codex", "messages": user("FAIL")}, config)
    assert info.value.status == 429


def test_crash_reports_stderr(config):
    with pytest.raises(BridgeError, match="boom: something broke"):
        complete_chat({"model": "claude-code", "messages": user("CRASH")}, config)


def test_timeout_kills_the_cli(config):
    config.timeout = 1
    with pytest.raises(BridgeError, match="timed out") as info:
        complete_chat({"model": "claude-code", "messages": user("SLOW")}, config)
    assert info.value.status == 504


def test_missing_binary():
    config = BridgeConfig(claude_bin="definitely-not-installed-xyz")
    with pytest.raises(BridgeError, match="Could not find"):
        complete_chat({"model": "claude-code", "messages": user("hi")}, config)


def test_empty_messages_are_rejected(config):
    with pytest.raises(BridgeError) as info:
        complete_chat({"model": "claude-code", "messages": []}, config)
    assert info.value.status == 400


def test_api_keys_are_scrubbed_from_cli_env(config, calls, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    complete_chat({"model": "claude-code", "messages": user("hi")}, config)
    complete_chat({"model": "codex", "messages": user("hi")}, config)
    assert not calls()[0]["env_has_anthropic_key"] and not calls()[1]["env_has_openai_key"]
    config.scrub_env = False
    complete_chat({"model": "claude-code", "messages": user("hi")}, config)
    assert calls()[2]["env_has_anthropic_key"]


def test_async_matches_sync(config):
    async def go():
        items = [i async for i in aiter_chat({"model": "claude-code", "messages": user("hey"), "stream": True}, config)]
        return items

    items = asyncio.run(go())
    assert "".join(items[:-1]) == "pong from claude-default: hey"


def test_async_timeout(config):
    config.timeout = 1

    async def go():
        return [i async for i in aiter_chat({"model": "claude-code", "messages": user("SLOW")}, config)]

    with pytest.raises(BridgeError, match="timed out"):
        asyncio.run(go())


def test_fake_cli_dir_exists():
    assert (FAKE_CLI / "claude").exists()
