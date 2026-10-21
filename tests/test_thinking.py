"""Thinking levels in, thinking text out."""

import json

import pytest

from clibridge import complete_chat, iter_chat
from clibridge.core import Thinking


def ask(text, **extra):
    return {"messages": [{"role": "user", "content": text}], **extra}


def test_claude_thinking_text_and_tokens(config, calls):
    result = complete_chat(ask("THINK about 391", model="claude-code"), config)
    assert result.reasoning == "Considering: THINK about 391"
    assert result.reasoning_tokens == 42
    argv = calls()[0]["argv"]
    assert argv[argv.index("--thinking-display") + 1] == "summarized"


def test_claude_streams_thinking_before_text(config):
    items = list(iter_chat(ask("THINK hard", model="claude-code", stream=True), config))
    kinds = [type(i).__name__ for i in items]
    assert kinds[0] == "Thinking" and kinds.index("str") > 0
    assert "".join(i.text for i in items if isinstance(i, Thinking)) == "Considering: THINK hard"
    assert items[-1].reasoning_streamed


def test_codex_reasoning(config):
    result = complete_chat(ask("hi", model="codex"), config)
    assert result.reasoning == "**Thinking**" and result.reasoning_tokens == 4


@pytest.mark.parametrize(
    "effort, flag, disabled",
    [("high", "high", False), ("max", "max", False), ("minimal", "low", False), ("none", None, True)],
)
def test_claude_effort_levels(config, calls, effort, flag, disabled):
    complete_chat(ask("hi", model="claude-code", reasoning_effort=effort), config)
    call = calls()[0]
    assert (call["argv"][call["argv"].index("--effort") + 1] if "--effort" in call["argv"] else None) == flag
    assert call["thinking_disabled"] == disabled


@pytest.mark.parametrize("effort, sent", [("max", "xhigh"), ("minimal", "low"), ("none", "none"), ("high", "high")])
def test_codex_effort_levels(config, calls, effort, sent):
    complete_chat(ask("hi", model="codex", reasoning_effort=effort), config)
    argv = calls()[0]["argv"]
    assert argv[argv.index("--config") + 1] == f"model_reasoning_effort={json.dumps(sent)}"


def test_no_effort_means_the_cli_default(config, calls):
    complete_chat(ask("hi", model="claude-code"), config)
    complete_chat(ask("hi", model="codex"), config)
    assert "--effort" not in calls()[0]["argv"] and "--config" not in calls()[1]["argv"]


def test_rejected_optional_flag_is_retried_and_remembered(config, calls, monkeypatch):
    monkeypatch.setenv("FAKE_REJECT_FLAGS", "--thinking-display,--safe-mode")
    for _ in range(2):
        assert complete_chat(ask("hi", model="claude-code"), config).content.startswith("pong")
    argvs = [c["argv"] for c in calls()]
    assert len(argvs) == 2  # the rejected attempts exit before logging
    assert all("--thinking-display" not in a and "--safe-mode" not in a for a in argvs)


def test_rejected_codex_flag_is_retried(config, calls, monkeypatch):
    monkeypatch.setenv("FAKE_REJECT_FLAGS", "--ephemeral")
    assert complete_chat(ask("hi", model="codex"), config).content.startswith("pong")
    assert "--ephemeral" not in calls()[0]["argv"]


def test_rejected_required_flag_is_an_error(config, monkeypatch):
    from clibridge import BridgeError

    monkeypatch.setenv("FAKE_REJECT_FLAGS", "--no-session-persistence")
    with pytest.raises(BridgeError, match="unknown option"):
        complete_chat(ask("hi", model="claude-code"), config)


def test_async_retry(config, monkeypatch):
    import asyncio

    from clibridge import acomplete_chat

    monkeypatch.setenv("FAKE_REJECT_FLAGS", "--thinking-display")
    result = asyncio.run(acomplete_chat(ask("THINK x", model="claude-code"), config))
    assert result.content.startswith("pong") and result.reasoning == ""  # no summaries without the flag


# --- wire formats ---

openai = pytest.importorskip("openai")


def test_openai_reasoning_content(config):
    import clibridge

    client = clibridge.OpenAI(config=config)
    reply = client.chat.completions.create(model="claude-code", messages=[{"role": "user", "content": "THINK a"}],
                                           reasoning_effort="high")
    assert reply.choices[0].message.reasoning_content == "Considering: THINK a"
    assert reply.usage.completion_tokens_details.reasoning_tokens == 42
    stream = client.chat.completions.create(model="claude-code", stream=True,
                                            messages=[{"role": "user", "content": "THINK b"}])
    chunks = [c.choices[0].delta for c in stream if c.choices]
    thinking = "".join(getattr(d, "reasoning_content", None) or "" for d in chunks)
    assert thinking == "Considering: THINK b"
    assert "".join(d.content or "" for d in chunks) == "pong from claude-default: THINK b"


def test_openai_reasoning_with_tools(config):
    import clibridge

    from .conftest import WEATHER_TOOL

    reply = clibridge.OpenAI(config=config).chat.completions.create(
        model="claude-code", messages=[{"role": "user", "content": "THINK weather?"}], tools=[WEATHER_TOOL])
    assert reply.choices[0].message.tool_calls
    assert reply.choices[0].message.reasoning_content.startswith("Considering")


anthropic = pytest.importorskip("anthropic")


def test_anthropic_thinking_only_when_enabled(config, calls):
    import clibridge

    client = clibridge.Anthropic(config=config)
    plain = client.messages.create(model="claude-code", max_tokens=100, messages=[{"role": "user", "content": "THINK x"}])
    assert [b.type for b in plain.content] == ["text"]

    thoughtful = client.messages.create(model="claude-code", max_tokens=20000,
                                        thinking={"type": "enabled", "budget_tokens": 8000},
                                        messages=[{"role": "user", "content": "THINK y"}])
    assert [b.type for b in thoughtful.content] == ["thinking", "text"]
    assert thoughtful.content[0].thinking == "Considering: THINK y"
    argv = calls()[-1]["argv"]
    assert argv[argv.index("--effort") + 1] == "medium"


def test_anthropic_thinking_stream(config):
    import clibridge

    client = clibridge.Anthropic(config=config)
    with client.messages.stream(model="claude-code", max_tokens=20000, thinking={"type": "adaptive"},
                                messages=[{"role": "user", "content": "THINK z"}]) as stream:
        events = [e.type for e in stream]
        final = stream.get_final_message()
    assert "thinking" in events
    assert [b.type for b in final.content] == ["thinking", "text"]
    assert final.content[0].thinking == "Considering: THINK z"


def test_anthropic_thinking_disabled(config, calls):
    import clibridge

    clibridge.Anthropic(config=config).messages.create(
        model="claude-code", max_tokens=100, thinking={"type": "disabled"},
        messages=[{"role": "user", "content": "THINK q"}])
    assert calls()[0]["thinking_disabled"]


def test_responses_reasoning_items(config):
    import clibridge

    client = clibridge.OpenAI(config=config)
    plain = client.responses.create(model="claude-code", input="THINK a")
    assert [o.type for o in plain.output] == ["message"]

    response = client.responses.create(model="claude-code", input="THINK b", reasoning={"effort": "high"})
    assert [o.type for o in response.output] == ["reasoning", "message"]
    assert response.output[0].summary[0].text == "Considering: THINK b"
    assert response.usage.output_tokens_details.reasoning_tokens == 42

    events = list(client.responses.create(model="claude-code", input="THINK c", reasoning={"effort": "low"},
                                          stream=True))
    deltas = "".join(e.delta for e in events if e.type == "response.reasoning_summary_text.delta")
    assert deltas == "Considering: THINK c"
    final = events[-1].response
    assert [o.type for o in final.output] == ["reasoning", "message"] and final.output_text.startswith("pong")


def test_litellm_reasoning(config, calls):
    litellm = pytest.importorskip("litellm")
    import clibridge.litellm

    clibridge.litellm.register(config)
    try:
        reply = litellm.completion(model="claude-code/default", messages=[{"role": "user", "content": "THINK l"}],
                                   reasoning_effort="high")
        assert reply.choices[0].message.reasoning_content == "Considering: THINK l"
        litellm.completion(model="codex/default", messages=[{"role": "user", "content": "x"}],
                           thinking={"type": "enabled", "budget_tokens": 20000})
        claude_argv, codex_argv = calls()[0]["argv"], calls()[1]["argv"]
        assert claude_argv[claude_argv.index("--effort") + 1] == "high"
        assert codex_argv[codex_argv.index("--config") + 1] == 'model_reasoning_effort="high"'
        chunks = litellm.completion(model="claude-code/default", messages=[{"role": "user", "content": "THINK m"}],
                                    stream=True)
        assert "".join(c.choices[0].delta.content or "" for c in chunks) == "pong from claude-default: THINK m"
    finally:
        litellm.custom_provider_map = [p for p in litellm.custom_provider_map
                                       if p["provider"] not in clibridge.litellm.PROVIDERS]
