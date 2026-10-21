import asyncio

import pytest

pytest.importorskip("langchain_openai")

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from clibridge.langchain import ChatClaudeCode, ChatCodex  # noqa: E402


@tool
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f"sunny in {city}"


def test_invoke_and_stream(config):
    llm = ChatClaudeCode(model="opus", config=config)
    reply = llm.invoke([SystemMessage("Be terse."), HumanMessage("hello")])
    assert reply.content == "pong from opus: hello"
    assert reply.usage_metadata["total_tokens"] == 122
    chunks = list(llm.stream("streaming"))
    assert "".join(c.content for c in chunks) == "pong from opus: streaming"


def test_model_names(config):
    assert ChatClaudeCode(config=config).model_name == "claude-code"
    assert ChatClaudeCode(model="haiku", agent=True, config=config).model_name == "claude-code-agent/haiku"
    assert ChatCodex(model="gpt-5-codex", config=config).model_name == "codex/gpt-5-codex"
    assert ChatCodex(config=config).use_responses_api is False


@pytest.mark.parametrize("cls", [ChatClaudeCode, ChatCodex])
def test_bind_tools_round_trip(config, cls):
    llm = cls(config=config).bind_tools([get_weather])
    messages = [HumanMessage("weather in Paris?")]
    first = llm.invoke(messages)
    assert first.tool_calls[0]["name"] == "get_weather"
    assert first.tool_calls[0]["args"] == {"city": "Paris"}
    messages += [first, ToolMessage(get_weather.invoke(first.tool_calls[0]["args"]),
                                    tool_call_id=first.tool_calls[0]["id"])]
    assert llm.invoke(messages).content == "Final answer using: sunny in Paris"


@pytest.mark.parametrize("method", ["json_schema", "function_calling"])
def test_structured_output(config, method):
    class Person(BaseModel):
        name: str
        age: int

    llm = ChatClaudeCode(config=config).with_structured_output(Person, method=method)
    assert llm.invoke("Ada, 36") == Person(name="text", age=1)


def test_json_mode_is_enforced_with_a_schema(config, calls):
    reply = ChatClaudeCode(config=config).bind(response_format={"type": "json_object"}).invoke("JSON please")
    assert reply.content == "{}"
    argv = calls()[0]["argv"]
    assert argv[argv.index("--json-schema") + 1] == '{"type": "object"}'


def test_async(config):
    async def go():
        llm = ChatClaudeCode(config=config)
        reply = await llm.ainvoke("async hi")
        parts = [c.content async for c in llm.astream("async stream")]
        return reply.content, "".join(parts)

    assert asyncio.run(go()) == ("pong from claude-default: async hi", "pong from claude-default: async stream")


def test_langgraph_react_agent(config):
    prebuilt = pytest.importorskip("langgraph.prebuilt")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent = prebuilt.create_react_agent(ChatClaudeCode(config=config), [get_weather])
    state = agent.invoke({"messages": [HumanMessage("What's the weather in Paris?")]})
    assert state["messages"][-1].content == "Final answer using: sunny in Paris"
    assert any(isinstance(m, ToolMessage) for m in state["messages"])
