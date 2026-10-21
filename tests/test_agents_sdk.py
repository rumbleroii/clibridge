"""OpenAI Agents SDK, with both its Responses and Chat Completions model classes."""

import pytest

agents = pytest.importorskip("agents")

import clibridge  # noqa: E402


@agents.function_tool
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f"windy in {city}"


@pytest.fixture(autouse=True)
def no_tracing():
    agents.set_tracing_disabled(True)


@pytest.mark.parametrize("model_cls", [agents.OpenAIResponsesModel, agents.OpenAIChatCompletionsModel])
def test_agent_with_tool(config, model_cls):
    model = model_cls(model="claude-code", openai_client=clibridge.AsyncOpenAI(config=config))
    agent = agents.Agent(name="Weather bot", instructions="Use tools.", model=model, tools=[get_weather])
    result = agents.Runner.run_sync(agent, "What's the weather in Paris?")
    assert result.final_output == "Final answer using: windy in Paris"


def test_streamed_run(config):
    import asyncio

    model = agents.OpenAIResponsesModel(model="codex", openai_client=clibridge.AsyncOpenAI(config=config))
    agent = agents.Agent(name="Echo", instructions="Be brief.", model=model)

    async def go():
        result = agents.Runner.run_streamed(agent, "hello")
        async for _ in result.stream_events():
            pass
        return result.final_output

    assert asyncio.run(go()) == "pong from gpt-default: hello"
