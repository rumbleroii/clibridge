"""Plain OpenAI Agents SDK code.  pip install openai-agents"""

from agents import Agent, Runner, function_tool, set_tracing_disabled

set_tracing_disabled(True)  # tracing would try to upload to OpenAI


@function_tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


agent = Agent(name="Math helper", instructions="Use the add tool for arithmetic.", model="claude-code", tools=[add])
print(Runner.run_sync(agent, "What is 1234 + 4321?").final_output)
