"""Plain LangChain + LangGraph agent with a tool.  pip install langgraph"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's 21C and sunny in {city}."


llm = ChatOpenAI(model="claude-code")
agent = create_react_agent(llm, [get_weather])
result = agent.invoke({"messages": [("user", "What's the weather in Lisbon?")]})
print(result["messages"][-1].content)
