"""Thinking: set how hard the model thinks, and read what it thought.

The thinking you get back is a short summary (that's what the CLIs show), and the
model decides when to think, so easy questions often come back without any.
"""

from anthropic import Anthropic
from openai import OpenAI

# OpenAI style: reasoning_effort in, message.reasoning_content out.
reply = OpenAI().chat.completions.create(
    model="claude-code",
    reasoning_effort="high",  # none, minimal, low, medium, high, xhigh, max
    messages=[{"role": "user", "content": "Is 391 a prime number?"}],
)
message = reply.choices[0].message
print("Thinking:", getattr(message, "reasoning_content", None))
print("Answer:  ", message.content)

# Anthropic style: thinking={...} in, thinking blocks out.
answer = Anthropic().messages.create(
    model="claude-code",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": "Is 323 a prime number?"}],
)
for block in answer.content:
    print(f"{block.type}: {getattr(block, 'thinking', None) or getattr(block, 'text', '')}")
