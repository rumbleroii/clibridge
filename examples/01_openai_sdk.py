"""Plain OpenAI SDK code. The only special thing is the model name."""

from openai import OpenAI

client = OpenAI()  # no API key needed for clibridge models

reply = client.chat.completions.create(
    model="claude-code",  # or "claude-code/fable", "codex", "codex/gpt-5.5", ...
    messages=[{"role": "user", "content": "Explain recursion in one sentence."}],
)
print(reply.choices[0].message.content)

# Streaming works too.
stream = client.chat.completions.create(
    model="claude-code",
    messages=[{"role": "user", "content": "Write a haiku about the terminal."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
