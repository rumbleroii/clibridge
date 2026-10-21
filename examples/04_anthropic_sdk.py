"""Plain Anthropic SDK code."""

from anthropic import Anthropic

client = Anthropic()

with client.messages.stream(
    model="claude-code",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a limerick about debugging."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
print()
