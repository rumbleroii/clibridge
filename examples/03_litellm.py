"""Plain LiteLLM code."""

import litellm

reply = litellm.completion(
    model="claude-code",  # or "claude-code/fable", "codex", "codex/gpt-5.5", ...
    messages=[{"role": "user", "content": "Give me three fun facts about octopuses."}],
)
print(reply.choices[0].message.content)
