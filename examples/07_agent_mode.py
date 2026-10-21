"""Agent mode: Claude Code works inside a folder with its own tools (reads files, searches...).

It works in the current directory (set CLIBRIDGE_WORKDIR to change it) and is read-only
unless you set CLIBRIDGE_PERMISSION_MODE=acceptEdits.
"""

from openai import OpenAI

reply = OpenAI().chat.completions.create(
    model="claude-code-agent",  # or "codex-agent"
    messages=[{"role": "user", "content": "Look at the files here and summarize this project in 3 bullets."}],
)
print(reply.choices[0].message.content)
