"""The startup hook: plain SDK code, no clibridge import, no API key, no server.

Each test runs a fresh interpreter, because the hook is installed at startup by clibridge.pth.
"""

import json
import os
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from .conftest import fake_cli

pytestmark = pytest.mark.skipif(
    subprocess.run([sys.executable, "-c", "import _clibridge_autoload"], capture_output=True).returncode != 0,
    reason="clibridge is not installed with its startup hook (pip install -e .)",
)


def run(code, cwd=None, **env):
    full_env = {k: v for k, v in os.environ.items()
                if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLIBRIDGE_AUTOLOAD")}
    full_env.update(CLIBRIDGE_CLAUDE_BIN=fake_cli("claude"), CLIBRIDGE_CODEX_BIN=fake_cli("codex"))
    full_env.update(env)
    proc = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, env=full_env, timeout=120, cwd=cwd)
    return proc


def output(code, cwd=None, **env):
    proc = run(code, cwd=cwd, **env)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()


def test_plain_openai():
    pytest.importorskip("openai")
    assert output("""
        from openai import OpenAI
        client = OpenAI()
        print(client.chat.completions.create(model="claude-code", messages=[{"role": "user", "content": "a"}]).choices[0].message.content)
        print("".join(c.choices[0].delta.content or "" for c in client.chat.completions.create(
            model="codex", messages=[{"role": "user", "content": "b"}], stream=True)))
        print(client.responses.create(model="claude-code/opus", input="c").output_text)
    """) == ["pong from claude-default: a", "pong from gpt-default: b", "pong from opus: c"]


def test_plain_async_openai():
    pytest.importorskip("openai")
    assert output("""
        import asyncio
        from openai import AsyncOpenAI
        async def main():
            r = await AsyncOpenAI().chat.completions.create(model="claude-code", messages=[{"role": "user", "content": "x"}])
            print(r.choices[0].message.content)
        asyncio.run(main())
    """) == ["pong from claude-default: x"]


def test_plain_litellm():
    pytest.importorskip("litellm")
    assert output("""
        import litellm
        for model in ("claude-code", "claude-code/sonnet", "codex"):
            print(litellm.completion(model=model, messages=[{"role": "user", "content": "hi"}]).choices[0].message.content)
    """) == ["pong from claude-default: hi", "pong from sonnet: hi", "pong from gpt-default: hi"]


def test_plain_anthropic():
    pytest.importorskip("anthropic")
    assert output("""
        from anthropic import Anthropic
        msg = Anthropic().messages.create(model="claude-code", max_tokens=100, messages=[{"role": "user", "content": "hey"}])
        print(msg.content[0].text)
    """) == ["pong from claude-default: hey"]


def test_plain_langchain_chat_openai():
    pytest.importorskip("langchain_openai")
    assert output("""
        from langchain_openai import ChatOpenAI
        print(ChatOpenAI(model="claude-code").invoke("lc").content)
    """) == ["pong from claude-default: lc"]


def test_other_models_still_reach_the_real_endpoint():
    pytest.importorskip("openai")
    seen = []

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append((body["model"], self.headers["authorization"]))
            payload = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                                  "choices": [{"index": 0, "finish_reason": "stop",
                                               "message": {"role": "assistant", "content": "from upstream"}}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    upstream = HTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    try:
        lines = output(f"""
            from openai import OpenAI
            client = OpenAI(base_url="http://127.0.0.1:{upstream.server_port}/v1", api_key="sk-real")
            for model in ("gpt-4o", "claude-code"):
                print(client.chat.completions.create(model=model, messages=[{{"role": "user", "content": "q"}}]).choices[0].message.content)
        """)
    finally:
        upstream.shutdown()
    assert lines == ["from upstream", "pong from claude-default: q"]
    assert seen == [("gpt-4o", "Bearer sk-real")]


def test_missing_key_still_errors_without_the_hook():
    pytest.importorskip("openai")
    proc = run("from openai import OpenAI; OpenAI()", CLIBRIDGE_AUTOLOAD="0")
    assert proc.returncode != 0 and "OPENAI_API_KEY" in proc.stderr


def test_plain_litellm_with_reasoning_effort():
    pytest.importorskip("litellm")
    assert output("""
        import litellm
        reply = litellm.completion(model="claude-code", reasoning_effort="high",
                                   messages=[{"role": "user", "content": "THINK hard"}])
        print(reply.choices[0].message.content)
        print(reply.choices[0].message.reasoning_content)
    """) == ["pong from claude-default: THINK hard", "Considering: THINK hard"]
