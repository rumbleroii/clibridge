import json
import os
import pathlib

# The SDK tests below check the in-process clients and the HTTP server themselves,
# so keep the startup hook out of the way (test_autoload.py covers it in subprocesses).
os.environ["CLIBRIDGE_AUTOLOAD"] = "0"

import pytest

from clibridge import BridgeConfig

FAKE_CLI = pathlib.Path(__file__).parent / "fake_cli"


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Ignore the developer's own CLIBRIDGE_* variables, and forget rejected flags."""
    from clibridge import backends

    for name in list(os.environ):
        if name.startswith("CLIBRIDGE_") and name != "CLIBRIDGE_AUTOLOAD":
            monkeypatch.delenv(name)
    backends._unsupported.clear()
    yield
    backends._unsupported.clear()


@pytest.fixture
def calls(tmp_path, monkeypatch):
    """Returns a function that reads what the fake CLIs were called with."""
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_CLI_LOG", str(log))

    def read():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    return read


@pytest.fixture
def config(calls):
    return BridgeConfig(claude_bin=str(FAKE_CLI / "claude"), codex_bin=str(FAKE_CLI / "codex"), timeout=20)


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
