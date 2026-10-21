import json
import os
import pathlib
import sys
import tempfile

# The SDK tests below check the in-process clients and the HTTP server themselves,
# so keep the startup hook out of the way (test_autoload.py covers it in subprocesses).
os.environ["CLIBRIDGE_AUTOLOAD"] = "0"

import pytest

from clibridge import BridgeConfig

FAKE_CLI = pathlib.Path(__file__).parent / "fake_cli"
_WRAPPERS = {}
_SCRIPTS = {}


def fake_cli(name: str) -> str:
    """An executable fake CLI: the script itself elsewhere, a .cmd wrapper on Windows."""
    script = FAKE_CLI / name
    if os.name != "nt":
        return str(script)
    if name not in _WRAPPERS:
        folder = pathlib.Path(tempfile.mkdtemp(prefix="clibridge-fake-cli-"))
        wrapper = folder / f"{name}.cmd"
        wrapper.write_text(f'@echo off\n"{sys.executable}" "{script}" %*\n', encoding="utf-8")
        _WRAPPERS[name] = str(wrapper)
        _SCRIPTS[os.path.normcase(str(wrapper))] = str(script)
    return _WRAPPERS[name]


@pytest.fixture(autouse=True)
def run_fake_scripts_directly(monkeypatch):
    """Windows can only execute .cmd wrappers, which mangle multi-line arguments.

    In-process tests bypass the wrapper and start the scripts with the current
    interpreter so argv reaches them unchanged.
    """
    if os.name != "nt":
        yield
        return

    from clibridge import runner as bridge_runner

    subprocess_module = bridge_runner.subprocess
    asyncio_module = bridge_runner.asyncio

    def translated(argv):
        argv = list(argv)
        if argv:
            script = _SCRIPTS.get(os.path.normcase(str(argv[0])))
            if script:
                return [sys.executable, script, *argv[1:]]
        return argv

    real_popen = subprocess_module.Popen
    real_exec = asyncio_module.create_subprocess_exec

    class SubprocessProxy:
        def __getattr__(self, name):
            return getattr(subprocess_module, name)

        def Popen(self, argv, *args, **kwargs):
            return real_popen(translated(argv), *args, **kwargs)

    class AsyncioProxy:
        def __getattr__(self, name):
            return getattr(asyncio_module, name)

        async def create_subprocess_exec(self, *argv, **kwargs):
            return await real_exec(*translated(argv), **kwargs)

    monkeypatch.setattr(bridge_runner, "subprocess", SubprocessProxy())
    monkeypatch.setattr(bridge_runner, "asyncio", AsyncioProxy())
    yield


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
    return BridgeConfig(claude_bin=fake_cli("claude"), codex_bin=fake_cli("codex"), timeout=20)


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
