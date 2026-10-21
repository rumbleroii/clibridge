"""Command line: ``clibridge serve | ask | doctor``."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from typing import List, Optional

from .config import BridgeConfig
from .core import ChatResult, Thinking, iter_chat
from .errors import BridgeError


def _config_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("backend options")
    group.add_argument("--claude-bin", help="path to the claude CLI (default: claude)")
    group.add_argument("--codex-bin", help="path to the codex CLI (default: codex)")
    group.add_argument("--default-backend", choices=["claude", "codex"],
                       help="backend for model names that don't pick one (default: claude)")
    group.add_argument("--workdir", help="working directory for agent mode (default: current directory)")
    group.add_argument("--permission-mode", help="claude --permission-mode in agent mode, e.g. acceptEdits")
    group.add_argument("--sandbox", choices=["read-only", "workspace-write", "danger-full-access"],
                       help="codex --sandbox in agent mode (default: read-only)")
    group.add_argument("--timeout", type=float, help="seconds before a CLI call is killed (default: 600)")
    group.add_argument("--max-concurrency", type=int, help="CLI processes allowed at once (default: 4)")
    group.add_argument("--keep-env", action="store_true",
                       help="pass API key env vars through to the CLIs instead of removing them")


def _config(args: argparse.Namespace) -> BridgeConfig:
    names = ("claude_bin", "codex_bin", "default_backend", "workdir", "permission_mode", "sandbox",
             "timeout", "max_concurrency", "api_key")
    given = {name: getattr(args, name, None) for name in names}
    return BridgeConfig(**{k: v for k, v in given.items() if v is not None}, scrub_env=not args.keep_env)


def _logging(verbosity: int, default: int) -> None:
    level = default - 10 * verbosity
    logging.basicConfig(level=max(level, logging.DEBUG), format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")


def _serve(args: argparse.Namespace) -> int:
    from .server import serve

    _logging(args.verbose, logging.INFO)
    serve(args.host, args.port, _config(args))
    return 0


def _ask(args: argparse.Namespace) -> int:
    _logging(args.verbose, logging.WARNING)
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    messages = [{"role": "system", "content": args.system}] if args.system else []
    messages.append({"role": "user", "content": prompt})
    request = {"model": args.model, "messages": messages, "stream": True}
    if args.agent:
        request["agent"] = True
    if args.effort:
        request["reasoning_effort"] = args.effort
    try:
        for item in iter_chat(request, _config(args)):
            if isinstance(item, str):
                sys.stdout.write(item)
                sys.stdout.flush()
            elif isinstance(item, Thinking):
                if args.show_thinking:
                    sys.stderr.write(f"\033[2m{item.text}\033[0m")  # dimmed
                    sys.stderr.flush()
            elif isinstance(item, ChatResult):
                if args.show_thinking and item.reasoning and not item.reasoning_streamed:
                    sys.stderr.write(f"\033[2m{item.reasoning}\033[0m\n")
                if not item.streamed and item.content:
                    sys.stdout.write(item.content)
    except BridgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write("\n")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    config = _config(args)
    ok = True
    for backend, binary in (("claude", config.claude_bin), ("codex", config.codex_bin)):
        path = shutil.which(binary)
        if not path:
            print(f"✗ {backend}: `{binary}` not found on PATH")
            ok = False
            continue
        try:
            version = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            version = f"(could not run: {exc})"
        print(f"✓ {backend}: {path} {version}")
        if args.ping:
            request = {"model": backend, "messages": [{"role": "user", "content": "Reply with the word: ok"}],
                       "reasoning_effort": "low"}
            try:
                reply = None
                for item in iter_chat(request, config):
                    reply = item
                print(f"  ✓ test call answered: {reply.content.strip()[:60]!r}")
            except BridgeError as exc:
                print(f"  ✗ test call failed: {exc}")
                ok = False
    if any(type(f).__module__ == "_clibridge_autoload" for f in sys.meta_path):
        print('✓ auto mode: on. Plain openai / litellm / anthropic / langchain code can use model="claude-code"')
    else:
        print("✗ auto mode: off (CLIBRIDGE_AUTOLOAD=0, or clibridge isn't pip-installed in this Python)")
    if not args.ping:
        print("Run `clibridge doctor --ping` to make a tiny test call through each CLI.")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="clibridge",
        description="Use your Claude Code / Codex subscriptions from any OpenAI- or Anthropic-compatible SDK.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the OpenAI/Anthropic-compatible HTTP server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--api-key", help="require this bearer token from clients (default: accept any)")
    serve.add_argument("-v", "--verbose", action="count", default=0, help="-v: log CLI commands")
    _config_args(serve)

    ask = sub.add_parser("ask", help="send one prompt and stream the reply")
    ask.add_argument("prompt", nargs="?", help="the prompt (read from stdin if omitted)")
    ask.add_argument("-m", "--model", default="claude-code", help="e.g. claude-code, claude-code/opus, codex")
    ask.add_argument("-e", "--effort", choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
                     help="thinking level")
    ask.add_argument("-t", "--show-thinking", action="store_true", help="print the model's thinking (dimmed)")
    ask.add_argument("-s", "--system", help="system prompt")
    ask.add_argument("--agent", action="store_true", help="let the CLI use its own tools in --workdir")
    ask.add_argument("-v", "--verbose", action="count", default=0, help="-v: show agent tool use")
    _config_args(ask)

    doctor = sub.add_parser("doctor", help="check that the CLIs are installed (and logged in, with --ping)")
    doctor.add_argument("--ping", action="store_true", help="make a tiny test call through each CLI")
    _config_args(doctor)

    args = parser.parse_args(argv)
    handler = {"serve": _serve, "ask": _ask, "doctor": _doctor}[args.command]
    sys.exit(handler(args))
