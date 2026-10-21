"""Turn a normalized request into a `claude -p` / `codex exec` invocation and parse its output."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import BridgeConfig
from .errors import BridgeError

DEFAULT_SYSTEM = "You are a helpful assistant."
CODEX_CHAT_NOTE = (
    "Answer directly from your own knowledge. Do not run shell commands, read or change "
    "files, or browse the web."
)
# Thinking levels a request may ask for, mapped to what each CLI accepts.
# For Claude, "none" turns thinking off instead (CLAUDE_CODE_DISABLE_THINKING).
CLAUDE_EFFORT = {"minimal": "low"}
CODEX_EFFORT = {"minimal": "low", "max": "xhigh"}
LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]", "clibridge.local")


@dataclass
class RunSpec:
    backend: str
    prompt: str
    system: Optional[str] = None
    model: Optional[str] = None
    images: List[Tuple[str, bytes]] = field(default_factory=list)  # (media type, data)
    json_schema: Optional[Dict[str, Any]] = None
    agent: bool = False
    cwd: Optional[str] = None
    effort: Optional[str] = None
    max_turns: Optional[int] = None
    stream: bool = False


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    """Part of the model's thinking (a summary, as the CLIs show it)."""

    text: str


@dataclass
class ToolActivity:
    """The CLI agent used one of its own tools (agent mode only)."""

    name: str
    detail: str = ""


@dataclass
class Final:
    text: str
    structured: Any = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning: str = ""
    reasoning_tokens: int = 0
    session_id: Optional[str] = None
    cost_usd: Optional[float] = None
    num_turns: Optional[int] = None
    truncated: bool = False


@dataclass
class Invocation:
    argv: List[str]
    stdin: bytes
    cwd: str
    env: Dict[str, str]
    cleanup: List[str] = field(default_factory=list)


def resolve_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise BridgeError(
            f"Could not find the `{name}` CLI. Install it and log in, or point "
            "CLIBRIDGE_CLAUDE_BIN / CLIBRIDGE_CODEX_BIN at it.",
            status=500,
        )
    return path


class UnsupportedFlag(BridgeError):
    """The installed CLI is too old (or too new) for one of our optional flags."""

    def __init__(self, binary: str, flag: str):
        super().__init__(f"{os.path.basename(binary)} does not support {flag}")
        self.binary = binary
        self.flag = flag


# Optional flags the installed CLIs rejected; the runner retries without them.
_unsupported: Set[Tuple[str, str]] = set()
_UNKNOWN_FLAG = re.compile(r"(?:unknown option|unexpected argument) '(--[\w-]+)'")


def mark_unsupported(error: UnsupportedFlag) -> bool:
    """Remember a rejected flag. False if it was already known (so retrying won't help)."""
    key = (error.binary, error.flag)
    if key in _unsupported:
        return False
    _unsupported.add(key)
    return True


def _optional(argv: List[str], binary: str, flag: str, *values: str) -> None:
    """Add a flag the CLI might not know; if it rejects it once, it is skipped from then on."""
    if (binary, flag) not in _unsupported:
        argv += [flag, *values]


def _check_unknown_flag(binary: str, stderr: str, optional: Tuple[str, ...]) -> None:
    match = _UNKNOWN_FLAG.search(stderr)
    if match and match.group(1) in optional:
        raise UnsupportedFlag(binary, match.group(1))


def chat_dir() -> str:
    """An empty directory to run chat-mode calls in, away from any project files."""
    path = os.path.join(tempfile.gettempdir(), "clibridge-chat")
    os.makedirs(path, exist_ok=True)
    return path


def child_env(config: BridgeConfig) -> Dict[str, str]:
    env = dict(os.environ)
    if config.scrub_env:
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("OPENAI_API_KEY", None)
        for name in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL"):
            if any(host in env.get(name, "") for host in LOCAL_HOSTS):
                env.pop(name)
    return env


def _write_temp(data: bytes, suffix: str, cleanup: List[str]) -> str:
    fd, path = tempfile.mkstemp(prefix="clibridge-", suffix=suffix)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    cleanup.append(path)
    return path


def _summarize(value: Any, limit: int = 120) -> str:
    if isinstance(value, dict):
        for key in ("command", "file_path", "path", "pattern", "url", "query", "description"):
            if isinstance(value.get(key), str):
                value = value[key]
                break
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tail(text: str, limit: int = 1500) -> str:
    text = text.strip()
    return text if len(text) <= limit else "…" + text[-limit:]


class ClaudeBackend:
    name = "claude"

    def __init__(self, config: BridgeConfig):
        self.config = config

    OPTIONAL_FLAGS = ("--safe-mode", "--json-schema", "--effort", "--thinking-display")

    def build(self, spec: RunSpec) -> Invocation:
        binary = resolve_binary(self.config.claude_bin)
        argv = [binary, "-p", "--output-format", "stream-json", "--verbose", "--no-session-persistence"]
        if spec.stream:
            argv.append("--include-partial-messages")
        if spec.model:
            argv += ["--model", spec.model]
        if spec.agent:
            cwd = spec.cwd or self.config.workdir or os.getcwd()
            if spec.system:
                argv += ["--append-system-prompt", spec.system]
            if self.config.permission_mode:
                argv += ["--permission-mode", self.config.permission_mode]
            if spec.max_turns:
                argv += ["--max-turns", str(spec.max_turns)]
        else:
            # Plain LLM: no tools, no MCP servers, our system prompt instead of Claude Code's.
            cwd = chat_dir()
            argv += ["--tools", "", "--strict-mcp-config", "--system-prompt", spec.system or DEFAULT_SYSTEM]
            _optional(argv, binary, "--safe-mode")
        if spec.json_schema is not None:
            _optional(argv, binary, "--json-schema", json.dumps(spec.json_schema))
        env = child_env(self.config)
        if spec.effort == "none":
            env["CLAUDE_CODE_DISABLE_THINKING"] = "1"
        elif spec.effort:
            _optional(argv, binary, "--effort", CLAUDE_EFFORT.get(spec.effort, spec.effort))
        # Headless Claude Code sends thinking as empty text unless asked for summaries.
        _optional(argv, binary, "--thinking-display", "summarized")
        argv += self.config.claude_args

        if spec.images:
            argv += ["--input-format", "stream-json"]
            content: List[Dict[str, Any]] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(data).decode("ascii"),
                    },
                }
                for media_type, data in spec.images
            ]
            content.append({"type": "text", "text": spec.prompt})
            message = {
                "type": "user",
                "message": {"role": "user", "content": content},
                "parent_tool_use_id": None,
                "session_id": "default",
            }
            stdin = json.dumps(message) + "\n"
        else:
            stdin = spec.prompt
        return Invocation(argv, stdin.encode("utf-8"), cwd, env)

    def parser(self) -> "ClaudeParser":
        return ClaudeParser(resolve_binary(self.config.claude_bin), self.OPTIONAL_FLAGS)


class ClaudeParser:
    """Parses `claude -p --output-format stream-json --verbose` lines."""

    def __init__(self, binary: str = "claude", optional: Tuple[str, ...] = ()) -> None:
        self.binary = binary
        self.optional = optional
        self.final: Optional[Final] = None
        self.thinking: List[str] = []
        self._emitted = False
        self._need_break = False

    def feed(self, obj: Dict[str, Any]) -> List[Any]:
        kind = obj.get("type")
        out: List[Any] = []
        if obj.get("parent_tool_use_id"):
            return out  # subagent traffic
        if kind == "stream_event":
            event = obj.get("event") or {}
            if event.get("type") == "message_start":
                self._need_break = self._emitted
            elif event.get("type") == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "thinking_delta" and delta.get("thinking"):
                    out.append(ThinkingDelta(delta["thinking"]))
                elif delta.get("type") == "text_delta" and delta.get("text"):
                    if self._need_break:
                        out.append(TextDelta("\n\n"))
                        self._need_break = False
                    self._emitted = True
                    out.append(TextDelta(delta["text"]))
        elif kind == "assistant":
            for block in (obj.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "thinking" and block.get("thinking"):
                    self.thinking.append(block["thinking"].strip())
                elif block.get("type") == "tool_use" and block.get("name") != "StructuredOutput":
                    out.append(ToolActivity(block.get("name", "tool"), _summarize(block.get("input"))))
        elif kind == "result":
            self.final = self._final(obj)
        return out

    def _final(self, obj: Dict[str, Any]) -> Final:
        subtype = obj.get("subtype") or ""
        truncated = subtype == "error_max_turns"
        if not truncated and (obj.get("is_error") or subtype.startswith("error")):
            message = obj.get("result") or "; ".join(map(str, obj.get("errors") or [])) or subtype
            status = 429 if obj.get("api_error_status") == 429 or "limit" in str(message).lower() else 424
            raise BridgeError(f"claude: {message}", status=status)
        usage = obj.get("usage") or {}
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_write = int(usage.get("cache_creation_input_tokens") or 0)
        return Final(
            text=obj.get("result") or "",
            structured=obj.get("structured_output"),
            input_tokens=int(usage.get("input_tokens") or 0) + cache_read + cache_write,
            output_tokens=int(usage.get("output_tokens") or 0),
            cached_tokens=cache_read,
            reasoning="\n\n".join(t for t in self.thinking if t),
            reasoning_tokens=int((usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0),
            session_id=obj.get("session_id"),
            cost_usd=obj.get("total_cost_usd"),
            num_turns=obj.get("num_turns"),
            truncated=truncated,
        )

    def finish(self, returncode: int, stderr: str) -> List[Any]:
        if self.final is None:
            _check_unknown_flag(self.binary, stderr, self.optional)
            raise BridgeError(f"claude exited with code {returncode} without a result. {_tail(stderr)}")
        return [self.final]


class CodexBackend:
    name = "codex"

    def __init__(self, config: BridgeConfig):
        self.config = config

    OPTIONAL_FLAGS = ("--ephemeral",)

    def build(self, spec: RunSpec) -> Invocation:
        binary = resolve_binary(self.config.codex_bin)
        argv = [binary, "exec", "--json", "--skip-git-repo-check", "--color", "never"]
        _optional(argv, binary, "--ephemeral")
        if spec.model:
            argv += ["--model", spec.model]
        if spec.effort:
            effort = CODEX_EFFORT.get(spec.effort, spec.effort)
            argv += ["--config", f"model_reasoning_effort={json.dumps(effort)}"]
        if spec.agent:
            cwd = spec.cwd or self.config.workdir or os.getcwd()
            argv += ["--sandbox", self.config.sandbox]
        else:
            cwd = chat_dir()
            argv += ["--sandbox", "read-only"]
        cleanup: List[str] = []
        if spec.json_schema is not None:
            schema = json.dumps(spec.json_schema).encode("utf-8")
            argv += ["--output-schema", _write_temp(schema, ".json", cleanup)]
        for media_type, data in spec.images:
            suffix = mimetypes.guess_extension(media_type) or ".png"
            argv += ["--image", _write_temp(data, suffix, cleanup)]
        argv += self.config.codex_args
        # No PROMPT argument: `codex exec` then reads the prompt from stdin.

        system = spec.system
        if not spec.agent:
            system = CODEX_CHAT_NOTE + ("\n\n" + system if system else "")
        prompt = f"<instructions>\n{system}\n</instructions>\n\n{spec.prompt}" if system else spec.prompt
        return Invocation(argv, prompt.encode("utf-8"), cwd, child_env(self.config), cleanup)

    def parser(self) -> "CodexParser":
        return CodexParser(resolve_binary(self.config.codex_bin), self.OPTIONAL_FLAGS)


class CodexParser:
    """Parses `codex exec --json` JSONL events."""

    def __init__(self, binary: str = "codex", optional: Tuple[str, ...] = ()) -> None:
        self.binary = binary
        self.optional = optional
        self.messages: List[str] = []
        self.thinking: List[str] = []
        self.usage: Dict[str, Any] = {}
        self.thread_id: Optional[str] = None
        self.error: Optional[str] = None
        self.completed = False

    def feed(self, obj: Dict[str, Any]) -> List[Any]:
        kind = obj.get("type")
        if kind == "thread.started":
            self.thread_id = obj.get("thread_id")
        elif kind == "item.completed":
            item = obj.get("item") or {}
            item_type = item.get("type")
            if item_type == "agent_message":
                text = item.get("text") or ""
                delta = ("\n\n" if self.messages else "") + text
                self.messages.append(text)
                return [TextDelta(delta)] if text else []
            if item_type == "reasoning" and item.get("text"):
                delta = ("\n\n" if self.thinking else "") + item["text"]
                self.thinking.append(item["text"])
                return [ThinkingDelta(delta)]
            if item_type == "command_execution":
                return [ToolActivity("shell", _summarize(item.get("command", "")))]
            if item_type == "file_change":
                paths = ", ".join(c.get("path", "") for c in item.get("changes") or [])
                return [ToolActivity("edit", paths)]
            if item_type == "mcp_tool_call":
                return [ToolActivity(f"{item.get('server', 'mcp')}.{item.get('tool', '')}")]
            if item_type == "web_search":
                return [ToolActivity("web_search", _summarize(item.get("query", "")))]
        elif kind == "turn.completed":
            self.completed = True
            self.usage = obj.get("usage") or {}
        elif kind == "turn.failed":
            self.completed = False
            self.error = (obj.get("error") or {}).get("message") or "turn failed"
        elif kind == "error":
            self.error = obj.get("message") or "error"
        return []

    def finish(self, returncode: int, stderr: str) -> List[Any]:
        if not self.completed:
            _check_unknown_flag(self.binary, stderr, self.optional)
            detail = self.error or f"exited with code {returncode}. {_tail(stderr)}"
            status = 429 if "limit" in detail.lower() else 424
            raise BridgeError(f"codex: {detail}", status=status)
        usage = self.usage
        return [
            Final(
                text=self.messages[-1] if self.messages else "",
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                cached_tokens=int(usage.get("cached_input_tokens") or 0),
                reasoning="\n\n".join(self.thinking),
                reasoning_tokens=int(usage.get("reasoning_output_tokens") or 0),
                session_id=self.thread_id,
            )
        ]


def get_backend(name: str, config: BridgeConfig):
    if name == "claude":
        return ClaudeBackend(config)
    if name == "codex":
        return CodexBackend(config)
    raise BridgeError(f"Unknown backend {name!r}", status=400)
