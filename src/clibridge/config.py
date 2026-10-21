from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import List, Optional

CLAUDE_ALIASES = ("fable", "opus", "sonnet", "haiku")
_CLAUDE_BACKENDS = ("claude", "claude-code", "claude-cli")
_CODEX_BACKENDS = ("codex", "codex-cli")
_VENDOR_PREFIXES = ("openai", "anthropic")

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value else default


@dataclass
class BridgeConfig:
    """Settings shared by the server, the in-process clients and the integrations.

    Every field can also be set through a ``CLIBRIDGE_*`` environment variable.
    """

    claude_bin: str = field(default_factory=lambda: _env("CLIBRIDGE_CLAUDE_BIN", "claude"))
    codex_bin: str = field(default_factory=lambda: _env("CLIBRIDGE_CODEX_BIN", "codex"))
    # Backend used for model names that don't say which CLI they want.
    default_backend: str = field(default_factory=lambda: _env("CLIBRIDGE_DEFAULT_BACKEND", "claude"))
    # Working directory for agent mode (None = the current directory).
    workdir: Optional[str] = field(default_factory=lambda: _env("CLIBRIDGE_WORKDIR"))
    # Claude Code --permission-mode for agent mode (None = the CLI's default, which
    # denies edits and shell commands in headless mode).
    permission_mode: Optional[str] = field(default_factory=lambda: _env("CLIBRIDGE_PERMISSION_MODE"))
    # Codex --sandbox for agent mode.
    sandbox: str = field(default_factory=lambda: _env("CLIBRIDGE_SANDBOX", "read-only"))
    timeout: float = field(default_factory=lambda: float(_env("CLIBRIDGE_TIMEOUT", "600")))
    max_concurrency: int = field(default_factory=lambda: int(_env("CLIBRIDGE_MAX_CONCURRENCY", "4")))
    # Extra arguments appended to every `claude` / `codex exec` invocation.
    claude_args: List[str] = field(default_factory=list)
    codex_args: List[str] = field(default_factory=list)
    # Drop API keys (and base URLs that point back at a local server) from the
    # CLI's environment, so the CLIs use your subscription login.
    scrub_env: bool = True
    # Bearer token the HTTP server requires (None = accept anything).
    api_key: Optional[str] = field(default_factory=lambda: _env("CLIBRIDGE_API_KEY"))

    slots: threading.BoundedSemaphore = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.slots = threading.BoundedSemaphore(max(1, self.max_concurrency))


def is_bridge_model(name: Optional[str]) -> bool:
    """True for names that explicitly ask for a CLI (``claude-code``, ``codex/gpt-5``, ``*-agent``...).

    Used by the startup hook to decide which SDK requests to take over, so
    names of real API models (``gpt-4o``, ``claude-sonnet-4-5``) are never matched.
    """
    if not isinstance(name, str) or not name:
        return False
    base = name.partition("/")[0].lower()
    if base.endswith("-agent"):
        base = base[: -len("-agent")]
    return base in _CLAUDE_BACKENDS or base in _CODEX_BACKENDS


@dataclass(frozen=True)
class Route:
    backend: str  # "claude" or "codex"
    model: Optional[str]  # passed to the CLI; None means the CLI's default
    agent: bool = False


def resolve_model(name: Optional[str], default_backend: str = "claude") -> Route:
    """Map the ``model`` field of a request to a CLI backend and model.

    * ``claude-code``, ``claude-code/opus``, ``codex``, ``codex/gpt-5-codex``: explicit.
    * ``claude-code-agent[/model]``, ``codex-agent[/model]``: agent mode.
    * ``sonnet``, ``claude-sonnet-4-5``, ``anthropic/claude-3-opus``: Claude, by family alias.
    * ``gpt-5*`` and ``*codex*``: Codex with that model; other ``gpt-*``/``o*``: Codex default.
    * Anything else goes to ``default_backend`` with its default model.
    """
    name = (name or "").strip()
    head, _, tail = name.partition("/")
    base = head.lower()
    tail = tail.strip() or None
    if tail and tail.lower() == "default":
        tail = None
    agent = base.endswith("-agent")
    if agent:
        base = base[: -len("-agent")]
    if base in _CLAUDE_BACKENDS:
        return Route("claude", tail, agent)
    if base in _CODEX_BACKENDS:
        return Route("codex", tail, agent)
    if base in _VENDOR_PREFIXES and tail:
        return resolve_model(tail, default_backend)

    lower = name.lower()
    if lower in CLAUDE_ALIASES:
        return Route("claude", lower)
    if lower.startswith("claude") or any(alias in lower for alias in CLAUDE_ALIASES):
        return Route("claude", next((a for a in CLAUDE_ALIASES if a in lower), None))
    if lower.startswith("gpt-5") or "codex" in lower:
        return Route("codex", name)
    if lower.startswith(("gpt", "chatgpt", "o1", "o3", "o4")):
        return Route("codex", None)
    return Route(default_backend if default_backend in ("claude", "codex") else "claude", None)
