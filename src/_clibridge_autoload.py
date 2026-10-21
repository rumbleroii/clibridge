"""Runs at Python startup (via clibridge.pth), so plain SDK code needs no clibridge import.

It only installs an import hook, which is cheap. When ``openai``, ``anthropic``,
``litellm`` or ``langchain_openai`` is imported, :mod:`clibridge.autopatch` makes
requests for clibridge models (``claude-code``, ``codex``, ...) run on the local
CLIs. Requests for any other model are left alone.

Set ``CLIBRIDGE_AUTOLOAD=0`` to turn this off.
"""

import os
import sys

TARGETS = ("openai", "anthropic", "litellm", "langchain_openai")


def _enabled():
    return os.environ.get("CLIBRIDGE_AUTOLOAD", "1").strip().lower() not in ("0", "false", "no", "off")


def _patch(name):
    if not _enabled():
        return
    try:
        from clibridge import autopatch

        autopatch.patch(name)
    except Exception as exc:  # never break the user's import
        import warnings

        warnings.warn(f"clibridge could not hook into {name}: {exc!r}", RuntimeWarning, stacklevel=2)


class _Loader:
    """Wraps a module's loader to run the patch right after the module finishes importing."""

    def __init__(self, loader, name):
        self.loader = loader
        self.name = name

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        # Hand the real loader back to the module before it runs (importlib.resources etc. use it).
        module.__loader__ = self.loader
        if getattr(module, "__spec__", None) is not None:
            module.__spec__.loader = self.loader
        self.loader.exec_module(module)
        _patch(self.name)

    def __getattr__(self, attr):
        return getattr(self.loader, attr)


class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is not None and hasattr(spec.loader, "exec_module"):
            spec.loader = _Loader(spec.loader, fullname)
        return spec


def install():
    if not _enabled():
        return
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    for name in TARGETS:
        if name in sys.modules:
            _patch(name)


try:
    install()
except Exception:
    pass
