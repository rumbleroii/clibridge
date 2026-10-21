"""Run a CLI invocation as a subprocess and stream parsed events (sync and async)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import threading
from collections import deque
from typing import Any, AsyncIterator, Deque, Dict, Iterator, List, Optional

from .backends import RunSpec, UnsupportedFlag, get_backend, mark_unsupported
from .config import BridgeConfig
from .errors import BridgeError

log = logging.getLogger("clibridge")

STREAM_LIMIT = 64 * 1024 * 1024  # stream-json lines can be large


def _decode(raw: bytes) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except ValueError:
        log.debug("ignoring non-JSON output: %r", raw[:200])
        return None
    return obj if isinstance(obj, dict) else None


def _cleanup(paths: List[str]) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


def _describe(argv: List[str]) -> str:
    return " ".join(a if len(a) < 80 else a[:77] + "..." for a in argv)


def _write_stdin(pipe, data: bytes) -> None:
    try:
        pipe.write(data)
        pipe.close()
    except (BrokenPipeError, OSError):
        pass


def _drain(pipe, sink: Deque[bytes]) -> None:
    for line in pipe:
        sink.append(line)


def _timeout_error(config: BridgeConfig) -> BridgeError:
    return BridgeError(f"CLI call timed out after {config.timeout:.0f}s", status=504)


def run(spec: RunSpec, config: BridgeConfig) -> Iterator[Any]:
    """Run the CLI for ``spec`` and yield TextDelta / ThinkingDelta / ToolActivity / Final events.

    If the CLI rejects one of our optional flags, it is retried without that flag.
    """
    while True:
        started = False
        events = _run_once(spec, config)
        try:
            for event in events:
                started = True
                yield event
            return
        except UnsupportedFlag as exc:
            if started or not mark_unsupported(exc):
                raise
            log.info("%s; retrying without it", exc)
        finally:
            events.close()  # stops the CLI if our consumer stopped early


def _run_once(spec: RunSpec, config: BridgeConfig) -> Iterator[Any]:
    backend = get_backend(spec.backend, config)
    inv = backend.build(spec)
    parser = backend.parser()
    log.debug("exec: %s (cwd=%s)", _describe(inv.argv), inv.cwd)
    config.slots.acquire()
    proc: Optional[subprocess.Popen] = None
    timer: Optional[threading.Timer] = None
    timed_out = threading.Event()
    try:
        try:
            proc = subprocess.Popen(
                inv.argv,
                cwd=inv.cwd,
                env=inv.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise BridgeError(f"Could not start {inv.argv[0]}: {exc}", status=500) from exc

        def expire() -> None:
            timed_out.set()
            proc.kill()

        timer = threading.Timer(config.timeout, expire)
        timer.daemon = True
        timer.start()
        stderr: Deque[bytes] = deque(maxlen=100)
        threading.Thread(target=_write_stdin, args=(proc.stdin, inv.stdin), daemon=True).start()
        threading.Thread(target=_drain, args=(proc.stderr, stderr), daemon=True).start()

        for raw in proc.stdout:
            obj = _decode(raw)
            if obj is not None:
                yield from parser.feed(obj)
        returncode = proc.wait()
        if timed_out.is_set():
            raise _timeout_error(config)
        yield from parser.finish(returncode, b"".join(stderr).decode("utf-8", "replace"))
    finally:
        if timer is not None:
            timer.cancel()
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        config.slots.release()
        _cleanup(inv.cleanup)


async def arun(spec: RunSpec, config: BridgeConfig) -> AsyncIterator[Any]:
    """Async version of :func:`run`."""
    while True:
        started = False
        events = _arun_once(spec, config)
        try:
            async for event in events:
                started = True
                yield event
            return
        except UnsupportedFlag as exc:
            if started or not mark_unsupported(exc):
                raise
            log.info("%s; retrying without it", exc)
        finally:
            await events.aclose()  # stops the CLI if our consumer stopped early


async def _arun_once(spec: RunSpec, config: BridgeConfig) -> AsyncIterator[Any]:
    backend = get_backend(spec.backend, config)
    inv = backend.build(spec)
    parser = backend.parser()
    log.debug("exec: %s (cwd=%s)", _describe(inv.argv), inv.cwd)
    while not config.slots.acquire(blocking=False):
        await asyncio.sleep(0.05)
    proc: Optional[asyncio.subprocess.Process] = None
    tasks: List[asyncio.Future] = []
    timer: Optional[asyncio.TimerHandle] = None
    timed_out = False
    try:
        try:
            proc = await asyncio.create_subprocess_exec(
                *inv.argv,
                cwd=inv.cwd,
                env=inv.env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=STREAM_LIMIT,
            )
        except OSError as exc:
            raise BridgeError(f"Could not start {inv.argv[0]}: {exc}", status=500) from exc

        def expire() -> None:
            nonlocal timed_out
            timed_out = True
            try:
                proc.kill()
            except ProcessLookupError:
                pass

        timer = asyncio.get_running_loop().call_later(config.timeout, expire)
        stderr: Deque[bytes] = deque(maxlen=100)

        async def write_stdin() -> None:
            try:
                proc.stdin.write(inv.stdin)
                await proc.stdin.drain()
                proc.stdin.close()
            except (BrokenPipeError, ConnectionResetError):
                pass

        async def drain() -> None:
            async for line in proc.stderr:
                stderr.append(line)

        tasks = [asyncio.ensure_future(write_stdin()), asyncio.ensure_future(drain())]
        async for raw in proc.stdout:
            obj = _decode(raw)
            if obj is not None:
                for event in parser.feed(obj):
                    yield event
        returncode = await proc.wait()
        await asyncio.gather(*tasks, return_exceptions=True)
        if timed_out:
            raise _timeout_error(config)
        for event in parser.finish(returncode, b"".join(stderr).decode("utf-8", "replace")):
            yield event
    finally:
        if timer is not None:
            timer.cancel()
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
        for task in tasks:
            task.cancel()
        config.slots.release()
        _cleanup(inv.cleanup)
