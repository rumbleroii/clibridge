"""The CLI must survive output streams that cannot encode the check marks (Windows cp1252)."""

import io
import sys
from argparse import Namespace

from clibridge import cli


class VersionResult:
    stdout = "9.9.9"


def make_args(**overrides):
    values = dict(
        claude_bin="claude",
        codex_bin="codex",
        default_backend=None,
        workdir=None,
        permission_mode=None,
        sandbox=None,
        timeout=None,
        max_concurrency=None,
        api_key=None,
        keep_env=False,
        ping=False,
    )
    values.update(overrides)
    return Namespace(**values)


def doctor_output(monkeypatch, encoding):
    monkeypatch.setattr(cli.shutil, "which", lambda binary: f"C:\\tools\\{binary}.exe")
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: VersionResult())
    stream = io.TextIOWrapper(io.BytesIO(), encoding=encoding)
    monkeypatch.setattr(sys, "stdout", stream)
    assert cli._doctor(make_args()) == 0
    stream.flush()
    return stream.buffer.getvalue().decode(encoding)


def test_doctor_falls_back_to_ascii_on_cp1252(monkeypatch):
    output = doctor_output(monkeypatch, "cp1252")
    assert "OK claude" in output
    assert "OK codex" in output


def test_doctor_keeps_unicode_marks_on_utf8(monkeypatch):
    output = doctor_output(monkeypatch, "utf-8")
    assert "\u2713 claude" in output
    assert "\u2713 codex" in output
