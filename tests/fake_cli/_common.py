"""Shared helpers for the fake CLIs used in tests."""

import json
import os
import re
import sys


def opt(args, name):
    if name in args:
        i = args.index(name)
        return args[i + 1] if i + 1 < len(args) else ""
    return None


def log_call(record):
    path = os.environ.get("FAKE_CLI_LOG")
    if path:
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def sample(schema):
    """A value that satisfies a simple JSON Schema."""
    if not isinstance(schema, dict):
        return None
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return sample(schema["anyOf"][0])
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = kind[0]
    if kind == "object" or "properties" in schema:
        return {k: sample(v) for k, v in (schema.get("properties") or {}).items()}
    if kind == "array":
        return [sample(schema.get("items") or {"type": "string"})]
    return {"string": "text", "integer": 1, "number": 1.5, "boolean": True, "null": None}.get(kind, "text")


def tool_reply(prompt, names, args_as_string, instructions=""):
    """Call the first allowed tool, or answer once a tool result is in the prompt."""
    results = re.findall(r"<tool_result[^>]*>\n(.*?)\n</tool_result>", prompt, re.S)
    if results:
        return {"content": "Final answer using: " + results[-1], "tool_calls": []}
    params = {}
    for line in re.findall(r"^\{.*\}$", instructions + "\n" + prompt, re.M):
        try:
            fn = json.loads(line)
        except ValueError:
            continue
        if fn.get("name") == names[0]:
            params = fn.get("parameters") or {}
    args = {"city": "Paris"} if "city" in (params.get("properties") or {}) else sample(params)
    return {"content": "", "tool_calls": [{"name": names[0], "arguments": json.dumps(args) if args_as_string else args}]}


def reply_text(model, prompt):
    last = prompt.strip().splitlines()[-1] if prompt.strip() else ""
    return f"pong from {model}: {last[:200]}"


def reject_flags(args, message):
    """Act like a CLI version that lacks some flags (FAKE_REJECT_FLAGS="--a,--b")."""
    for flag in filter(None, os.environ.get("FAKE_REJECT_FLAGS", "").split(",")):
        if flag in args:
            sys.stderr.write(message.format(flag=flag) + "\n")
            sys.exit(2)
