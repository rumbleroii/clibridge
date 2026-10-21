import ast
import json
import operator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

WORKDIR = Path(__file__).resolve().parent / "workdir"
MAX_FILE_BYTES = 200_000

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression using numbers, + - * / // % ** and parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The expression to evaluate, e.g. (17*23)+4"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time, optionally for an IANA timezone such as Asia/Tokyo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone name. Omit for the machine's local time."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and folders in the demo workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside the workspace. Defaults to the root."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the demo workspace. Paths outside the workspace are refused.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace root."}
                },
                "required": ["path"],
            },
        },
    },
]

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent is too large for this demo")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("only numbers, parentheses and basic arithmetic are supported")


def _workspace_path(relative: str) -> Path:
    base = WORKDIR.resolve()
    path = (base / relative).resolve()
    if path != base and base not in path.parents:
        raise ValueError("path must stay inside the workspace")
    return path


def _calculate(arguments):
    expression = str(arguments.get("expression", "")).strip()
    if not expression:
        raise ValueError("expression is required")
    if len(expression) > 200:
        raise ValueError("expression is too long")
    tree = ast.parse(expression, mode="eval")
    result = _evaluate(tree)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return {"expression": expression, "result": result}


def _current_time(arguments):
    timezone = str(arguments.get("timezone") or "").strip()
    if not timezone:
        now = datetime.now().astimezone()
        return {"iso": now.isoformat(), "timezone": now.tzname() or "local"}
    try:
        zone = ZoneInfo(timezone)
    except Exception as exc:
        raise ValueError(f"unknown timezone {timezone!r}: {exc}") from exc
    now = datetime.now(zone)
    return {"iso": now.isoformat(), "timezone": timezone}


def _list_files(arguments):
    relative = str(arguments.get("path") or ".").strip()
    path = _workspace_path(relative)
    if not path.exists():
        raise ValueError(f"no such path: {relative}")
    if path.is_file():
        return {"path": relative, "entries": [{"name": path.name, "type": "file", "bytes": path.stat().st_size}]}
    entries = []
    for child in sorted(path.iterdir())[:200]:
        entries.append(
            {
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
                "bytes": 0 if child.is_dir() else child.stat().st_size,
            }
        )
    return {"path": relative, "entries": entries}


def _read_file(arguments):
    relative = str(arguments.get("path", "")).strip()
    if not relative:
        raise ValueError("path is required")
    path = _workspace_path(relative)
    if not path.is_file():
        raise ValueError(f"not a file: {relative}")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"file is too large ({size} bytes)")
    return {"path": relative, "content": path.read_text(encoding="utf-8", errors="replace")}


_EXECUTORS = {
    "calculate": _calculate,
    "get_current_time": _current_time,
    "list_files": _list_files,
    "read_file": _read_file,
}


def execute_tool(name, arguments_json):
    try:
        arguments = json.loads(arguments_json) if arguments_json else {}
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a JSON object")
        executor = _EXECUTORS.get(name)
        if executor is None:
            raise ValueError(f"unknown tool: {name}")
        return json.dumps({"ok": True, **executor(arguments)}, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
