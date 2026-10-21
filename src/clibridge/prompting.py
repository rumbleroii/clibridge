"""Render OpenAI-style chat messages into a single CLI prompt, and parse replies back."""

from __future__ import annotations

import base64
import json
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .errors import BridgeError

Image = Tuple[str, bytes]

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_DATA_URL = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.DOTALL)


def _load_image(part: Dict[str, Any]) -> Image:
    ref = part.get("image_url")
    url = ref.get("url") if isinstance(ref, dict) else ref
    if not isinstance(url, str) or not url:
        raise BridgeError("image part without a URL", status=400)
    match = _DATA_URL.match(url)
    if match:
        media_type = match.group(1) or "image/png"
        payload = match.group(3)
        data = base64.b64decode(payload) if match.group(2) else payload.encode("utf-8")
        return media_type, data
    if url.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                media_type = resp.headers.get_content_type() or "image/png"
                return media_type, resp.read()
        except OSError as exc:
            raise BridgeError(f"could not download image {url}: {exc}", status=400) from exc
    raise BridgeError("images must be data: or http(s) URLs", status=400)


def content_text(content: Any, images: Optional[List[Image]] = None) -> str:
    """Flatten message content to text. Images are collected into ``images`` and
    replaced by ``[image N]`` markers (or dropped to ``[image]`` if ``images`` is None)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    parts: List[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text"):
            parts.append(part.get("text") or "")
        elif kind == "refusal":
            parts.append(part.get("refusal") or "")
        elif kind in ("image_url", "input_image"):
            if images is None:
                parts.append("[image]")
            else:
                images.append(_load_image(part))
                parts.append(f"[image {len(images)}]")
    return "\n".join(p for p in parts if p)


def build_prompt(
    messages: List[Dict[str, Any]], tool_mode: bool = False
) -> Tuple[str, Optional[str], List[Image]]:
    """Return ``(prompt, system, images)`` for a list of chat messages.

    A single user message is passed through as-is. Longer conversations are
    rendered as a transcript the model is asked to continue.
    """
    system_parts: List[str] = []
    turns: List[Tuple[str, str, Dict[str, str]]] = []
    images: List[Image] = []
    call_names: Dict[str, str] = {}

    for message in messages:
        role = message.get("role")
        if role in ("system", "developer"):
            system_parts.append(content_text(message.get("content")))
        elif role == "user":
            turns.append(("user", content_text(message.get("content"), images), {}))
        elif role == "assistant":
            lines = [content_text(message.get("content"))]
            calls = list(message.get("tool_calls") or [])
            if message.get("function_call"):
                calls.append({"id": "", "function": message["function_call"]})
            for call in calls:
                fn = call.get("function") or {}
                call_names[call.get("id") or ""] = fn.get("name") or ""
                lines.append(
                    f'<tool_call id="{call.get("id") or ""}" name="{fn.get("name") or ""}">'
                    f'{fn.get("arguments") or "{}"}</tool_call>'
                )
            turns.append(("assistant", "\n".join(line for line in lines if line), {}))
        elif role in ("tool", "function"):
            call_id = message.get("tool_call_id") or ""
            name = message.get("name") or call_names.get(call_id, "")
            turns.append(("tool_result", content_text(message.get("content")), {"id": call_id, "name": name}))

    system = "\n\n".join(p for p in system_parts if p) or None

    if not turns:
        prompt = "Follow your instructions."
    elif len(turns) == 1 and turns[0][0] == "user":
        prompt = turns[0][1]
    else:
        lines = ["The conversation so far, oldest message first:", "", "<conversation>"]
        for role, text, attrs in turns:
            if role == "tool_result":
                lines.append(f'<tool_result id="{attrs["id"]}" name="{attrs["name"]}">\n{text}\n</tool_result>')
            else:
                lines.append(f"<{role}>\n{text}\n</{role}>")
        lines += ["</conversation>", ""]
        if tool_mode:
            lines.append("Write the assistant's next reply, in the required JSON format.")
        elif turns[-1][0] == "assistant":
            lines.append("Continue the assistant's last message exactly where it stops. Output only the continuation.")
        else:
            lines.append("Write the assistant's next reply. Output only the reply itself, without any tags.")
        prompt = "\n".join(lines)

    if images:
        prompt += "\n\n(The images referred to as [image N] are attached, in order.)"
    return prompt, system, images


def function_tools(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Function definitions from ``tools`` (chat or responses style) or legacy ``functions``."""
    out = []
    for tool in request.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if fn.get("name"):
            out.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description") or "",
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
    for fn in request.get("functions") or []:
        if isinstance(fn, dict) and fn.get("name"):
            out.append({"name": fn["name"], "description": fn.get("description") or "",
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}}})
    return out


def forced_tool(choice: Any) -> Optional[str]:
    """Name of the function ``tool_choice`` forces, if any."""
    if isinstance(choice, dict):
        fn = choice.get("function") if isinstance(choice.get("function"), dict) else choice
        return fn.get("name")
    return None


def tool_instructions(tools: List[Dict[str, Any]], choice: Any, parallel: bool, args_as_string: bool) -> str:
    forced = forced_tool(choice)
    if forced:
        rule = f'You must call the function "{forced}" in this reply.'
    elif choice in ("required", "any"):
        rule = "You must call at least one function in this reply."
    else:
        rule = "Call functions when they help you answer; otherwise answer directly."
    if parallel:
        rule += " You may call several functions at once when they don't depend on each other."
    else:
        rule += " Call at most one function per reply."
    arguments = (
        "a JSON-encoded string of the arguments object" if args_as_string
        else "an object with the arguments, matching the function's parameters schema"
    )
    return "\n".join(
        [
            "# Functions",
            "You are connected to an application that runs the functions below on your behalf. "
            "You cannot run them yourself: to use one, put a call in your reply. The application "
            "runs it and sends you the result in a later turn, inside <tool_result> tags.",
            "",
            "<functions>",
            *(json.dumps(tool, ensure_ascii=False) for tool in tools),
            "</functions>",
            "",
            "# Reply format",
            "Reply with a single JSON object and nothing else, with exactly these keys:",
            '- "content": your message to the user, as a string ("" if you are only calling functions).',
            f'- "tool_calls": the function calls to make now, as a list of {{"name", "arguments"}} objects, '
            f"where arguments is {arguments}. Use [] when you are not calling any function.",
            rule,
            "Once the results you need are in the conversation, answer the user in \"content\" "
            'with "tool_calls": [].',
        ]
    )


def tool_schema(names: List[str], choice: Any, parallel: bool, args_as_string: bool, strict: bool) -> Dict[str, Any]:
    """JSON Schema for the tool-mode reply. ``strict`` keeps it within OpenAI's strict subset."""
    forced = forced_tool(choice)
    arguments: Dict[str, Any] = (
        {"type": "string", "description": "JSON-encoded arguments object"} if args_as_string
        else {"type": "object"}
    )
    call = {
        "type": "object",
        "properties": {"name": {"type": "string", "enum": [forced] if forced else names}, "arguments": arguments},
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }
    calls: Dict[str, Any] = {"type": "array", "items": call}
    if not strict:
        if forced or choice in ("required", "any"):
            calls["minItems"] = 1
        if not parallel:
            calls["maxItems"] = 1
    return {
        "type": "object",
        "properties": {"content": {"type": "string"}, "tool_calls": calls},
        "required": ["content", "tool_calls"],
        "additionalProperties": False,
    }


def format_instructions(schema: Optional[Dict[str, Any]]) -> str:
    if schema:
        return (
            "# Output format\nRespond with only a JSON value that conforms to this JSON Schema, "
            "with no Markdown code fences and no other text:\n" + json.dumps(schema, ensure_ascii=False)
        )
    return "# Output format\nRespond with only a valid JSON object, with no Markdown code fences and no other text."


def extract_json(text: Optional[str]) -> Any:
    """Best-effort: parse JSON from a model reply that may be fenced or have prose around it."""
    if not text:
        return None
    stripped = text.strip()
    candidates = [stripped]
    fenced = _FENCE.search(stripped)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except ValueError:
            pass
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(stripped) if ch in "{["][:20]
    for start in starts:
        try:
            return decoder.raw_decode(stripped, start)[0]
        except ValueError:
            continue
    return None


def parse_tool_reply(data: Any, text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Return ``(content, [(name, arguments_json), ...])`` from a tool-mode reply."""
    if data is None:
        data = extract_json(text)
    if not isinstance(data, dict):
        return text, []
    raw_calls = data.get("tool_calls")
    if raw_calls is None and "name" in data and "arguments" in data:
        raw_calls = [data]
    calls: List[Tuple[str, str]] = []
    for raw in raw_calls if isinstance(raw_calls, list) else []:
        if not isinstance(raw, dict):
            continue
        fn = raw.get("function") if isinstance(raw.get("function"), dict) else raw
        name = fn.get("name")
        if not name:
            continue
        args = fn.get("arguments", fn.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                pass
        calls.append((name, args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)))
    content = data.get("content")
    if content is None:
        content = ""
    elif not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return content, calls
