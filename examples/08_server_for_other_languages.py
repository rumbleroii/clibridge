"""The HTTP server, for code that isn't Python (JavaScript, curl, ...) or runs elsewhere.

    clibridge serve                                  # terminal 1
    export OPENAI_BASE_URL=http://localhost:8787/v1  # terminal 2
    export OPENAI_API_KEY=anything
    python examples/08_server_for_other_languages.py

This file is Python only to keep the examples runnable; any OpenAI-compatible client works.
"""

import json
import os
import urllib.request

url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8787/v1") + "/chat/completions"
body = {"model": "claude-code", "messages": [{"role": "user", "content": "Say hello!"}]}
request = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
print(json.load(urllib.request.urlopen(request))["choices"][0]["message"]["content"])
