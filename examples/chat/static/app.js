const KEYS = {
  chats: "clibridge.chats.v1",
  current: "clibridge.current",
  model: "clibridge.model",
  effort: "clibridge.effort",
  tools: "clibridge.tools",
};

const SUGGESTIONS = [
  { title: "Do some math", hint: "Watch the calculator tool run", prompt: "What is 17*23? Use the calculator tool." },
  { title: "Check the time", hint: "Ask for a specific timezone", prompt: "What time is it in Tokyo right now? Use the time tool." },
  { title: "Inspect the workspace", hint: "List files, then read one", prompt: "List the files in the workspace, then read task.txt and tell me the codename." },
  { title: "Ask a general question", hint: "No tools needed", prompt: "Is 391 prime? Explain briefly." },
];

const state = {
  chats: readChats(),
  currentId: localStorage.getItem(KEYS.current) || null,
  streaming: null,
  matrixLoaded: false,
  matrixCaps: [],
  tools: localStorage.getItem(KEYS.tools) !== "0",
};

let stick = true;

const dom = {
  sidebar: document.getElementById("sidebar"),
  chatList: document.getElementById("chat-list"),
  messages: document.getElementById("messages"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  model: document.getElementById("model"),
  effort: document.getElementById("effort"),
  toolsToggle: document.getElementById("tools-toggle"),
  chatView: document.getElementById("chat-view"),
  matrixView: document.getElementById("matrix-view"),
  grid: document.getElementById("grid"),
  runall: document.getElementById("runall"),
  progress: document.getElementById("progress"),
  newChat: document.getElementById("new-chat"),
  menuBtn: document.getElementById("menu-btn"),
  closeSidebar: document.getElementById("close-sidebar"),
  scrim: document.getElementById("scrim"),
  openMatrix: document.getElementById("open-matrix"),
};

function readChats() {
  try {
    const value = JSON.parse(localStorage.getItem(KEYS.chats));
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

function saveChats() {
  try {
    localStorage.setItem(KEYS.chats, JSON.stringify(state.chats));
  } catch {
    /* storage can be unavailable in private modes */
  }
}

function newId() {
  return crypto.randomUUID ? crypto.randomUUID() : `id-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function currentChat() {
  return state.chats.find((chat) => chat.id === state.currentId) || null;
}

function titleFor(text) {
  return text.replace(/\s+/g, " ").trim().slice(0, 60) || "New chat";
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function inlineMarkdown(text) {
  let out = escapeHtml(text);
  out = out.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return out;
}

function renderMarkdown(source) {
  const codeBlocks = [];
  const text = String(source).replace(/```([\w+-]*)\n([\s\S]*?)(?:```|$)/g, (_, lang, code) => {
    codeBlocks.push({ lang: lang || "code", code });
    return `\n@@CODE${codeBlocks.length - 1}@@\n`;
  });
  const html = [];
  let list = null;
  const closeList = () => {
    if (list) {
      html.push(`</${list}>`);
      list = null;
    }
  };
  for (const rawLine of text.split("\n")) {
    const line = rawLine.replace(/\s+$/, "");
    const codeMatch = line.match(/^@@CODE(\d+)@@$/);
    if (codeMatch) {
      closeList();
      const block = codeBlocks[Number(codeMatch[1])];
      html.push(
        `<div class="code-block"><div class="code-head"><span>${escapeHtml(block.lang)}</span>` +
        `<button class="copy-btn" type="button">Copy</button></div><pre><code>${escapeHtml(block.code)}</code></pre></div>`
      );
      continue;
    }
    if (!line.trim()) {
      closeList();
      html.push('<div class="spacer"></div>');
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      closeList();
      const level = heading[1].length;
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    if (bullet) {
      if (list !== "ul") {
        closeList();
        html.push("<ul>");
        list = "ul";
      }
      html.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
      continue;
    }
    const numbered = line.match(/^\s*\d+\.\s+(.*)$/);
    if (numbered) {
      if (list !== "ol") {
        closeList();
        html.push("<ol>");
        list = "ol";
      }
      html.push(`<li>${inlineMarkdown(numbered[1])}</li>`);
      continue;
    }
    if (line.startsWith("> ")) {
      closeList();
      html.push(`<blockquote>${inlineMarkdown(line.slice(2))}</blockquote>`);
      continue;
    }
    closeList();
    html.push(`<p>${inlineMarkdown(line)}</p>`);
  }
  closeList();
  return html.join("");
}

function prettyJson(value) {
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return value;
    }
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function setView(view) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === view));
  dom.chatView.hidden = view !== "chat";
  dom.matrixView.hidden = view !== "matrix";
  if (view === "matrix") loadMatrix();
  else {
    renderMessages();
    dom.input.focus();
  }
}

function closeSidebar() {
  document.body.classList.remove("sidebar-open");
}

function renderSidebar() {
  dom.chatList.replaceChildren();
  const sorted = [...state.chats].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  if (!sorted.length) {
    const empty = document.createElement("p");
    empty.className = "empty-list";
    empty.textContent = "No chats yet";
    dom.chatList.appendChild(empty);
    return;
  }
  for (const chat of sorted) {
    const item = document.createElement("div");
    item.className = "chat-item" + (chat.id === state.currentId ? " active" : "");
    item.title = chat.title || "Chat";
    const title = document.createElement("span");
    title.className = "chat-title";
    title.textContent = chat.title || "Chat";
    const remove = document.createElement("button");
    remove.className = "chat-delete";
    remove.type = "button";
    remove.title = "Delete chat";
    remove.textContent = "\u00d7";
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteChat(chat.id);
    });
    item.append(title, remove);
    item.addEventListener("click", () => {
      if (state.streaming) stopStreaming();
      state.currentId = chat.id;
      localStorage.setItem(KEYS.current, chat.id);
      closeSidebar();
      setView("chat");
      renderSidebar();
      renderMessages(true);
    });
    dom.chatList.appendChild(item);
  }
}

function deleteChat(id) {
  const chat = state.chats.find((entry) => entry.id === id);
  if (chat && !confirm(`Delete "${chat.title || "this chat"}"?`)) return;
  if (state.streaming && state.streaming.chatId === id) stopStreaming();
  state.chats = state.chats.filter((entry) => entry.id !== id);
  if (state.currentId === id) {
    state.currentId = null;
    localStorage.removeItem(KEYS.current);
  }
  saveChats();
  renderSidebar();
  renderMessages();
}

function renderEmpty() {
  const wrap = document.createElement("div");
  wrap.className = "empty";
  wrap.innerHTML = `
    <div class="empty-mark">cb</div>
    <h1>What can I help with?</h1>
    <p>Ask a question, do some math, check the time, or inspect the demo workspace. Tool calls appear as the model makes them.</p>
  `;
  const grid = document.createElement("div");
  grid.className = "suggestions";
  for (const suggestion of SUGGESTIONS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "suggestion";
    button.dataset.prompt = suggestion.prompt;
    const title = document.createElement("b");
    title.textContent = suggestion.title;
    const hint = document.createElement("small");
    hint.textContent = suggestion.hint;
    button.append(title, hint);
    grid.appendChild(button);
  }
  wrap.appendChild(grid);
  dom.messages.replaceChildren(wrap);
  dom.messages.scrollTop = 0;
}

function toolElement(tool) {
  const card = document.createElement("div");
  card.className = "tool" + (tool.status === "error" ? " error" : "");
  const head = document.createElement("div");
  head.className = "tool-head";
  const glyph = document.createElement("span");
  glyph.className = "tool-glyph";
  glyph.textContent = "tool";
  const name = document.createElement("span");
  name.className = "tool-name";
  name.textContent = tool.name || "tool";
  const status = document.createElement("span");
  status.className = "tool-status " + (tool.status || "running");
  status.textContent = tool.status === "running" ? "running..." : tool.status === "error" ? "error" : "done";
  head.append(glyph, name, status);
  card.appendChild(head);

  const args = document.createElement("pre");
  args.className = "tool-args";
  args.textContent = prettyJson(tool.arguments);
  card.appendChild(args);

  if (tool.result) {
    const details = document.createElement("details");
    details.className = "tool-result";
    details.open = true;
    const summary = document.createElement("summary");
    summary.textContent = "Result";
    const result = document.createElement("pre");
    result.textContent = prettyJson(tool.result);
    details.append(summary, result);
    card.appendChild(details);
  }
  return card;
}

function messageElement(message) {
  if (message.role === "user") {
    const wrap = document.createElement("div");
    wrap.className = "msg user";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = message.content;
    wrap.appendChild(bubble);
    return wrap;
  }

  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "cb";
  const body = document.createElement("div");
  body.className = "assistant-body";

  const meta = document.createElement("div");
  meta.className = "meta";
  const label = document.createElement("span");
  label.textContent = message.model || dom.model.value;
  meta.appendChild(label);
  if (message.seconds) {
    const seconds = document.createElement("span");
    seconds.textContent = `${message.seconds}s`;
    meta.appendChild(seconds);
  }
  if (message.usage && message.usage.total_tokens) {
    meta.title = `${message.usage.total_tokens} tokens`;
  }
  if (message.streaming) {
    const dot = document.createElement("span");
    dot.className = "streaming-dot";
    meta.appendChild(dot);
  }
  body.appendChild(meta);

  if (message.reasoning) {
    const details = document.createElement("details");
    details.className = "reasoning";
    details.open = Boolean(message.streaming);
    const summary = document.createElement("summary");
    summary.textContent = `Thinking \u00b7 ${message.reasoning.length.toLocaleString()} chars`;
    const text = document.createElement("div");
    text.className = "rtext";
    text.textContent = message.reasoning;
    details.append(summary, text);
    body.appendChild(details);
  }

  for (const tool of message.tools || []) body.appendChild(toolElement(tool));

  if (message.content) {
    const content = document.createElement("div");
    content.className = "content";
    content.innerHTML = renderMarkdown(message.content);
    if (message.streaming) {
      const cursor = document.createElement("span");
      cursor.className = "cursor";
      content.appendChild(cursor);
    }
    body.appendChild(content);
  } else if (message.streaming && !(message.tools || []).some((tool) => tool.status === "running")) {
    const waiting = document.createElement("div");
    waiting.className = "meta";
    waiting.textContent = "waiting for the model...";
    body.appendChild(waiting);
  }

  if (message.error) {
    const error = document.createElement("div");
    error.className = "error-box";
    error.textContent = message.error;
    body.appendChild(error);
  }
  if (message.stopped) {
    const stopped = document.createElement("div");
    stopped.className = "meta";
    stopped.textContent = "stopped";
    body.appendChild(stopped);
  }

  wrap.append(avatar, body);
  return wrap;
}

function renderMessages(forceScroll = false) {
  const chat = currentChat();
  const previousTop = dom.messages.scrollTop;
  const wasStick = stick;
  if (forceScroll) stick = true;

  if (!chat || !chat.messages.length) {
    renderEmpty();
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const message of chat.messages) fragment.appendChild(messageElement(message));
  dom.messages.replaceChildren(fragment);

  if (stick || wasStick) dom.messages.scrollTop = dom.messages.scrollHeight;
  else dom.messages.scrollTop = previousTop;
}

function setStreaming(active) {
  dom.send.classList.toggle("stop", active);
  dom.send.title = active ? "Stop" : "Send";
  dom.send.setAttribute("aria-label", active ? "Stop" : "Send");
  dom.model.disabled = active;
  dom.effort.disabled = active;
  dom.toolsToggle.disabled = active;
}

function stopStreaming() {
  if (!state.streaming) return;
  state.streaming.controller.abort();
}

function updateToolsChip() {
  dom.toolsToggle.classList.toggle("active", state.tools);
  dom.toolsToggle.setAttribute("aria-pressed", String(state.tools));
  dom.toolsToggle.title = state.tools ? "Tools are on" : "Tools are off";
}

function resizeInput() {
  dom.input.style.height = "auto";
  dom.input.style.height = `${Math.min(dom.input.scrollHeight, 180)}px`;
}

async function sendMessage(text) {
  if (state.streaming) return;
  const content = (text ?? dom.input.value).trim();
  if (!content) return;

  const now = Date.now();
  let chat = currentChat();
  if (!chat) {
    chat = { id: newId(), title: titleFor(content), createdAt: now, updatedAt: now, messages: [] };
    state.chats.unshift(chat);
    state.currentId = chat.id;
    localStorage.setItem(KEYS.current, chat.id);
  }
  if (!chat.messages.some((message) => message.role === "user")) chat.title = titleFor(content);

  chat.messages.push({ id: newId(), role: "user", content, createdAt: now });
  chat.updatedAt = now;

  const history = [];
  for (const message of chat.messages) {
    if (message.role === "user") history.push({ role: "user", content: message.content });
    else if (message.role === "assistant" && message.content) history.push({ role: "assistant", content: message.content });
  }

  const assistant = {
    id: newId(),
    role: "assistant",
    content: "",
    reasoning: "",
    tools: [],
    streaming: true,
    model: dom.model.value,
    createdAt: Date.now(),
  };
  chat.messages.push(assistant);

  dom.input.value = "";
  resizeInput();
  dom.input.focus();
  renderSidebar();
  stick = true;
  renderMessages(true);

  const controller = new AbortController();
  state.streaming = { controller, chatId: chat.id, messageId: assistant.id };
  setStreaming(true);

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        messages: history,
        model: dom.model.value,
        effort: dom.effort.value || null,
        tools: state.tools,
      }),
    });
    if (!response.ok || !response.body) {
      const detail = await response.text().catch(() => "");
      throw new Error(detail || `HTTP ${response.status}`);
    }
    await consumeSSE(response.body, (event) => handleEvent(chat.id, assistant.id, event));
  } catch (error) {
    if (error.name === "AbortError") assistant.stopped = true;
    else assistant.error = error.message || String(error);
  } finally {
    assistant.streaming = false;
    if (state.streaming && state.streaming.controller === controller) state.streaming = null;
    setStreaming(false);
    chat.updatedAt = Date.now();
    saveChats();
    renderSidebar();
    if (state.currentId === chat.id) renderMessages();
  }
}

function handleEvent(chatId, messageId, event) {
  const chat = state.chats.find((entry) => entry.id === chatId);
  if (!chat) return;
  const message = chat.messages.find((entry) => entry.id === messageId);
  if (!message) return;

  if (event.type === "start") message.model = event.model;
  else if (event.type === "reasoning") message.reasoning = (message.reasoning || "") + event.text;
  else if (event.type === "text") message.content = (message.content || "") + event.text;
  else if (event.type === "tool_call") {
    message.tools = message.tools || [];
    message.tools.push({ id: event.id, name: event.name, arguments: event.arguments, status: "running" });
  } else if (event.type === "tool_result") {
    const tool = (message.tools || []).find((entry) => entry.id === event.id);
    if (tool) {
      tool.result = event.result;
      tool.status = event.ok === false ? "error" : "done";
    }
  } else if (event.type === "usage") {
    message.usage = event;
  } else if (event.type === "done") {
    message.seconds = event.seconds;
    message.streaming = false;
  } else if (event.type === "error") {
    message.error = event.message;
    message.streaming = false;
  }

  if (state.currentId === chatId) renderMessages();
}

async function consumeSSE(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("\n");
      if (!data) continue;
      try {
        onEvent(JSON.parse(data));
      } catch (error) {
        console.error("bad SSE event", error, data);
      }
    }
  }
}

async function loadMatrix() {
  if (state.matrixLoaded) return;
  dom.grid.innerHTML = '<p class="empty-list">Loading capabilities...</p>';
  try {
    const response = await fetch("/api/capabilities");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.matrixCaps = await response.json();
    state.matrixLoaded = true;
    renderMatrix();
  } catch (error) {
    dom.grid.innerHTML = `<p class="empty-list">Could not load capabilities: ${escapeHtml(error.message)}</p>`;
  }
}

function renderMatrix() {
  dom.grid.replaceChildren();
  for (const cap of state.matrixCaps) {
    const card = document.createElement("article");
    card.className = "card";
    card.id = `cap-${cap.id}`;
    const row = document.createElement("div");
    row.className = "row";
    const title = document.createElement("h3");
    title.textContent = cap.label;
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = "idle";
    row.append(title, badge);
    const description = document.createElement("p");
    description.textContent = cap.description;
    const action = document.createElement("div");
    action.className = "row";
    const run = document.createElement("button");
    run.type = "button";
    run.className = "run-card";
    run.textContent = "Run";
    run.addEventListener("click", () => runCapability(cap, card));
    action.appendChild(run);
    const output = document.createElement("pre");
    card.append(row, description, action, output);
    dom.grid.appendChild(card);
  }
}

async function runCapability(cap, card) {
  const badge = card.querySelector(".badge");
  const output = card.querySelector("pre");
  const button = card.querySelector(".run-card");
  const started = performance.now();
  badge.className = "badge running";
  badge.textContent = "running...";
  button.disabled = true;
  output.classList.add("show");
  output.textContent = "working...";
  try {
    const response = await fetch(`/api/run/${encodeURIComponent(cap.id)}?model=${encodeURIComponent(dom.model.value)}`, { method: "POST" });
    const data = await response.json();
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    badge.className = "badge " + (data.ok ? "pass" : "fail");
    badge.textContent = `${data.ok ? "pass" : "fail"} \u00b7 ${seconds}s`;
    output.textContent = data.ok ? JSON.stringify(data.result, null, 2) : `ERROR: ${data.error}`;
    return Boolean(data.ok);
  } catch (error) {
    badge.className = "badge fail";
    badge.textContent = "fail";
    output.textContent = `ERROR: ${error.message}`;
    return false;
  } finally {
    button.disabled = false;
  }
}

dom.messages.addEventListener("scroll", () => {
  const distance = dom.messages.scrollHeight - dom.messages.scrollTop - dom.messages.clientHeight;
  stick = distance < 120;
});

dom.messages.addEventListener("click", (event) => {
  const suggestion = event.target.closest(".suggestion");
  if (suggestion) {
    sendMessage(suggestion.dataset.prompt);
    return;
  }
  const copyButton = event.target.closest(".copy-btn");
  if (!copyButton) return;
  const block = copyButton.closest(".code-block");
  const code = block ? block.querySelector("code").textContent : "";
  navigator.clipboard.writeText(code).then(() => {
    copyButton.textContent = "Copied";
    setTimeout(() => {
      copyButton.textContent = "Copy";
    }, 1200);
  });
});

dom.input.addEventListener("input", resizeInput);
dom.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!state.streaming) sendMessage();
  }
});

dom.send.addEventListener("click", () => {
  if (state.streaming) stopStreaming();
  else sendMessage();
});

dom.newChat.addEventListener("click", () => {
  if (state.streaming) stopStreaming();
  state.currentId = null;
  localStorage.removeItem(KEYS.current);
  closeSidebar();
  setView("chat");
  renderSidebar();
  renderMessages();
  dom.input.focus();
});

dom.toolsToggle.addEventListener("click", () => {
  state.tools = !state.tools;
  localStorage.setItem(KEYS.tools, state.tools ? "1" : "0");
  updateToolsChip();
});

dom.model.addEventListener("change", () => {
  localStorage.setItem(KEYS.model, dom.model.value);
});

dom.effort.addEventListener("change", () => {
  localStorage.setItem(KEYS.effort, dom.effort.value);
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => setView(tab.dataset.view));
});

dom.openMatrix.addEventListener("click", () => {
  closeSidebar();
  setView("matrix");
});

dom.menuBtn.addEventListener("click", () => {
  document.body.classList.add("sidebar-open");
});

dom.closeSidebar.addEventListener("click", closeSidebar);
dom.scrim.addEventListener("click", closeSidebar);

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeSidebar();
});

dom.runall.addEventListener("click", async () => {
  if (!state.matrixLoaded) await loadMatrix();
  if (!state.matrixCaps.length) return;
  const model = dom.model.value;
  dom.runall.disabled = true;
  let passed = 0;
  for (let index = 0; index < state.matrixCaps.length; index++) {
    const cap = state.matrixCaps[index];
    dom.progress.textContent = `running ${index + 1}/${state.matrixCaps.length} on ${model}`;
    const card = document.getElementById(`cap-${cap.id}`);
    if (card && (await runCapability(cap, card))) passed++;
  }
  dom.progress.textContent = `done: ${passed}/${state.matrixCaps.length} passed on ${model}`;
  dom.runall.disabled = false;
});

function init() {
  if (state.currentId && !state.chats.some((chat) => chat.id === state.currentId)) {
    state.currentId = null;
    localStorage.removeItem(KEYS.current);
  }
  const savedModel = localStorage.getItem(KEYS.model);
  if (savedModel && [...dom.model.options].some((option) => option.value === savedModel)) dom.model.value = savedModel;
  const savedEffort = localStorage.getItem(KEYS.effort);
  if (savedEffort !== null) dom.effort.value = savedEffort;
  updateToolsChip();
  renderSidebar();
  renderMessages();
  setView("chat");
  resizeInput();
}

init();
