"""OpenAI-compatible chat endpoint for the local Coach.

  uv run python -m healthllm.server                            # real data (dashboard/.env), fine-tuned adapter
  DEMO=1 uv run python -m healthllm.server                     # synthetic data
  HEALTH_LLM_ADAPTER=none uv run python -m healthllm.server    # the base model, for comparison

Then, from any OpenAI client:
  curl -N localhost:8800/v1/chat/completions -H 'content-type: application/json' \\
       -d '{"model":"coach","stream":true,"messages":[{"role":"user","content":"How did I sleep this week?"}]}'

Or in Claude's Messages API format (works with the `anthropic` SDK via base_url):
  curl -N localhost:8800/v1/messages -H 'content-type: application/json' \\
       -d '{"model":"coach","max_tokens":400,"stream":true,"messages":[{"role":"user","content":"Am I on track?"}]}'

The tool loop (agent.py) runs here, not in the client. Only the answer text is streamed back.
Everything binds to 127.0.0.1: nothing on your network, let alone the internet, can reach it.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import agent
from .bridge import Runtime

ROOT = Path(__file__).resolve().parents[2]
BASE = os.environ.get("HEALTH_LLM_BASE", "Qwen/Qwen3.5-0.8B")
ADAPTER = os.environ.get("HEALTH_LLM_ADAPTER", str(ROOT / "models" / "coach-0.8b-v3" / "adapter"))
DEMO = os.environ.get("DEMO", "0") == "1"
PORT = int(os.environ.get("HEALTH_LLM_PORT", 8800))
NOTES_PATH = ROOT / "data" / "notes.json"  # the local Coach's own notes (gitignored)


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatRequest(BaseModel):
    model: str = "coach"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = Field(0.3, ge=0, le=2)
    max_tokens: int | None = Field(None, ge=1, le=1024)


class MessagesRequest(BaseModel):
    """Claude's Messages API request. `system` is accepted but ignored, as above."""
    model: str = "coach"
    messages: list[ChatMessage]
    max_tokens: int = Field(400, ge=1, le=1024)
    system: str | list[dict[str, Any]] | None = None
    stream: bool = False
    temperature: float = Field(0.3, ge=0, le=1)


def _text(content: str | list[dict[str, Any]] | None) -> str:
    """A message's content as plain text: a string, or the text parts of a content-block list."""
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in (content or []) if isinstance(p, dict))


class State:
    def __init__(self) -> None:
        self.model, self.tok = agent.load(BASE, ADAPTER)
        self.adapter = ADAPTER if ADAPTER.lower() != "none" and Path(ADAPTER).exists() else None
        self.runtime = Runtime(demo=DEMO)
        NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.tools = self.runtime.tools(NOTES_PATH)
        self.lock = threading.Lock()  # one GPU: one generation at a time


app = FastAPI(title="Local Coach", docs_url="/docs")
state: State | None = None


@app.on_event("startup")
def _load() -> None:
    global state
    state = State()
    print(f"ready on http://127.0.0.1:{PORT}  base={BASE} adapter={state.adapter or 'none'} "
          f"data={'demo' if DEMO else 'real'}", flush=True)


def _history(req: ChatRequest | MessagesRequest) -> list[dict]:
    """Client messages -> the model's format. The date rides on each user message, as in the Coach."""
    today = state.tools.today()
    msgs = []
    for m in req.messages:
        text = _text(m.content)
        if m.role == "user":
            msgs.append({"role": "user", "content": f"[Today is {today:%A, %B %-d, %Y}.]\n\n{text}"})
        elif m.role == "assistant":
            msgs.append({"role": "assistant", "content": text})
        # client "system" messages are ignored: the Coach's system prompt is fixed by training
    if not msgs or msgs[-1]["role"] != "user":
        raise HTTPException(400, "The last message must be from the user.")
    return msgs


def _locked_run(msgs: list[dict], temperature: float, max_new: int, stream: bool) -> Iterator[dict]:
    """Run the agent under the GPU lock in a worker thread, yielding its events.

    The worker always runs to the end and releases the lock, even if the client disconnects mid-stream.
    (Holding the lock inside the response generator deadlocks: a disconnected client leaves that
    generator suspended forever, lock held, and every later request waits.)
    """
    q: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            with state.lock:
                for e in agent.run(state.model, state.tok, state.tools, msgs, temperature, max_new, stream,
                                   include_tools=state.adapter is None):
                    q.put(e)
        except Exception as exc:  # hand errors to the request instead of losing them in the thread
            q.put(exc)
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (e := q.get()) is not None:
        if isinstance(e, Exception):
            raise e
        yield e


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": state is not None, "base": BASE, "adapter": state and state.adapter, "demo": DEMO}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": "coach", "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions")
def completions(req: ChatRequest):
    msgs = _history(req)
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
    max_new = req.max_tokens or 400

    def events() -> Iterator[dict]:
        return _locked_run(msgs, req.temperature, max_new, req.stream)

    def chunk(delta: dict, finish: str | None = None) -> str:
        return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                      "model": "coach", "choices": [{"index": 0, "delta": delta,
                                                                     "finish_reason": finish}]}) + "\n\n"

    if req.stream:
        def sse() -> Iterator[str]:
            yield chunk({"role": "assistant"})
            for ev in events():
                if ev["type"] == "text":
                    yield chunk({"content": ev["text"]})
                elif ev["type"] == "done":
                    yield chunk({}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    text, tools_used = "", []
    for ev in events():
        if ev["type"] == "text":
            text += ev["text"]
        elif ev["type"] == "tool":
            tools_used.append({"name": ev["name"], "arguments": ev["arguments"], "error": ev["error"]})
    return JSONResponse({
        "id": cid, "object": "chat.completion", "created": created, "model": "coach",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text.strip()},
                     "finish_reason": "stop"}],
        "tools_used": tools_used,  # non-standard extra field: which tools answered the question
    })


@app.post("/v1/messages")
def messages(req: MessagesRequest):
    """Claude's Messages API shape: content blocks, stop_reason, usage, and its named SSE events.

    Tools still run server-side (like Claude's server tools), so the client only ever sees text.
    """
    msgs = _history(req)
    mid = f"msg_{uuid.uuid4().hex[:24]}"
    n_in = len(state.tok("".join(m["content"] for m in msgs)).input_ids)  # approximate: excludes the system prompt
    n_out = lambda s: len(state.tok(s).input_ids)

    def events() -> Iterator[dict]:
        return _locked_run(msgs, req.temperature, req.max_tokens, req.stream)

    def message(content: list, stop: str | None, out: int) -> dict:
        return {"id": mid, "type": "message", "role": "assistant", "model": req.model, "content": content,
                "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": n_in, "output_tokens": out}}

    if req.stream:
        def sse() -> Iterator[str]:
            def ev(name: str, data: dict) -> str:
                return f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n"

            yield ev("message_start", {"message": message([], None, 1)})
            yield ev("ping", {})
            text, index, open_text = "", 0, False
            for e in events():
                if e["type"] == "tool":
                    # Like Claude's server tools: a server_tool_use block per call, already run here.
                    if open_text:
                        yield ev("content_block_stop", {"index": index})
                        index, open_text = index + 1, False
                    block = {"type": "server_tool_use", "id": f"srvtoolu_{uuid.uuid4().hex[:24]}",
                             "name": e["name"], "input": e["arguments"]}
                    yield ev("content_block_start", {"index": index, "content_block": block})
                    yield ev("content_block_stop", {"index": index})
                    index += 1
                elif e["type"] == "text":
                    if not open_text:
                        yield ev("content_block_start", {"index": index, "content_block": {"type": "text", "text": ""}})
                        open_text = True
                    text += e["text"]
                    yield ev("content_block_delta", {"index": index, "delta": {"type": "text_delta", "text": e["text"]}})
            if open_text:
                yield ev("content_block_stop", {"index": index})
            yield ev("message_delta", {"delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                        "usage": {"output_tokens": n_out(text)}})
            yield ev("message_stop", {})

        return StreamingResponse(sse(), media_type="text/event-stream")

    text, tools_used = "", []
    for e in events():
        if e["type"] == "text":
            text += e["text"]
        elif e["type"] == "tool":
            tools_used.append({"name": e["name"], "input": e["arguments"], "error": e["error"]})
    text = text.strip()
    return JSONResponse({**message([{"type": "text", "text": text}], "end_turn", n_out(text)),
                         "tools_used": tools_used})  # non-standard extra field, as in the OpenAI endpoint


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
