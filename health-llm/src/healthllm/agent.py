"""The local Coach's agent loop, shared by the chat server and the evaluation script.

    generate -> parse tool calls -> run tools -> append results -> generate again ... -> plain-text answer
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .bridge import ToolError, call_tool
from .chat_format import parse_reply, render

MAX_STEPS = 6


def load(base: str, adapter: str | None, device: str | None = None):
    """(model, tokenizer). With an adapter, LoRA's B·A is merged into the weights for full-speed inference."""
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa")
    if adapter and adapter.lower() != "none" and Path(adapter).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.to(device).eval(), tok


def generate(model, tok, messages: list[dict], temperature: float = 0.0, max_new: int = 400,
             on_text: Callable[[str], None] | None = None, include_tools: bool = False) -> str:
    prompt = render(tok, messages, add_generation_prompt=True, include_tools=include_tools)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    kwargs = dict(**ids, max_new_tokens=max_new, eos_token_id=end_id, do_sample=temperature > 0,
                  pad_token_id=tok.pad_token_id or end_id)
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=0.9)
    if on_text is None:
        with torch.no_grad():
            out = model.generate(**kwargs)
        return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=False)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)
    th = threading.Thread(target=model.generate, kwargs={**kwargs, "streamer": streamer})
    th.start()
    text = ""
    for piece in streamer:
        text += piece
        on_text(text)
    th.join()
    return text


_MARKERS = ("<tool_call>", "<|im_end|>")


def _visible(so_far: str) -> str:
    """The part of the output that is safe to show: text before any tool call or end marker, minus a
    trailing "<too" that might be the start of one (it's held back until we know)."""
    text = so_far.split("<tool_call>")[0].split("<|im_end|>")[0]
    cut = text.rfind("<")
    if cut >= 0 and any(m.startswith(text[cut:]) for m in _MARKERS):
        text = text[:cut]
    return text


def _stream_step(model, tok, messages, temperature, max_new, include_tools):
    """Generate one model turn in a worker thread, yielding text events *while* it's written.

    Returns the raw output (via StopIteration, so callers use `raw = yield from ...`).
    """
    q: queue.Queue = queue.Queue()
    result: dict[str, Any] = {}

    def work() -> None:
        try:
            result["raw"] = generate(model, tok, messages, temperature, max_new, q.put, include_tools)
        except Exception as e:  # noqa: BLE001 - re-raised in the caller's thread
            result["error"] = e
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()
    sent = 0
    while (so_far := q.get()) is not None:
        visible = _visible(so_far)
        if len(visible) > sent:
            yield {"type": "text", "text": visible[sent:]}
            sent = len(visible)
    if "error" in result:
        raise result["error"]
    return result["raw"]


def run(model, tok, tools, messages: list[dict], temperature: float = 0.0, max_new: int = 400,
        stream: bool = False, include_tools: bool = False) -> Iterator[dict[str, Any]]:
    """Run one user turn. Appends assistant/tool messages to `messages` in place and yields events:
    {"type": "text", "text"} | {"type": "tool", "name", "arguments", "error"} | {"type": "done", "steps"}.
    """
    for step in range(MAX_STEPS):
        if stream:
            raw = yield from _stream_step(model, tok, messages, temperature, max_new, include_tools)
        else:
            raw = generate(model, tok, messages, temperature, max_new, None, include_tools)
        text, calls = parse_reply(raw)
        msg = {"role": "assistant", "content": text}
        if calls:
            msg["tool_calls"] = calls
        messages.append(msg)
        if not calls:
            if not stream:
                yield {"type": "text", "text": text}
            yield {"type": "done", "steps": step + 1}
            return
        for c in calls:
            try:
                out, err = call_tool(tools, c["name"], c["arguments"]), None
            except ToolError as e:
                out, err = {"error": str(e)}, str(e)
            except Exception as e:  # noqa: BLE001  (e.g. the database is unreachable)
                out, err = {"error": f"Tool failed: {type(e).__name__}"}, type(e).__name__
            yield {"type": "tool", "name": c["name"], "arguments": c["arguments"], "error": err}
            messages.append({"role": "tool", "name": c["name"],
                             "content": json.dumps(out, separators=(",", ":"), ensure_ascii=False),
                             **({"is_error": True} if err else {})})
    messages.append({"role": "assistant", "content": "(I needed too many steps for that one.)"})
    yield {"type": "text", "text": "(I needed too many steps for that one. Try a narrower question.)"}
    yield {"type": "done", "steps": MAX_STEPS, "gave_up": True}
