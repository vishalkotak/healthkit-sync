"""One renderer for both training and serving, so the model sees the same text format in both.

Why this exists: Qwen3.5's chat template puts an empty `<think>\\n\\n</think>` block only on the
newest assistant turn. At chat time every assistant step is generated right after that block,
but plain template output would show it only on the last turn of each training conversation.
That mismatch is called train/serve skew. Here we render every assistant turn the same way.

Training labels: only the assistant's own text is learned (everything else is masked to -100).
The model learns what to say, not to predict the system prompt, the user, or tool results.
"""

from __future__ import annotations

import json
from typing import Any

from .bridge import TOOL_SPECS
from app.coach import SYSTEM as COACH_SYSTEM  # noqa: E402  (bridge put the dashboard on sys.path)

THINK = "<think>\n\n</think>\n\n"
ASSISTANT_OPEN = "<|im_start|>assistant\n"
END = "<|im_end|>"
GEN_PROMPT = ASSISTANT_OPEN + THINK  # what the server appends before each generation

# The student sees the Coach's own system prompt (the teacher's extra rules are *not* included:
# the student learns that behaviour from the examples, which is called context distillation).
STUDENT_SYSTEM = COACH_SYSTEM.strip()

# Shorter tool descriptions than the teacher saw. They're re-read on every step, so each saved
# token is saved thousands of times in training; fine-tuning teaches the details instead.
SHORT = {
    "list_available_data": "What data exists, how much, how fresh; plus today's date, time zone and age.",
    "get_metric": "One metric over a date range (default last 30 days) by day/week/month, with summary stats. "
                  "metric: steps, distance, active_energy, heart_rate, resting_hr, hrv, body_mass, spo2, "
                  "body_temp, bp_systolic, bp_diastolic, or a name like \"VO2 Max\".",
    "compare_periods": "Mean daily value of a metric in period A vs period B.",
    "get_sleep": "Sleep per night (hours, deep/REM/awake, bedtime), averages, nights under 7 h, bedtime spread.",
    "get_workouts": "Workouts by type and week, strength minutes, recent sessions.",
    "longevity_status": "Each research-backed marker: value, target, status, evidence.",
    "get_research": "Citation, finding and link for one marker or all.",
    "remember": "Save context the person mentions (illness, injury, travel, goal, experiment).",
    "recall": "All saved notes. Check before judging a period.",
    "forget": "Delete a note by id.",
    "search_papers": "Search the research papers' full text; returns cited excerpts.",
}


def tool_schemas() -> list[dict[str, Any]]:
    return [{"type": "function", "function": {
        "name": s["name"], "description": SHORT[s["name"]], "parameters": s["input_schema"]}}
        for s in TOOL_SPECS]


def to_hf(messages: list[dict]) -> list[dict]:
    """Our neutral record format -> the message dicts Hugging Face chat templates expect."""
    out = [{"role": "system", "content": STUDENT_SYSTEM}]
    for m in messages:
        if m["role"] == "assistant":
            hf = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                hf["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                    for c in m["tool_calls"]]
            out.append(hf)
        elif m["role"] == "tool":
            out.append({"role": "tool", "name": m["name"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def render(tokenizer, messages: list[dict], add_generation_prompt: bool = False, include_tools: bool = False) -> str:
    """Full prompt text, with the empty think block on *every* assistant turn.

    include_tools=False (the fine-tuned model): no tool schemas in the prompt. Fine-tuning teaches the
    tools, and leaving the schemas out saves ~1,180 tokens on every training example and chat step.
    include_tools=True (the base model): it has never seen these tools, so it needs their definitions.
    """
    text = tokenizer.apply_chat_template(to_hf(messages), tools=tool_schemas() if include_tools else None,
                                         tokenize=False, enable_thinking=False)
    parts = text.split(ASSISTANT_OPEN)
    text = parts[0] + "".join(ASSISTANT_OPEN + (p if p.startswith("<think>") else THINK + p) for p in parts[1:])
    if add_generation_prompt:
        text += GEN_PROMPT
    return text


def encode_for_training(tokenizer, messages: list[dict], include_tools: bool = False) -> tuple[list[int], list[int]]:
    """(input_ids, labels). Labels are -100 everywhere except the assistant's replies (incl. <|im_end|>)."""
    text = render(tokenizer, messages, include_tools=include_tools)
    ids: list[int] = []
    labels: list[int] = []
    pos = 0
    while True:
        start = text.find(GEN_PROMPT, pos)
        if start == -1:
            chunk = tokenizer(text[pos:], add_special_tokens=False).input_ids
            ids += chunk
            labels += [-100] * len(chunk)
            break
        prompt_end = start + len(GEN_PROMPT)
        end = text.index(END, prompt_end) + len(END)
        before = tokenizer(text[pos:prompt_end], add_special_tokens=False).input_ids
        reply = tokenizer(text[prompt_end:end], add_special_tokens=False).input_ids
        ids += before + reply
        labels += [-100] * len(before) + reply
        pos = end
    return ids, labels


def parse_reply(text: str) -> tuple[str, list[dict]]:
    """Split generated text into (visible text, tool calls), reading Qwen's XML tool-call format:

        <tool_call>\\n<function=get_sleep>\\n<parameter=start>\\n2026-11-01\\n</parameter>\\n</function>\\n</tool_call>
    """
    import re

    calls = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S):
        fn = re.search(r"<function=([\w-]+)>", block)
        if not fn:
            continue
        args: dict[str, Any] = {}
        for k, v in re.findall(r"<parameter=([\w-]+)>\n?(.*?)\n?</parameter>", block, re.S):
            v = v.strip()
            try:  # the template writes non-strings (ints, objects) as JSON
                args[k] = json.loads(v) if v[:1] in "[{" or v.lstrip("-").isdigit() else v
            except json.JSONDecodeError:
                args[k] = v
        calls.append({"name": fn.group(1), "arguments": args})
    visible = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S)
    visible = visible.replace(END, "").replace("<think>", "").replace("</think>", "").strip()
    return visible, calls
