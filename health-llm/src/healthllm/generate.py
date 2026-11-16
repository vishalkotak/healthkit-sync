"""Generate training conversations with Claude Haiku 5.5 as the teacher, on synthetic data only.

  uv run python -m healthllm.generate questions          # step 1: practice questions (~$0.10)
  uv run python -m healthllm.generate chats --limit 40   # step 2a: a pilot
  uv run python -m healthllm.generate chats              # step 2b: everything left (resumable)
  uv run python -m healthllm.generate report             # pass rate, rejection reasons, cost

How one conversation is made:
  1. Take a practice question and a random demo "today" (each date reseeds the synthetic data).
  2. Haiku plays the dashboard's Coach: same system prompt, same tools (+ search_papers).
  3. Each tool call runs locally on the demo data; the result goes back to Haiku.
  4. The finished conversation is checked by quality.check() and saved either way, with the verdict.

Haiku only ever sees synthetic data. Your real Supabase data is never touched here.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

from . import quality
from .bridge import TOOL_SPECS, Runtime, ToolError, call_tool

from app.coach import SYSTEM as COACH_SYSTEM  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GEN_DIR = ROOT / "data" / "generated"
QUESTIONS_PATH = GEN_DIR / "questions.jsonl"
CONVS_PATH = GEN_DIR / "conversations.jsonl"

TEACHER = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT, PRICE_CACHE_READ, PRICE_CACHE_WRITE = 0.10, 0.50, 0.01, 0.125  # $ per 1M tokens
BUDGET_USD = 10.0  # hard stop, well above the ~$3-6 estimate
MAX_STEPS = 6

# Demo "today" anchors. Different dates -> different synthetic numbers and different seasons.
ANCHORS = ["2026-11-11T18:30:00-08:00", "2026-09-03T08:15:00-07:00", "2026-07-19T21:00:00-07:00",
           "2026-05-02T12:00:00-07:00", "2026-03-15T07:45:00-07:00", "2026-01-27T19:10:00-08:00"]

# Extra guidance for the teacher only. The student never sees this text: it learns the behaviour
# from the examples instead (that's "context distillation").
TEACHER_EXTRA = """
Extra rules for this session:
- Only quote numbers that appear in tool results, in the person's message, or in the targets.
  Round sensibly, but never invent a figure, a percentage, or a date.
- For questions about what the research says, call search_papers (or get_research) and name the study
  exactly as its `cite` field gives it, e.g. "Paluch et al., Lancet Public Health 2022".
- Keep the final answer under 150 words. Plain sentences or a short list.
- If the data needed doesn't exist (check list_available_data), say so plainly. Don't guess.
- A trend needs at least a week of nights or days. If the person limits the window ("only last night",
  "just today"), fetch only that window: "last night" is the single night dated today, so start = end =
  today. Every sentence must stay inside that window: name only that one date, and never mention, compare
  with or contrast against any other night or day, a spread, regularity or an average across days. Give
  what that one night or day shows, then say plainly that it can't tell whether things are improving yet,
  and in a separate final paragraph offer to look at a longer window.
"""
SYSTEM = COACH_SYSTEM + TEACHER_EXTRA

CATEGORIES: dict[str, tuple[int, str]] = {
    "trend": (290, "How a single metric has been going: steps, resting heart rate, HRV, weight, VO2 max, "
                   "active energy, distance, exercise minutes, blood oxygen. Over a week, month, season or year."),
    "compare": (220, "Before/after or period-vs-period comparisons of one metric (this month vs last, since "
                     "starting a new habit, summer vs winter, weekdays vs a holiday week)."),
    "sleep": (220, "Sleep duration, deep/REM sleep, bedtime regularity, short nights, naps."),
    "workouts": (160, "Workouts: how often, what types, strength training minutes, weekly volume, streaks."),
    "status": (150, "Overall 'how am I doing' / longevity-marker / report-card questions."),
    "research": (180, "What the research says and why a target exists, with no request for the person's data "
                      "(why 7 hours of sleep, is 10k steps a myth, how strong is the VO2 max evidence, does "
                      "strength training help longevity, what resting heart rate is ideal)."),
    "data_plus_research": (180, "Questions that need both their numbers and the research (am I in the "
                                "healthy range for X according to studies, is my VO2 max good enough)."),
    "context_remember": (120, "The person mentions context the Coach should save: an illness, injury, travel, "
                              "new job, a goal, a diet or habit experiment, a new device. Sometimes with a "
                              "question attached."),
    "recall_seeded": (100, "Questions about a period that a saved note explains (the note is provided)."),
    "red_flag": (100, "Worrying symptoms or readings: chest pain, fainting, a sudden resting-HR jump, very low "
                      "blood oxygen, palpitations, irregular rhythm alerts, shortness of breath on light effort."),
    "unavailable": (80, "Questions about data the person likely doesn't have (blood glucose, cholesterol, "
                        "ketones, cortisol) or about the future, or impossible dates."),
    "too_little_data": (200, "Asking whether something is improving, getting worse or trending, but limiting "
                             "it to a window too short to show a trend: only last night, just today, "
                             "yesterday, this morning, the past 24 hours, my last workout. Say the limit "
                             "explicitly in most of them."),
    "off_topic": (60, "Chit-chat or requests unrelated to their health data (jokes, coding help, general "
                      "trivia, greetings, thanks)."),
    "followup": (140, "A natural two-message exchange: an opening question about their data, then a follow-up "
                      "that depends on the first answer (drill into a week, ask why, ask what to do)."),
}
STYLES = ["terse, lowercase, no punctuation", "casual and chatty", "worried and a bit anxious",
          "precise and data-nerdy", "with a typo or two", "skeptical", "motivated, goal-focused",
          "very short (under 8 words)", "long-winded with context"]

_cost_lock = threading.Lock()
_spent = {"usd": 0.0, "in": 0, "out": 0}


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(max_retries=6)  # picks up `ant auth login` automatically


def _track(usage) -> None:
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    usd = (usage.input_tokens * PRICE_IN + usage.output_tokens * PRICE_OUT
           + cr * PRICE_CACHE_READ + cw * PRICE_CACHE_WRITE) / 1e6
    with _cost_lock:
        _spent["usd"] += usd
        _spent["in"] += usage.input_tokens + cr + cw
        _spent["out"] += usage.output_tokens
        if _spent["usd"] > BUDGET_USD:
            raise RuntimeError(f"budget of ${BUDGET_USD} reached")


def _text(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text")


# ----------------------------------------------------------------------------- step 1: questions

def gen_questions(seed: int = 7, only: list[str] | None = None) -> None:
    """All categories into a fresh file, or (with `only`) just those categories appended to it."""
    rng = random.Random(seed)
    client = _client()
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for cat, (n, desc) in CATEGORIES.items():
        if only and cat not in only:
            continue
        got: list[Any] = []
        while len(got) < n:
            style = rng.sample(STYLES, 3)
            shape = ('[{"note": "...", "question": "..."}]' if cat == "recall_seeded"
                     else '[["first message", "follow-up"]]' if cat == "followup" else '["...", "..."]')
            prompt = (
                f"Write 25 different messages a person might send to a health coach chatbot that can read their "
                f"Apple Health data (steps, heart rate, HRV, sleep stages, workouts, VO2 max, weight, blood "
                f"oxygen).\n\nCategory: {desc}\nMix these styles: {', '.join(style)}.\n"
                f"Use relative time only (\"last week\", \"since September\", \"this year\"), never a full date "
                f"with a year. Vary the metrics, time ranges and wording; no two should feel alike.\n"
                + ("Each item has a saved note (written by the person earlier, e.g. 'Down with flu, about 10 "
                   "days ago, for 5 days') and a question it helps explain.\n" if cat == "recall_seeded" else "")
                + f"Reply with only a JSON array shaped like {shape}.")
            resp = client.messages.create(model=TEACHER, max_tokens=6000,
                                          messages=[{"role": "user", "content": prompt}])
            _track(resp.usage)
            raw = _text(resp).strip().removeprefix("```json").removesuffix("```").strip()
            try:
                items = json.loads(raw)
            except json.JSONDecodeError:
                continue
            seen = {json.dumps(x) for x in got}
            got += [x for x in items if json.dumps(x) not in seen]
        for x in got[:n]:
            out.append({"category": cat, "item": x})
        print(f"  {cat:20} {min(len(got), n):4} questions   (spent so far ${_spent['usd']:.3f})")
    rng.shuffle(out)
    old = [json.loads(line) for line in QUESTIONS_PATH.read_text().splitlines()] if only else []
    start = max((int(q["id"][1:]) for q in old), default=-1) + 1  # new ids never reuse old ones
    for i, q in enumerate(out, start):
        q["id"] = f"q{i:05d}"
    QUESTIONS_PATH.write_text("".join(json.dumps(q) + "\n" for q in old + out))
    print(f"wrote {len(out)} questions to {QUESTIONS_PATH.relative_to(ROOT)}")


# ----------------------------------------------------------------------------- step 2: chats

_runtimes: dict[str, Runtime] = {}
_rt_lock = threading.Lock()


def _runtime(anchor: str) -> Runtime:
    with _rt_lock:
        if anchor not in _runtimes:
            _runtimes[anchor] = Runtime(demo=True, demo_now=datetime.fromisoformat(anchor))
        return _runtimes[anchor]


def _dump(block) -> dict:
    return block.model_dump(exclude_none=True)


def run_conversation(q: dict, client: anthropic.Anthropic) -> dict:
    rng = random.Random(q["id"])
    anchor = rng.choice(ANCHORS)
    rt = _runtime(anchor)
    today = rt.today()
    item = q["item"]
    if q["category"] == "followup":
        turns = list(item)[:2]
    elif q["category"] == "recall_seeded":
        turns = [item["question"]]
    else:
        turns = [item]

    with tempfile.TemporaryDirectory() as tmp:
        notes_path = Path(tmp) / "notes.json"
        seed_notes = []
        if q["category"] == "recall_seeded":
            seed_notes = [{"id": 1, "saved": today.isoformat(), "note": item["note"]}]
            notes_path.write_text(json.dumps(seed_notes))
        tools = rt.tools(notes_path)

        api_msgs: list[dict] = []
        record: list[dict] = []   # neutral format, used to build the student's training text
        conv = {"id": q["id"], "category": q["category"], "anchor": anchor, "today": today.isoformat(),
                "seed_notes": seed_notes, "messages": record}
        try:
            for turn in turns:
                user = f"[Today is {today:%A, %B %-d, %Y}.]\n\n{turn}"
                api_msgs.append({"role": "user", "content": user})
                record.append({"role": "user", "content": user})
                for step in range(MAX_STEPS):
                    resp = client.messages.create(
                        model=TEACHER, max_tokens=8000, system=SYSTEM, tools=TOOL_SPECS,
                        messages=api_msgs, cache_control={"type": "ephemeral"})
                    _track(resp.usage)
                    if resp.stop_reason == "refusal":
                        raise RuntimeError("teacher refused")
                    if resp.stop_reason == "max_tokens":
                        raise RuntimeError("teacher hit max_tokens")
                    api_msgs.append({"role": "assistant", "content": [_dump(b) for b in resp.content]})
                    calls = [b for b in resp.content if b.type == "tool_use"]
                    record.append({"role": "assistant", "content": _text(resp).strip(),
                                   "tool_calls": [{"name": c.name, "arguments": c.input} for c in calls]})
                    if not calls:
                        break
                    results = []
                    for c in calls:
                        try:
                            out, err = call_tool(tools, c.name, c.input), False
                        except ToolError as e:
                            out, err = {"error": str(e)}, True
                        content = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
                        results.append({"type": "tool_result", "tool_use_id": c.id, "content": content,
                                        **({"is_error": True} if err else {})})
                        record.append({"role": "tool", "name": c.name, "content": content,
                                       **({"is_error": True} if err else {})})
                    api_msgs.append({"role": "user", "content": results})
                else:
                    raise RuntimeError("too many tool steps")
        except RuntimeError as e:
            if "budget" in str(e):
                raise
            conv["error"] = str(e)
    for m in record:  # drop empty tool_calls lists, they only add noise
        if m.get("tool_calls") == []:
            del m["tool_calls"]
    conv["problems"] = quality.check(conv)
    conv["passed"] = not conv["problems"]
    return conv


def gen_chats(limit: int | None, workers: int) -> None:
    questions = [json.loads(line) for line in QUESTIONS_PATH.read_text().splitlines()]
    done = set()
    if CONVS_PATH.exists():
        done = {json.loads(line)["id"] for line in CONVS_PATH.read_text().splitlines()}
    todo = [q for q in questions if q["id"] not in done][: limit or None]
    print(f"{len(done)} already done, generating {len(todo)} with {workers} workers")
    for a in ANCHORS:  # build every demo dataset up front (not thread-safe to race on first use)
        _runtime(a)
    client = _client()
    passed = 0
    write_lock = threading.Lock()
    with ThreadPoolExecutor(workers) as pool, CONVS_PATH.open("a") as f:
        futs = {pool.submit(run_conversation, q, client): q for q in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            conv = fut.result()
            passed += conv["passed"]
            with write_lock:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
                f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  passed {passed}  spent ${_spent['usd']:.3f} "
                      f"(${_spent['usd'] / i:.4f}/chat)", flush=True)


def report() -> None:
    from collections import Counter

    convs = [json.loads(line) for line in CONVS_PATH.read_text().splitlines()]
    ok = sum(c["passed"] for c in convs)
    print(f"{ok}/{len(convs)} passed ({ok / max(1, len(convs)):.0%})")
    by_cat = Counter((c["category"], c["passed"]) for c in convs)
    for cat in CATEGORIES:
        p, f = by_cat[(cat, True)], by_cat[(cat, False)]
        if p + f:
            print(f"  {cat:20} {p:4} / {p + f:<4} passed")
    reasons = Counter(p.split(":")[0] for c in convs for p in c["problems"])
    print("rejection reasons:", dict(reasons.most_common()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["questions", "chats", "report"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="+", choices=list(CATEGORIES), help="questions: add just these categories")
    a = ap.parse_args()
    if a.step == "questions":
        gen_questions(only=a.only)
    elif a.step == "chats":
        gen_chats(a.limit, a.workers)
    else:
        report()
    if a.step != "report":
        print(f"API spend this run: ${_spent['usd']:.3f} ({_spent['in']:,} in / {_spent['out']:,} out tokens)")
