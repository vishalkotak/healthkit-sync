"""Before/after evaluation on held-out questions the model never trained on.

  uv run python -m healthllm.evaluate --base Qwen/Qwen3.5-2B --adapter none           --name base-2b
  uv run python -m healthllm.evaluate --base Qwen/Qwen3.5-2B --adapter models/coach-2b/adapter --name coach-2b

For each test question we replay the user's turn(s) through the same agent loop the server uses, on
the same synthetic data the teacher saw, then score the result:

  * pass rate      - the same quality.check() the training data had to pass (grounded numbers, real
                     citations, tools used when needed, clinician referral for red flags, length)
  * tool match     - did it call the same first tool as the teacher? (a proxy for "right tool")
  * tool errors    - share of tool calls rejected as malformed
  * finished       - answered within the step limit
  * words, seconds - answer length and wall-clock time per conversation
  * hedged         - share of answers that say they can't tell, per category. It should be high for
                     too_little_data and low for ordinary trend questions (over-hedging is a failure too)
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from . import agent, quality
from .bridge import Runtime
from .train import load_split

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "eval"


def first_tool(messages: list[dict]) -> str | None:
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            return m["tool_calls"][0]["name"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--category", nargs="+", help="only test questions in these categories")
    a = ap.parse_args()

    test = [c for c in load_split()["test"] if not a.category or c["category"] in a.category][: a.limit]
    model, tok = agent.load(a.base, a.adapter)
    base_model = a.adapter.lower() == "none"  # the untuned model needs the tool schemas in its prompt
    runtimes: dict[str, Runtime] = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for i, ref in enumerate(test, 1):
        rt = runtimes.setdefault(ref["anchor"], Runtime(demo=True, demo_now=datetime.fromisoformat(ref["anchor"])))
        with tempfile.TemporaryDirectory() as tmp:
            notes = Path(tmp) / "notes.json"
            if ref.get("seed_notes"):
                notes.write_text(json.dumps(ref["seed_notes"]))
            tools = rt.tools(notes)
            msgs: list[dict] = []
            t0, n_calls, n_err, finished = time.time(), 0, 0, True
            for user in [m["content"] for m in ref["messages"] if m["role"] == "user"]:
                msgs.append({"role": "user", "content": user})
                for ev in agent.run(model, tok, tools, msgs, temperature=0.0, include_tools=base_model):
                    if ev["type"] == "tool":
                        n_calls += 1
                        n_err += ev["error"] is not None
                    if ev["type"] == "done" and ev.get("gave_up"):
                        finished = False
        conv = {**ref, "messages": msgs}
        conv.pop("error", None)
        problems = quality.check(conv)
        finals = [m for m in msgs if m["role"] == "assistant" and not m.get("tool_calls")]
        results.append({
            "id": ref["id"], "category": ref["category"], "passed": not problems, "problems": problems,
            "tool_match": first_tool(msgs) == first_tool(ref["messages"]),
            "calls": n_calls, "tool_errors": n_err, "finished": finished,
            "words": sum(len(m["content"].split()) for m in finals) / max(1, len(finals)),
            "seconds": round(time.time() - t0, 1), "messages": msgs,
            "hedged": any(quality.CANT_TELL.search(m["content"]) for m in finals),
        })
        if i % 10 == 0 or i == len(test):
            print(f"  {i}/{len(test)}  pass so far {sum(r['passed'] for r in results) / i:.0%}", flush=True)

    out = OUT_DIR / f"{a.name}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
    n = len(results)
    calls = sum(r["calls"] for r in results)
    summary = {
        "name": a.name, "n": n,
        "pass_rate": sum(r["passed"] for r in results) / n,
        "tool_match": sum(r["tool_match"] for r in results) / n,
        "tool_error_rate": sum(r["tool_errors"] for r in results) / max(1, calls),
        "finished": sum(r["finished"] for r in results) / n,
        "avg_words": sum(r["words"] for r in results) / n,
        "avg_seconds": sum(r["seconds"] for r in results) / n,
        "pass_by_category": {c: f"{sum(r['passed'] for r in results if r['category'] == c)}/{k}"
                             for c, k in Counter(r["category"] for r in results).items()},
        "hedged_by_category": {c: f"{sum(r['hedged'] for r in results if r['category'] == c)}/{k}"
                               for c, k in Counter(r["category"] for r in results).items()},
        "top_problems": Counter(p.split(":")[0] for r in results for p in r["problems"]).most_common(6),
    }
    (OUT_DIR / f"{a.name}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
