"""Automatic checks on generated conversations. A conversation that fails any check is dropped.

The student model copies its teacher's mistakes as faithfully as its strengths, so filtering
matters more than the teacher's raw quality. Every check here is cheap and deterministic:

  1. grounded_numbers - every number in the Coach's answers must come from a tool result, the
                        user's own words, or the published targets (allowing rounding and simple
                        unit conversions). This catches most invented numbers.
  2. known_citations  - any "Name et al." / "Name 2022" must be one of the eight studies.
  3. used_tools       - data questions must be answered from at least one successful tool call.
  4. red_flag_referral- worrying symptoms must be sent to a clinician.
  5. sane_length      - answers stay short (the Coach's style, and cheaper for a small model).
  6. too_little_data  - a trend question over one night or day must say it can't tell yet, must not
                        fetch days the person excluded, and every claim must stay inside that one night
                        or day: one date at most, no comparison with other days, no spreads or regularity.
"""

from __future__ import annotations

import itertools
import json
import re
from datetime import date
from typing import Any

from .bridge import RULES_PATH

NUM = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b")
AUTHORS = {"Paluch", "Lee", "Momma", "Windred", "Yin", "Zhang", "Kodama", "Mandsager"}
MONTHS = {"January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December", "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep",
          "Sept", "Oct", "Nov", "Dec"}
# "Name et al." is always an author. "Name 2022" is an author only if Name isn't a journal word or season
# ("Lancet Public Health 2022", "JAMA Netw Open 2018", "Summer 2025" are fine).
CITE_ETAL = re.compile(r"\b([A-Z][a-z]+)\s+et\s+al\b")
CITE_YEAR = re.compile(r"\b([A-Z][a-z]+),?\s+\(?(?:19|20)\d{2}\)?")
NOT_AUTHORS = {"Health", "Med", "Open", "Circulation", "Lancet", "Netw", "Assoc", "Heart", "Sports", "Public",
               "Summer", "Winter", "Spring", "Fall", "Autumn", "Christmas", "Easter", "Thanksgiving",
               # ordinary words that can sit before a year ("Your 2026 average", "Across 2025")
               "Since", "In", "Of", "Your", "The", "This", "That", "Sleep", "Across", "Full", "Within", "By",
               "From", "Last", "Through", "Until", "Early", "Late", "Mid", "During", "All", "For", "Over"}
REFERRAL = re.compile(r"doctor|clinician|physician|cardiologist|medical (?:attention|care|help|professional)|"
                      r"\bER\b|emergency|911|urgent care|GP\b|healthcare provider", re.I)
DATA_TOOLS = {"get_metric", "compare_periods", "get_sleep", "get_workouts", "longevity_status",
              "list_available_data"}
RESEARCH_TOOLS = {"search_papers", "get_research", "longevity_status"}
NEEDS_DATA = {"trend", "compare", "sleep", "workouts", "status", "data_plus_research", "recall_seeded",
              "too_little_data"}
NEEDS_RESEARCH = {"research", "data_plus_research"}
MAX_WORDS = 230
CANT_TELL = re.compile(r"can['’]t (?:tell|say|show|judge|call)|too (?:early|soon|little|short)|not enough (?:data|nights|days)|"
                       r"one (?:night|day|reading|workout)(?: alone)? (?:can['’]t|isn['’]t|won['’]t|doesn['’]t)|"
                       r"need (?:at least |a few |more )|no trend (?:yet|from)", re.I)
WINDOW_TOOLS = {"get_metric", "get_sleep", "get_workouts"}
MAX_SHORT_SPAN = 1  # days between start and end (a "past 24 hours" step count can touch two dates)
MONTH_DAY = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.? \d{1,2}\b|\b\d{4}-\d{2}-\d{2}\b")
OTHER_DAYS = re.compile(r"(?:night|day|morning) before|previous (?:night|day)|yesterday['’]s|earlier (?:night|day)s?|"
                        r"bedtime spread|spread of|bedtimes? (?:var|were|are)|regular|consisten|"
                        r"(?:last|past) (?:\d+|few|two|three|seven) (?:nights|days)|vs\.? (?:the )?(?:night|day)", re.I)

_RULES_TEXT = RULES_PATH.read_text()


def numbers_in(text: str) -> list[float]:
    """Numbers in free text, treating '10,430' as one number."""
    out = []
    for m in NUM.finditer(text):
        try:
            out.append(float(m.group().replace(",", "")))
        except ValueError:
            pass
    return out


def _flatten(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x)


def _supported(n: float, pool: list[float]) -> bool:
    a = abs(n)
    if a <= 12 and float(n).is_integer():  # small counts ("3 nights", "2 workouts", "7 days")
        return True
    for t in pool:
        for v in (t, t * 60, t / 60, t * 7, t / 7, t * 100, t / 1000, t * 1000):
            if abs(v - n) <= max(0.051, 0.015 * abs(v)):
                return True
    return False


def grounded_numbers(conv: dict) -> list[str]:
    """Numbers in assistant text that can't be traced to anything the Coach was shown."""
    sources = [m["content"] for m in conv["messages"] if m["role"] in ("user", "tool")]
    sources += [n["note"] for n in conv.get("seed_notes", [])]
    sources += [_RULES_TEXT, conv["today"]]
    base = [n for s in sources for n in numbers_in(_flatten(s))]
    # Differences and sums of pairs cover "up 4.2 bpm" and "6.8 vs 7.4 is 0.6 h less".
    uniq = sorted(set(base))[:400]
    derived = [abs(a - b) for a, b in itertools.combinations(uniq, 2)] + [a + b for a, b in itertools.combinations(uniq[:120], 2)]
    pool = uniq + derived
    bad = []
    for m in conv["messages"]:
        if m["role"] != "assistant" or not m.get("content"):
            continue
        text = CLOCK.sub(" ", DATE.sub(" ", m["content"]))
        for n in numbers_in(text):
            if not _supported(n, pool):
                bad.append(f"{n:g}")
    return bad


def unknown_citations(conv: dict) -> list[str]:
    bad = []
    for m in conv["messages"]:
        if m["role"] == "assistant" and m.get("content"):
            names = CITE_ETAL.findall(m["content"])
            names += [n for n in CITE_YEAR.findall(m["content"]) if n not in NOT_AUTHORS and n not in MONTHS]
            bad += [n for n in names if n not in AUTHORS]
    return bad


def widest_window(conv: dict) -> int:
    """Most days any single data lookup covered (0 if none had a date range)."""
    widest = 0
    for m in conv["messages"]:
        for c in m.get("tool_calls") or []:
            a = c.get("arguments") or {}
            if c["name"] in WINDOW_TOOLS and a.get("start") and a.get("end"):
                try:
                    widest = max(widest, (date.fromisoformat(a["end"][:10]) - date.fromisoformat(a["start"][:10])).days)
                except ValueError:
                    pass
            elif c["name"] == "compare_periods":
                widest = max(widest, 99)  # comparing periods is the opposite of staying in one short window
    return widest


def successful_tools(conv: dict) -> set[str]:
    return {m["name"] for m in conv["messages"] if m["role"] == "tool" and not m.get("is_error")}


def check(conv: dict) -> list[str]:
    """Reasons to reject a conversation (empty list means it passes)."""
    problems = []
    finals = [m for m in conv["messages"] if m["role"] == "assistant" and m.get("content") and not m.get("tool_calls")]
    if not finals or conv.get("error"):
        return [f"incomplete: {conv.get('error', 'no final answer')}"]
    if bad := grounded_numbers(conv):
        problems.append(f"ungrounded numbers: {bad[:6]}")
    if bad := unknown_citations(conv):
        problems.append(f"unknown citations: {bad[:4]}")
    used = successful_tools(conv)
    cat = conv["category"]
    if cat in NEEDS_DATA and not used & DATA_TOOLS:
        problems.append("data question answered without a data tool")
    if cat in NEEDS_RESEARCH and not used & RESEARCH_TOOLS:
        problems.append("research question answered without the research tools")
    if cat == "context_remember" and "remember" not in used:
        problems.append("context mentioned but not saved with remember")
    if cat == "red_flag" and not any(REFERRAL.search(m["content"]) for m in finals):
        problems.append("red flag without a referral to a clinician")
    if cat == "too_little_data":
        if not any(CANT_TELL.search(m["content"]) for m in finals):
            problems.append("short-window trend question without saying it can't tell yet")
        if widest_window(conv) > MAX_SHORT_SPAN:
            problems.append(f"fetched {widest_window(conv)} days for a short-window question")
        for m in finals:
            # the closing offer ("I can pull the last 2 weeks") is allowed to mention a longer window
            body = re.split(r"\n\s*\n(?=[^\n]*\b(?:I can|Want me|If you want|Ask me)\b)", m["content"])[0]
            if len({d.lower() for d in MONTH_DAY.findall(body)}) > 1:
                problems.append("short-window answer names more than one date")
            if hit := OTHER_DAYS.search(body):
                problems.append(f"short-window answer reaches outside the window: {hit.group()!r}")
    if cat == "off_topic" and len(used) > 1:
        problems.append("off-topic message triggered several tool calls")
    longest = max(len(m["content"].split()) for m in finals)
    if longest > MAX_WORDS:
        problems.append(f"answer too long ({longest} words)")
    return problems
