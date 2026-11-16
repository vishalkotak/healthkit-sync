"""A terminal chat with the local Coach, in the style of Claude's CLI.

  uv run coach                         # interactive chat (keeps the conversation until you quit)
  uv run coach "Am I on track?"        # one question, print the answer, exit
  echo "How did I sleep?" | uv run coach

It talks to the running server (`uv run python -m healthllm.server`) through the Claude-style
/v1/messages endpoint, using the same `anthropic` SDK code you'd write for Claude itself.
Commands inside the chat: /clear (start over), /exit (or Ctrl-D).
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import threading

import anthropic

URL = os.environ.get("COACH_URL", "http://127.0.0.1:8800")
TTY = sys.stdout.isatty()
DIM, BOLD, ORANGE, RESET = ("\033[2m", "\033[1m", "\033[38;5;173m", "\033[0m") if TTY else ("",) * 4


class Spinner:
    """A small "thinking" indicator while the model calls tools, before the first word arrives."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        for frame in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if self._stop.wait(0.1):
                break
            sys.stdout.write(f"\r{DIM}{frame} thinking…{RESET}")
            sys.stdout.flush()

    def __enter__(self) -> Spinner:
        if TTY:
            self._thread.start()
        return self

    def stop(self) -> None:
        if not self._stop.is_set():
            self._stop.set()
            if TTY:
                self._thread.join()
                sys.stdout.write("\r\033[K")  # erase the spinner line

    def __exit__(self, *exc) -> None:
        self.stop()


def ask(client: anthropic.Anthropic, history: list[dict]) -> str:
    """Stream one answer to stdout and return its full text."""
    text = ""
    with Spinner() as spin, client.messages.stream(model="coach", max_tokens=600, messages=history) as stream:
        for chunk in stream.text_stream:
            spin.stop()
            sys.stdout.write(chunk)
            sys.stdout.flush()
            text += chunk
    print()
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with your local health Coach.")
    ap.add_argument("question", nargs="*", help="ask one question and exit")
    a = ap.parse_args()

    client = anthropic.Anthropic(base_url=URL, api_key="unused", timeout=300, max_retries=0)
    one_shot = " ".join(a.question) or (None if sys.stdin.isatty() else sys.stdin.read().strip())

    try:
        if one_shot:
            ask(client, [{"role": "user", "content": one_shot}])
            return

        print(f"{ORANGE}{BOLD}✻ Coach{RESET} {DIM}local Qwen3.5-0.8B + your health data · /clear to start over, /exit to quit{RESET}\n")
        history: list[dict] = []
        while True:
            try:
                q = input(f"{BOLD}> {RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not q:
                continue
            if q in ("/exit", "/quit"):
                return
            if q == "/clear":
                history.clear()
                print(f"{DIM}(conversation cleared){RESET}\n")
                continue
            history.append({"role": "user", "content": q})
            print()
            try:
                history.append({"role": "assistant", "content": ask(client, history)})
            except KeyboardInterrupt:  # Ctrl-C stops this answer, not the chat
                history.pop()
                print(f"\n{DIM}(interrupted){RESET}")
            print()
    except anthropic.APIConnectionError:
        sys.exit(f"Can't reach the Coach at {URL}. Start it with:  uv run python -m healthllm.server")


if __name__ == "__main__":
    main()
