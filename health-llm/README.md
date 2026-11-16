# health-llm: a local Coach for your Apple Health data

A small open model (Qwen3.5-0.8B), fine-tuned with LoRA in plain PyTorch on an M4 Pro, that answers
questions about your health data the way the dashboard's Claude-powered Coach does: it calls the same
tools against your Supabase data, cites the research behind each target, and runs entirely on your Mac.

The full beginner's write-up is in [`docs/WRITEUP.md`](docs/WRITEUP.md).

```
   ONE-TIME (about $3)                TRAINING (your Mac)                 CHAT (free, offline)
   Haiku 5.5 + demo data  ──▶  2,000 practice chats ──▶ LoRA on Qwen3.5-0.8B ──▶ you ⇄ local Coach ⇄ tools ⇄ Supabase
         (synthetic only)        (quality-filtered)                                    (read-only login)
```

## Pieces

| File | What it does |
|---|---|
| `src/healthllm/bridge.py` | Imports the dashboard's tested tools (`../dashboard`), demo or real data, plus `search_papers` |
| `src/healthllm/papers.py` | Downloads the 8 cited studies from PubMed/PMC and chunks them |
| `src/healthllm/rag.py` | Embeds the chunks (bge-small) and serves `search_papers` |
| `src/healthllm/generate.py` | Haiku writes practice questions, then plays the Coach on demo data |
| `src/healthllm/quality.py` | Rejects chats with invented numbers, unknown studies, missed red flags |
| `src/healthllm/chat_format.py` | One renderer for training and serving; label masking; tool-call parser |
| `src/healthllm/train.py` | The PyTorch LoRA training loop |
| `src/healthllm/agent.py` | The generate → tool → generate loop |
| `src/healthllm/evaluate.py` | Before/after scores on held-out questions |
| `src/healthllm/server.py` | OpenAI-compatible endpoint on `127.0.0.1:8800` |

## Run it

```bash
uv sync                                                    # install (PyTorch, transformers, peft, ...)
ant auth login                                             # only for generating data

uv run python -m healthllm.papers                          # 1. research corpus  -> papers/
uv run python -m healthllm.rag                             #    build the search index
uv run python -m healthllm.generate questions              # 2. practice questions (~$0.10)
uv run python -m healthllm.generate chats                  #    practice chats (~$3, resumable)
uv run python -m healthllm.generate report
uv run python -m healthllm.train --run coach-0.8b          # 3. fine-tune (a few hours)
uv run python -m healthllm.evaluate --base Qwen/Qwen3.5-0.8B --adapter none --name base-0.8b
uv run python -m healthllm.evaluate --base Qwen/Qwen3.5-0.8B --adapter models/coach-0.8b/adapter --name coach-0.8b
uv run python -m healthllm.server                          # 4. chat on your real data
```

Then chat in the terminal (with the server running in another tab):

```bash
uv run coach                       # interactive: /clear to start over, /exit or Ctrl-D to quit
uv run coach "Am I on track?"      # one question, answer, exit
```

Or talk to it with any OpenAI client:

```bash
curl -N localhost:8800/v1/chat/completions -H 'content-type: application/json' \
  -d '{"stream": true, "messages": [{"role": "user", "content": "How did I sleep this week?"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8800/v1", api_key="unused")
print(client.chat.completions.create(model="coach", messages=[{"role": "user", "content": "Am I on track?"}]).choices[0].message.content)
```

Or in Claude's Messages API format, including the official `anthropic` SDK:

```bash
curl -N localhost:8800/v1/messages -H 'content-type: application/json' \
  -d '{"model": "coach", "max_tokens": 400, "stream": true, "messages": [{"role": "user", "content": "Am I on track?"}]}'
```

```python
import anthropic
client = anthropic.Anthropic(base_url="http://127.0.0.1:8800", api_key="unused")
print(client.messages.create(model="coach", max_tokens=400, messages=[{"role": "user", "content": "Am I on track?"}]).content[0].text)
```

`DEMO=1` serves synthetic data instead; `HEALTH_LLM_ADAPTER=none` serves the untuned base model.

## Privacy

* The teacher (Haiku) only ever sees the dashboard's **synthetic** demo data.
* At chat time the model, the tools and your data stay on this Mac. The server binds to `127.0.0.1`,
  and the database login is the dashboard's read-only one.
* `data/`, `models/` and `papers/` are gitignored.
