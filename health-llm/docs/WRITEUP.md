# Fine-tuning a small language model on your own health data, on a Mac

*A complete walk-through for someone who has never fine-tuned a language model.*

This document explains how we built a private health Coach: a small open-source language model that
runs entirely on a MacBook (Apple M4 Pro, 24 GB), reads your Apple Health data through tools, cites
the research behind each health target, and answers through a chat endpoint that any OpenAI-compatible
app can talk to. Every number here was measured while building it, including the mistakes.

**The result in one table** (112 held-out questions the model never trained on, same tools, same data):

| | Original Qwen3.5-0.8B | After fine-tuning |
|---|---|---|
| Finished an answer | 66% | **100%** |
| Malformed tool calls | 31% | **2%** |
| Passed every quality check *and* finished | 46% | **93%** |
| Red-flag questions referred to a clinician | 2 of 5 | **5 of 5** |
| Seconds per answer | 10.1 | **5.7** |

"Passed" means the answer finished and nothing in it was invented: no untraceable numbers or unknown
studies, tools used when needed, red flags referred. It does **not** mean the answer was correct. See
"What the 93% doesn't measure" below.

Total cost: about $3 of Claude Haiku time to write the practice conversations, and 8.4 hours of the Mac's GPU
(about 2.7 of them lost to the Mac sleeping and to a Docker VM competing for memory).

---

## Contents

1. [The idea in one picture](#1-the-idea-in-one-picture)
2. [The concepts you need](#2-the-concepts-you-need)
3. [Choosing a model that works on a Mac](#3-choosing-a-model-that-works-on-a-mac)
4. [Making training data](#4-making-training-data)
5. [Turning conversations into training examples](#5-turning-conversations-into-training-examples)
6. [The training loop](#6-the-training-loop)
7. [Did it work? Evaluation](#7-did-it-work-evaluation)
8. [Serving it as a chat endpoint](#8-serving-it-as-a-chat-endpoint)
9. [Privacy](#9-privacy)
10. [Lessons learned](#10-lessons-learned)
11. [Limitations and next steps](#11-limitations-and-next-steps)
12. [Glossary](#12-glossary)

---

## 1. The idea in one picture

```
 ONE-TIME, about $3                   ON YOUR MAC, a few hours            EVERY DAY, free and offline
 ─────────────────────                ────────────────────────            ───────────────────────────
 A strong "teacher" model             A small "student" model             You ask a question
 (Claude Haiku) answers 2,000         (Qwen3.5-0.8B) studies those        → the student picks a tool
 practice questions, using            conversations until it answers      → the tool reads your real data
 fake-but-realistic demo data         the same way                        → the student explains the numbers
```

The starting point was a dashboard that already syncs Apple Health into a Postgres database
(Supabase) and has a chat "Coach" powered by a large cloud model. That Coach works well but every
message goes to an API. The goal: **a Coach that runs on the laptop, costs nothing per message, and
never sends health data anywhere.**

---

## 2. The concepts you need

### What a language model actually does

A language model reads text as **tokens** (word pieces: "sleeping" might be `sleep` + `ing`) and does
one thing: given the tokens so far, it predicts a probability for every possible next token. Writing a
paragraph is just that prediction repeated, one token at a time. Our model's vocabulary has 248,320
possible tokens.

A model's **parameters** (or weights) are the numbers that turn input tokens into those predictions.
"0.8B" means about 0.8 billion of them.

### Base, instruct, and fine-tuned models

* A **base model** is trained on a huge amount of internet text. It can continue text, but doesn't know
  how to be an assistant.
* An **instruct model** has been further trained on conversations, so it follows instructions and chats.
  We start from one of these (Qwen3.5-0.8B).
* **Fine-tuning** continues training on *your* examples so the model picks up a specific behaviour: here,
  "act like the health Coach, use these tools, cite these studies, be brief and direct."

### The most important idea: fine-tuning teaches behaviour, not facts

It is tempting to fine-tune a model on your data table and expect it to "know" your resting heart rate.
That doesn't work well. Small models trained on facts learn the *shape* of the facts and then make up
plausible-looking numbers. So we split the job:

| Job | Who does it | Why |
|---|---|---|
| Knowing your numbers | **Tools** (Python code that queries the database) | Exact, always current |
| Knowing the research | **Retrieval (RAG)**: search the papers, paste excerpts into the conversation | Quotable and checkable |
| Knowing *how* to answer: which tool, how to read its output, how to cite, when to say "see a doctor" | **Fine-tuning** | A learnable behaviour |

### Tools (also called function calling)

The model can write a structured request like `get_sleep(start="2026-11-06", end="2026-11-12")`
instead of an answer. Our code runs it, gets back numbers as JSON, and shows them to the model, which
then writes its answer. The model never computes or remembers your numbers. It only chooses the tool and
explains the result.

We reused the dashboard's ten tested tools (metrics, period comparisons, sleep, workouts, longevity
targets, research, notes) and added one, `search_papers`.

### RAG (Retrieval-Augmented Generation)

We downloaded the 8 studies behind the dashboard's health targets, cut them into 179 overlapping chunks
of about 220 words, and turned each chunk into an **embedding**: a list of 384 numbers that captures its
meaning, so that texts about similar things get similar lists. When the model calls `search_papers("does a
regular bedtime matter?")`, we embed the question the same way and return the 3 most similar chunks.

### LoRA (Low-Rank Adaptation)

Training all 0.8 billion parameters needs memory for the weights, their gradients and the optimizer's
bookkeeping: several times the model's size. LoRA freezes the original weights and adds two small
matrices next to each layer. If a layer's weight is a 1024 × 1024 table, LoRA adds a 1024 × 16 table *B*
and a 16 × 1024 table *A*. Their product *B·A* is added to the frozen weight. Only *A* and *B* are trained:
**10.8 million parameters, 1.4% of the model.** The "16" is the **rank**: higher means more capacity to
change, and more memory.

After training, *B·A* can be added into the original weights permanently ("merging"), so the finished
model runs exactly as fast as the original.

### Distillation

**Distillation** means a strong *teacher* model produces examples and a small *student* model learns to
imitate them. Our teacher is Claude Haiku 5.5; the student is Qwen3.5-0.8B. The student's ceiling is
mostly set by the quality of the examples, which is why we filter them so carefully (section 4).

---

## 3. Choosing a model that works on a Mac

### Apple Silicon basics

* **Unified memory**: the CPU and GPU share one pool of RAM (24 GB here). There's no separate graphics
  memory, but macOS lets the GPU use only part of it by default, about 16 GB in practice before things
  slow down.
* **MPS** (Metal Performance Shaders) is how PyTorch uses the Apple GPU. You move a tensor there with
  `.to("mps")`.
* **No QLoRA**: the popular 4-bit training trick depends on a library (`bitsandbytes`) that only supports
  NVIDIA GPUs, so we train in **bfloat16** (2 bytes per number).

### What we measured

We timed three LoRA training steps on 2,048 random tokens for each candidate:

| Model | Released | Seconds per step | GPU memory | Generation speed |
|---|---|---|---|---|
| Qwen3-0.6B | Apr 2025 | 4.2 | 8.1 GB | 122 tokens/s |
| **Qwen3.5-0.8B** | **Feb 2026** | **5.4** | **10.9 GB** | **86 tokens/s** |
| Qwen3-1.7B | Apr 2025 | 7.5 | 10.9 GB | 76 tokens/s |
| Qwen3.5-2B | Feb 2026 | 8.5 | 13.0 GB | 54 tokens/s |

Then at realistic conversation lengths, which turned out to be the real constraint:

| Model | 4,096 tokens | 6,144 tokens |
|---|---|---|
| Qwen3.5-0.8B | 10.6 s, 8.6 GB | **20.7 s, 13.2 GB** |
| Qwen3-1.7B | 22.7 s, 10.7 GB | 147 s, 18.1 GB (swapping to disk) |
| Qwen3.5-2B | n/a | 50 to 65 s, 16.4 GB |

**We chose Qwen3.5-0.8B**: the newest family, Apache-2.0 licensed, tool calling built in, and the only
candidate that handles our longest conversations without running out of memory.

### Two surprises worth knowing about

**1. "Newest" can mean "slow on your hardware."** Qwen3.5 mixes normal attention layers with a newer
"linear attention" layer (Gated DeltaNet). Its fast implementation is written for NVIDIA GPUs, so on a Mac
PyTorch falls back to a slower reference version. It still works, just with a speed penalty that grows
with sequence length.

**2. A one-line bug made everything look 15× slower.** Our first measurements showed 62 seconds per
step and 22 GB of memory for a 0.6B model, which made no sense. The cause: models load in **evaluation
mode**, and *gradient checkpointing* (a memory-saving trick, section 6) only runs in **training mode**.
Without `model.train()`, every layer's intermediate results were kept, memory overflowed, and macOS
swapped to disk. One line fixed it. **Lesson: when numbers look absurd, suspect your setup first.**

---

## 4. Making training data

### Why synthetic data?

The student needs to learn *how to answer*, not *what your numbers are*. The dashboard has a demo mode
that generates a year of realistic, deliberately messy data (overlapping iPhone and Watch samples, missing
nights). Using it for training means **the teacher never sees a single real health record**, and at chat
time the same tools read your real data instead.

We used 6 different "today" dates for the demo data. Each date reseeds the generator, so the training
conversations see different numbers, seasons and gaps.

### Step 1: questions

Haiku wrote 2,000 practice questions across 13 categories, in varied styles (terse, worried, typo-ridden,
data-nerdy). Cost: **$0.10**.

| Category | Count | Example |
|---|---|---|
| trend | 290 | "did my vo2 max go up over the last season of running" |
| compare | 220 | before/after a habit, this month vs last |
| sleep | 220 | duration, deep sleep, bedtime regularity |
| workouts | 160 | strength minutes, weekly volume |
| status | 150 | "how am I doing overall?" |
| research | 180 | "is 10k steps a myth?" |
| data + research | 180 | "is my VO2 max good enough according to studies?" |
| context to remember | 120 | "I had the flu last week" |
| questions with a saved note | 100 | note: "started the pill 2 months ago" + "why is my HRV lower?" |
| red flags | 100 | "resting HR jumped 20 beats since Thursday" |
| unavailable data | 80 | blood glucose, cholesterol |
| off-topic | 60 | "tell me a fun fact about octopuses" |
| two-turn follow-ups | 140 | a question, then a follow-up that depends on the answer |

### Step 2: the teacher plays the Coach

For each question, Haiku received the dashboard Coach's exact system prompt plus a few extra rules ("only
quote numbers that appear in tool results", "keep answers under 150 words"), and the 11 tools. When it
asked for a tool, our code ran it on demo data and returned the result, until it wrote a final answer.

The extra rules are shown to the **teacher only**. The student never sees them; it learns the behaviour
from the examples. This is called **context distillation**, and it means the student's prompt stays short.

Cost: **$0.0014 per conversation, about $3 for 2,000.** We first ran a 40-conversation pilot to measure
real cost and quality before committing.

### Step 3: the quality filter

The student copies its teacher's mistakes as faithfully as its strengths, so every conversation passes
automatic checks or is thrown away:

| Check | How |
|---|---|
| **Grounded numbers** | Every number in an answer must appear in a tool result, the user's message or the published targets, allowing rounding, unit conversions (hours ↔ minutes) and differences between two shown numbers |
| **Known citations** | "Name et al." must be one of the 8 studies |
| **Used the right tools** | Data questions need a data tool; research questions need the research tools |
| **Red flags** | Worrying symptoms must be referred to a clinician |
| **Remembered context** | Mentions of illness, travel or goals must be saved with `remember` |
| **Length** | Answers under 230 words |

**Result: 1,878 of 2,000 passed (94%).** The top rejection reasons were research questions answered from
data alone (64), invented numbers (14), citations of studies outside our corpus (12) and context not saved (11).

**The checker needed fixing too.** On the pilot, 9 of 11 rejections were the *checker's* mistake: it read
"Lancet Public **Health 2022**" and "**Summer 2025**" as author names. Haiku had cited correctly. Fixing
the pattern took the pilot from 72% to 92%. **Always read your rejections before trusting a filter.**
But some catches were real: Haiku sometimes cited genuine researchers (Cappuccio, Reynolds) from its own
memory, studies that aren't in our corpus and so can't be checked. Those were rightly removed.

**A flaw no automatic check catches:** in one example Haiku attributed Yin 2017's finding ("risk is
lowest around 7 hours") to Windred 2024. The dashboard's rules file lists both studies under one combined
citation, so the teacher blended them. Some errors only a human reading samples will find.

### Data splits

The 1,878 passing conversations were shuffled and split once, saved to disk:

* **train** (1,654): what the model learns from
* **eval** (112): checked during training to decide which checkpoint to keep
* **test** (112): never used until the final report, so the final score is honest

---

## 5. Turning conversations into training examples

### Chat templates

A model doesn't see "messages"; it sees one long string with special markers. Qwen's format looks like:

```
<|im_start|>system
You are Coach, a blunt, funny, evidence-driven health coach...<|im_end|>
<|im_start|>user
[Today is Sunday, July 19, 2026.]

is 7.5 hours sleep optimal research says<|im_end|>
<|im_start|>assistant
<think>

</think>

<tool_call>
<function=search_papers>
<parameter=query>
optimal sleep duration mortality
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"query":"optimal sleep duration mortality","results":[{"cite":"Windred et al., SLEEP 2024", ...
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>

7.5 hours is in the right zone...<|im_end|>
```

### Pitfall 1: train/serve skew

Qwen's template puts the empty `<think></think>` block (the model's optional "reasoning" area, which we
leave empty) only on the newest assistant turn. But at chat time *every* assistant step is generated right
after that block. If training data showed it only on the last turn, the model would learn a slightly
different format from the one it's asked to produce. We wrote **one renderer used for both training and
serving** that puts the block on every assistant turn.

### Pitfall 2: learning the wrong text

By default, a language model is trained to predict *every* token. We only want it to learn the Coach's
replies, not to predict the system prompt, the user's question or the tool results (it should *read* those,
not *write* them). So every non-assistant token gets the label `-100`, which PyTorch's loss function
ignores. In our data, **only 15% of tokens are trained**; the other 85% are context.

### Removing the tool definitions

The 11 tool definitions cost about 1,180 tokens, re-read on every training example and every chat step.
Since fine-tuning teaches the tools, we leave them out of the fine-tuned model's prompt. This cut the
median conversation from 3,377 to 2,307 tokens. The trade-off: adding a new tool later means retraining.
(The untuned base model does get the definitions during evaluation, since it can't work without them.)

---

## 6. The training loop

The full loop is in `src/healthllm/train.py`, written in plain PyTorch so every step is visible.

### One step, in order

```python
for each conversation:
    loss = cross_entropy(model(tokens), assistant_tokens_only)   # how wrong were the predictions?
    (loss / 8).backward()                                         # compute gradients, add them up
    every 8 conversations:
        clip gradients to length 1.0                              # avoid one bad example blowing up
        set the learning rate for this step                       # warm-up, then cosine decay
        optimizer.step()                                          # nudge LoRA's A and B
        optimizer.zero_grad()
    every 40 steps:
        measure loss on the 112 eval conversations; save the adapter if it's the best so far
```

* **Loss** is a measure of surprise: how unlikely the model found the correct next token. Lower is better.
  It's averaged over the trained tokens only.
* **Backward pass**: PyTorch works out, for every trainable number, which direction would reduce the loss.
  That direction is the **gradient**.
* **Gradient accumulation**: we process one conversation at a time (they're too long to batch), but only
  update the weights every 8. Dividing each loss by 8 makes the summed gradient an average, so it behaves
  like a batch of 8.
* **AdamW** is the optimizer: it turns gradients into actual weight changes, adapting the step size per
  parameter.

### Settings and why

| Setting | Value | Why |
|---|---|---|
| LoRA rank / alpha | 16 / 32 | The common starting point; alpha/rank = 2 sets how strongly LoRA's changes count |
| Target layers | all linear layers | Behaviour changes (format, tools, style) benefit from adapting attention *and* feed-forward layers |
| LoRA dropout | 0.05 | Randomly drops a few adapter inputs while training, a mild guard against memorising |
| Learning rate | 2e-4, 5% warm-up, cosine to 10% | LoRA tolerates higher rates than full fine-tuning; warm-up avoids a shock at the start |
| Effective batch | 8 conversations | Smooths noisy gradients from very different conversations |
| Epochs | 2 | Each example seen twice. More risks memorising the teacher's exact phrasing |
| Max length | 6,144 tokens | Fits in memory; skips the ~6% of conversations longer than that |

### Fitting it in memory: five tricks

1. **bfloat16 weights**: half the memory of 32-bit floats, with the same range of values.
2. **LoRA adapter in float32**: the adapter's updates are tiny, and bfloat16 can round them away to zero.
   It's only 10.8M parameters, so keeping it in full precision costs almost nothing.
3. **Gradient checkpointing**: instead of keeping every layer's intermediate results for the backward pass,
   keep a few and recompute the rest when needed. About 30% more compute for far less memory.
4. **Scoring only the trained positions**: the model's last layer turns each position into 248,320 scores,
   one per vocabulary token. At 6,144 tokens that's a 6 GB table, and 85% of it would be thrown away by
   the label mask. Asking for scores only where they're used (`logits_to_keep`) was the difference
   between running and crashing out of memory.
5. **`torch.mps.empty_cache()` after every step**: our first real run slowed to 39 minutes for 46
   conversations, with memory creeping to 22.5 GB. MPS's memory allocator kept freed blocks reserved;
   macOS eventually swapped. Releasing them each step made training about **10× faster** (4.3 s per
   conversation, 472 tokens/s).

   We first guessed the cause was something else (the GPU rebuilding work for every new sequence length)
   and tested a fix for it, padding sequences to fixed sizes. The measurement showed it made things 20%
   *slower*, so we removed it. **Measure fixes; don't assume them.**

### Reading the loss curve

The script measures the loss on 112 held-out conversations every 40 optimizer steps. Here is the real
run (388 steps, 2 epochs; "train" is the average over the 8 conversations of that step, so it's noisy):

| Step | Train loss | Held-out loss | |
|---|---|---|---|
| 0 | | 1.630 | the untouched base model |
| 40 | 1.00 | 1.062 | the format is learned: tool-call syntax, persona, length |
| 80 | 0.90 | 0.981 | |
| 120 | 1.06 | 0.942 | |
| 160 | 0.97 | 0.919 | |
| 200 | 0.71 | 0.894 | end of epoch 1 |
| 240 | 0.68 | 0.894 | epoch 2 starts: the model has seen these chats before |
| 280 | 0.69 | 0.884 | |
| 320 | 0.68 | 0.875 | |
| 360 | 0.69 | 0.869 | |
| 388 | 0.64 | **0.867** | final, and the best: this is the adapter we keep |

How to read it:

* **Most of the learning happens early.** Three quarters of the total improvement came in the first 40 steps
  (about 320 conversations). That's the model picking up *how* to answer. The rest is slow polishing.
* **The gap between the two columns is the thing to watch.** In epoch 2 the training loss drops to
  about 0.68 because the model is seeing each chat a second time, while the held-out loss only creeps
  down. A gap that stays roughly constant is normal. A held-out loss that turns *upward* is
  overfitting: memorizing the teacher's exact wording instead of the skill. That never happened here,
  so the last checkpoint was also the best one. A third epoch would likely buy very little.
* **Loss is not quality.** A loss of 0.87 means "on average, the model gave the teacher's next word a
  probability of about e^-0.87 ≈ 42%". It says nothing about whether a number is right. That's what the
  next section measures.

A practical lesson from this run: the Mac fell asleep partway through and some steps took 30 minutes
instead of 40 seconds. On a laptop, run long jobs under `caffeinate -i -s` (plugged in, lid open).

---

## 7. Did it work? Evaluation

A lower loss means the model imitates the teacher better, but it doesn't tell you whether its answers are
*right*. So we also ran both the original and the fine-tuned model through the full chat loop on the 112
held-out test questions, on the same demo data the teacher saw, and scored the results with the same
quality checks the training data had to pass.

| Metric | Base model | Fine-tuned | What it measures |
|---|---|---|---|
| Finished an answer | 66% | **100%** | answered within 6 tool steps instead of looping |
| Malformed tool calls | 31% | **2%** | share of tool calls the tools rejected (bad syntax or arguments) |
| Same first tool as the teacher | 37% | **68%** | a rough proxy for "asked for the right data" |
| Quality checks passed | 68% | **93%** | the same filter the training data had to pass |
| Passed *and* finished | 46% | **93%** | the headline number (see below) |
| Average answer length | 104 words | 130 words | the Coach targets under about 230 |
| Seconds per answer | 10.1 | **5.7** | fewer wasted tool calls, so faster |

**Why "passed *and* finished".** The quality check looks for bad things in the text: invented numbers,
unknown studies, missed red flags. A model that gives up and writes nothing has no bad things in its
text, so it can "pass". The base model gave up on a third of the questions, which made its plain pass
rate (68%) look much better than it was. Lesson: **check what your metric rewards before you trust it.**

**What the 93% doesn't measure.** The checks look for invented content, not for right answers. There is
no answer key, so "I can't tell from your data" always passes, whether or not the data could have
answered. The fine-tuned model hedges more than its teacher, and some of those hedges are wrong (it once
said it couldn't pull step counts, which it can). The next step for the eval is a correctness grade:
Claude as the judge, comparing each answer to the teacher's reference.

By question type (passed out of total):

| Type | Base | Fine-tuned | | Type | Base | Fine-tuned |
|---|---|---|---|---|---|---|
| compare | 8/14 | **14/14** | | sleep | 4/8 | **8/8** |
| trend | 16/19 | **18/19** | | workouts | 9/11 | **11/11** |
| data + research | 6/12 | **10/12** | | research | 11/13 | 11/13 |
| red flag | 2/5 | **5/5** | | remember context | 0/5 | **4/5** |
| recall notes | 4/6 | **6/6** | | status | 6/7 | **7/7** |
| follow-up | 4/4 | 4/4 | | unavailable data | 4/4 | 4/4 |
| off-topic | 2/4 | 2/4 | | | | |

Two things stand out. **Remember context went from 0 of 5 to 4 of 5**, even though the student's prompt
never says "save what the user tells you". Only the teacher's prompt said that; the student learned it
purely from examples. That is context distillation working. **Red flags went from 2 of 5 to 5 of 5**,
the improvement that matters most for a health tool.

### Patch: "is it improving?" from one night

A reader asked the Coach *"Is my sleep improving? Only use last night's data."* It pulled a whole week and
said "no, not yet". That's two mistakes: it ignored the limit, and one night can't show a trend.

1. **v2.** Haiku wrote 200 conversations for a new `too_little_data` category, and the filter kept the
   ones that said "can't tell yet" and fetched only the asked-for window. Training continued from the
   v1 adapter on 113 of them plus 300 old ones (56 minutes). The caveat appeared, but the answers still
   compared last night with the night before. The cause was in the training data: in 64 of the 141
   kept examples, the teacher had fetched two nights and compared them. The filter checked for the
   caveat, not for whether every claim stayed inside one night.
2. **v3.** The filter now checks every claim: at most one date, no "night before", no spreads or
   regularity. That dropped 75 of the 141 examples. After regenerating with a stricter teacher rule,
   171 clean examples were left for training. Training started again from v1, so v2's habit wasn't
   carried over.

| 29 held-out one-night questions | v1 | v2 | v3 |
|---|---|---|---|
| Passed the stricter check | 0 | 15 | **26** |

**The cost: v3 hedges too much elsewhere.** On the original 112 questions its pass rate held (104), but
answers saying "I can't tell" went from 28 to 52, and about half of the new ones are wrong ("I can't see
the trend from one month"). It learned "add a caveat" more than "add a caveat when the data is short".
The fix is contrastive data: as many examples where there *is* enough data and the right answer is
confident, checked the same way.

**One example, same question, same data.** The user writes: *"last week my resting heart rate jumped
from like 58 to 79 for three days straight. I felt fine mostly, just tired. Is that something I
should worry about?"*

*Base model:* called `list_available_data`, `get_sleep`, `get_metric`, `list_available_data` again,
then `get_research` twice, ran out of steps and answered nothing.

*Fine-tuned:* called `get_metric` and `recall` (to check its notes), then:

> Your data doesn't show a jump. Your resting heart rate has been **55–59 bpm** since Oct 12, averaging
> **56.6 bpm**. The last three days (Nov 6–8) were **56, 55, and 56 bpm**, so nothing there. [...]
> See a clinician if you get chest pain, fainting, dizziness, or a racing or irregular heartbeat. [...]

Every number in that answer is in the tool output. It disagrees with the user when the data disagrees,
and it still gives the safety referral. It isn't perfect, though: the sentence *"A 3-day sustained
jump from 58 to 79 is worth a clinician's look, not a watch"* is clumsy. A 0.8B model writes like one.

---

## 8. Serving it as a chat endpoint

`src/healthllm/server.py` is a small FastAPI app that speaks the OpenAI chat format:

```bash
uv run python -m healthllm.server     # loads the model, connects to your data, listens on 127.0.0.1:8800
```

```bash
curl -N localhost:8800/v1/chat/completions -H 'content-type: application/json' \
  -d '{"stream": true, "messages": [{"role": "user", "content": "How did I sleep this week?"}]}'
```

Because it speaks the OpenAI format, any client works: the `openai` Python package, Open WebUI, or a few
lines of JavaScript in the dashboard.

What happens on each request:

1. Today's date is attached to the message, as the Coach does.
2. The model generates. If it writes a tool call, the server runs the tool against your database (through
   the dashboard's read-only login), adds the result to the conversation, and generates again.
3. When the model writes a plain answer, it's streamed back token by token.

At startup the LoRA adapter is **merged** into the base weights, so chat runs at the base model's speed.

---

## 9. Privacy

* **Training data never touched real health data.** The teacher only saw the dashboard's synthetic demo
  data. The code builds demo settings from an explicit dictionary, so it can't even read the real database
  address.
* **At chat time nothing leaves the Mac.** The model, tools and database connection run locally, and the
  server only listens on `127.0.0.1`.
* **Read-only access.** The database login can only `SELECT`, enforced in three places by the dashboard.
* **Only paper identifiers** were sent to PubMed to download the research.
* `data/`, `models/` and `papers/` are excluded from git.

---

## 10. Lessons learned

1. **Fine-tune for behaviour; use tools and retrieval for facts.**
2. **Test hardware fit before choosing a model.** Five minutes of timing ruled out options that would have
   failed hours into training.
3. **On a Mac, sequence length is the real constraint**, more than parameter count.
4. **When numbers look absurd, suspect your setup.** (`model.train()`, `empty_cache()`.)
5. **Measure your fixes.** Our first theory about the slowdown was wrong.
6. **A cheap teacher plus a strict filter works.** $3 of Haiku, filtered to 94%, rather than $75 of a larger model.
7. **Check your checker.** Most early "failures" were the filter's mistakes.
8. **Use the same code path for training and serving** (one renderer, one agent loop) to avoid skew.
9. **Train only on what the model should write** (label masking).
10. **Read samples yourself.** Some errors (a misattributed study) pass every automatic check.

---

## 11. Limitations and next steps

**What's still weak (measured):**

* **Off-topic and small talk: 2 of 4, no better than before.** "Hey there, how's your day going?" made
  it call four data tools before saying hello. Only 43 of the 1,654 training chats (2.6%) were
  off-topic, so it rarely saw an example of *not* using tools.
* **Research: 11 of 13, the same score, but for better reasons.** One answer was 236 words (the limit is
  230). The other is a fault in our checker, not the model: asked about flexibility, it searched,
  found the papers don't cover it, and said so honestly. The check only counts tool calls that
  *succeed*, so it marked that as "answered without research". Automatic checks need reading too.
* **The checks can't see everything.** They catch an invented number, not a *misinterpreted* one
  ("your sleep improved" when it got worse), and not a real study cited for the wrong claim. During
  generation the teacher once credited a 7-hour sleep finding to the wrong paper and it passed. Keep
  reading samples by hand.
* **On real data it makes date and arithmetic mistakes the checks would miss.** Asked "how many workouts
  last month compared to the month before?", it fetched October and *November* instead of September,
  then gave a session count that was right next to a total and a session split that were invented. On
  "am I on track?" its headline count of markers on target didn't match the list under it. It also
  often calls a sleep average that is above 7 hours "below the 7 h target". Status and sleep lookups were
  accurate. Calendar reasoning ("the month before", "this week") and combining several numbers are
  where a 0.8B model is weakest. A cheap fix is to do that work in code: give the tools a `period`
  argument ("last_month", "previous_month") so the model never computes dates.
* **The test data is synthetic.** All 112 test questions ran on the dashboard's demo data. Your real
  data has gaps, odd units and devices the demo doesn't. Try it on your own questions before trusting it.
* **It is a 0.8B model.** It handles one or two tool calls well. Long multi-step reasoning ("compare my
  last three training blocks and explain the trend against the research") is beyond it.
* **It's not medical advice.** It reads numbers and quotes studies. The red-flag referral is a safety
  net, not a diagnosis.

**Next steps, cheapest first:**

1. **More data for the weak spots.** Generate about 200 more off-topic and small-talk chats (around $0.30)
   and retrain. This is the highest-value change.
2. **Train on your real questions.** Log what you actually ask the local Coach, have the teacher answer
   those on demo data, and add them to training.
3. **A bigger student.** Qwen3.5-2B didn't fit at 6k tokens with this setup. With shorter conversations,
   or on a Mac with more memory, it would reason better.
4. **Prefer facts over fluency.** A second training stage (DPO) on pairs of "grounded answer vs. invented
   answer" would push harder against the one failure the loss can't see.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Token** | A word piece, the unit a model reads and writes |
| **Parameters / weights** | The numbers inside a model that it learns |
| **Instruct model** | A model already trained to follow instructions and chat |
| **Fine-tuning** | Further training on your own examples |
| **LoRA** | Training two small matrices per layer instead of the whole model |
| **Rank (r)** | The inner size of LoRA's matrices: its capacity to change the model |
| **Adapter** | The saved LoRA matrices (here about 40 MB) |
| **Merging** | Adding the adapter into the base weights for full-speed inference |
| **Distillation** | A small model learning from a larger model's examples |
| **Context distillation** | The student learns behaviour that the teacher was told in its prompt |
| **Tool / function calling** | The model requests a function call; code runs it and returns the result |
| **RAG** | Searching documents and putting relevant excerpts into the conversation |
| **Embedding** | A list of numbers representing a text's meaning |
| **Chat template** | The exact text format a model expects for conversations |
| **Train/serve skew** | When training text and chat-time text differ in format |
| **Label masking** | Excluding tokens from the loss (label `-100`) so the model doesn't learn to write them |
| **Loss** | How surprised the model is by the correct next token; lower is better |
| **Gradient** | The direction to change each parameter to reduce the loss |
| **Epoch** | One pass through the training data |
| **Gradient accumulation** | Adding up gradients over several examples before updating |
| **Learning rate** | How big each update step is |
| **Warm-up / cosine decay** | Starting with tiny steps, then gradually shrinking them toward the end |
| **Overfitting** | Memorising the training examples instead of learning the pattern; eval loss rises |
| **bfloat16** | A 16-bit number format with the same range as 32-bit floats |
| **Gradient checkpointing** | Recomputing intermediate results instead of storing them |
| **Logits** | The model's raw scores for every possible next token |
| **MPS** | PyTorch's backend for the Apple Silicon GPU |
| **Unified memory** | CPU and GPU sharing one pool of RAM |
| **Swap** | macOS moving memory to disk when RAM runs out, which is very slow |
