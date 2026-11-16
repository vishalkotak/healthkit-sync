"""LoRA fine-tuning in plain PyTorch on Apple Silicon (MPS).

  uv run python -m healthllm.train --model Qwen/Qwen3.5-0.8B --run smoke --max-examples 64 --epochs 1
  uv run python -m healthllm.train --model Qwen/Qwen3.5-2B   --run coach-2b
  uv run python -m healthllm.train --run coach-0.8b-v2 --init-adapter models/coach-0.8b/adapter \
      --focus too_little_data --replay 400 --epochs 1 --lr 1e-4      # patch one weakness

The loop, step by step:
  1. Load the base model in bfloat16 and freeze it.
  2. Wrap every linear layer with a LoRA adapter: two small matrices A (r x in) and B (out x r) whose
     product is added to the frozen weight. Only A and B are trained (~1% of the parameters).
  3. For each conversation: forward pass -> cross-entropy on the assistant tokens only -> backward.
  4. Accumulate gradients over several conversations (an "effective batch"), clip them, take an AdamW
     step. Learning rate: linear warm-up, then cosine decay to 10%.
  5. Every so often, measure loss on held-out conversations the model never trains on. Keep the
     adapter with the lowest held-out loss (later epochs can overfit).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .chat_format import encode_for_training

ROOT = Path(__file__).resolve().parents[2]
CONVS_PATH = ROOT / "data" / "generated" / "conversations.jsonl"
SPLIT_PATH = ROOT / "data" / "generated" / "split.json"
RUNS_DIR = ROOT / "models"


def load_split(seed: int = 0, eval_frac: float = 0.06, test_frac: float = 0.06) -> dict[str, list[dict]]:
    """Passed conversations, split into train / eval (for checkpoint choice) / test (for the final report).

    The split is saved, so every run and the evaluation script use exactly the same test set.
    Conversations generated later are added to it without moving any existing one; they get a larger
    test share (15%), because each new batch targets a specific weakness we want to measure.
    """
    convs = [c for c in map(json.loads, CONVS_PATH.read_text().splitlines()) if c["passed"]]
    by_id = {c["id"]: c for c in convs}
    if SPLIT_PATH.exists():
        ids = json.loads(SPLIT_PATH.read_text())
    else:
        order = sorted(by_id)
        random.Random(seed).shuffle(order)
        n_ev, n_te = int(len(order) * eval_frac), int(len(order) * test_frac)
        ids = {"test": order[:n_te], "eval": order[n_te:n_te + n_ev], "train": order[n_te + n_ev:]}
        SPLIT_PATH.write_text(json.dumps(ids))
    known = {i for v in ids.values() for i in v}
    new = sorted(i for i in by_id if i not in known)
    if new:
        random.Random(seed).shuffle(new)
        n_te, n_ev = round(len(new) * 0.15), round(len(new) * 0.05)
        ids["test"] += new[:n_te]
        ids["eval"] += new[n_te:n_te + n_ev]
        ids["train"] += new[n_te + n_ev:]
        SPLIT_PATH.write_text(json.dumps(ids))
    return {k: [by_id[i] for i in v if i in by_id] for k, v in ids.items()}


def encode_all(tok, convs: list[dict], max_len: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out, skipped = [], 0
    for c in convs:
        ids, labels = encode_for_training(tok, c["messages"])
        if len(ids) > max_len:
            skipped += 1
            continue
        out.append((torch.tensor(ids), torch.tensor(labels)))
    if skipped:
        print(f"  skipped {skipped} conversations longer than {max_len} tokens")
    return out


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def masked_loss(model, ids: torch.Tensor, labels: torch.Tensor, device) -> tuple[torch.Tensor, int]:
    """Cross-entropy on the assistant tokens only, computing logits only where they're needed.

    The logits at position t predict token t+1. Only ~10% of positions are followed by a trained token,
    so we ask the model for logits at just those positions (logits_to_keep). The usual approach builds a
    [seq_len x 248k-vocab] fp32 tensor (~6 GB at 6k tokens) and throws 90% of it away.
    (We tried padding sequences to fixed "bucket" lengths too: on MPS it was ~20% slower, so we don't.)
    """
    ids, labels = ids.to(device), labels.to(device)
    keep = (labels[1:] != -100).nonzero().squeeze(1)
    logits = model(input_ids=ids[None], logits_to_keep=keep).logits[0]
    return F.cross_entropy(logits.float(), labels[keep + 1]), len(keep)


@torch.no_grad()
def eval_loss(model, data, device) -> float:
    """Mean loss per assistant token over held-out conversations."""
    model.eval()
    tot, n = 0.0, 0
    for ids, labels in data:
        loss, k = masked_loss(model, ids, labels, device)
        tot += loss.item() * k
        n += k
    torch.mps.empty_cache()
    model.train()
    return tot / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--run", default="coach")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--accum", type=int, default=8, help="conversations per optimizer step")
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--eval-every", type=int, default=40, help="optimizer steps between evaluations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-adapter", help="continue training this adapter instead of starting a new one")
    ap.add_argument("--focus", nargs="+", help="train on all of these categories...")
    ap.add_argument("--replay", type=int, default=400, help="...plus this many other conversations, so old skills stay")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    run_dir = RUNS_DIR / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(a), indent=1))
    log = (run_dir / "log.jsonl").open("w")

    tok = AutoTokenizer.from_pretrained(a.model)
    split = load_split()
    train_convs = split["train"][: a.max_examples] if a.max_examples else split["train"]
    if a.focus:
        rest = [c for c in train_convs if c["category"] not in a.focus]
        train_convs = ([c for c in train_convs if c["category"] in a.focus]
                       + random.Random(a.seed).sample(rest, min(a.replay, len(rest))))
        print(f"focus run: {len(train_convs) - min(a.replay, len(rest))} {'/'.join(a.focus)} + {min(a.replay, len(rest))} replayed")
    train = encode_all(tok, train_convs, a.max_len)
    evals = encode_all(tok, split["eval"][: max(16, (a.max_examples or 0) // 4) if a.max_examples else None], a.max_len)
    n_tok = sum(len(x) for x, _ in train)
    n_lab = sum(int((y != -100).sum()) for _, y in train)
    print(f"train: {len(train)} conversations, {n_tok:,} tokens ({n_lab:,} trained = {n_lab / n_tok:.0%}) | "
          f"eval: {len(evals)}")

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False  # the KV cache is for generation; in training it only costs memory
    if a.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        lora = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules="all-linear",
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
    for p in model.parameters():  # keep the trainable adapter in fp32: tiny updates survive rounding
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to(device)
    model.train()  # from_pretrained returns eval mode; gradient checkpointing only runs in train mode

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
    total_steps = max(1, int(len(train) * a.epochs) // a.accum)
    warmup = max(1, int(0.05 * total_steps))
    print(f"{total_steps} optimizer steps ({a.epochs} epochs x {len(train)} / accum {a.accum}), warm-up {warmup}")

    best = eval_loss(model, evals, device)
    print(f"step 0 | eval loss {best:.4f} ({'the starting adapter' if a.init_adapter else 'the untrained base model'})")
    log.write(json.dumps({"step": 0, "eval_loss": best}) + "\n")
    best = float("inf")

    order: list[int] = []
    step, micro, t0, run_loss, run_n = 0, 0, time.time(), 0.0, 0
    while step < total_steps:
        if not order:
            order = list(range(len(train)))
            random.shuffle(order)
        ids, labels = train[order.pop()]
        loss, _ = masked_loss(model, ids, labels, device)
        (loss / a.accum).backward()  # dividing makes the accumulated gradient an average
        run_loss += loss.item()
        run_n += 1
        micro += 1
        if micro % a.accum:
            continue

        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, a.lr, warmup)
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.mps.empty_cache()  # hand freed blocks back, so the allocator's cache can't creep into swap
        step += 1

        rec = {"step": step, "train_loss": run_loss / run_n, "lr": opt.param_groups[0]["lr"], "grad_norm": gnorm,
               "sec": round(time.time() - t0, 1), "mps_gb": round(torch.mps.driver_allocated_memory() / 1e9, 1)}
        run_loss, run_n = 0.0, 0
        if step % a.eval_every == 0 or step == total_steps:
            rec["eval_loss"] = eval_loss(model, evals, device)
            if rec["eval_loss"] < best:
                best = rec["eval_loss"]
                model.save_pretrained(run_dir / "adapter")
                rec["saved"] = True
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 5 == 0 or "eval_loss" in rec:
            eta = (time.time() - t0) / step * (total_steps - step) / 60
            print(f"step {step}/{total_steps} | train {rec['train_loss']:.4f} | "
                  + (f"eval {rec['eval_loss']:.4f}{' *saved*' if rec.get('saved') else ''} | " if 'eval_loss' in rec else "")
                  + f"lr {rec['lr']:.2e} | {rec['mps_gb']} GB | eta {eta:.0f} min", flush=True)

    print(f"done in {(time.time() - t0) / 60:.1f} min | best eval loss {best:.4f} | adapter: {run_dir / 'adapter'}")


if __name__ == "__main__":
    main()
