"""Load, filter, and save the fixed training/validation/evaluation subsets.

Every subset is drawn the same way: shuffle the source split with a fixed seed,
walk it in order, keep rows that pass the single-turn and length filters, stop
at the target count. Length filters use the training tokenizer and prompt
format so that kept examples fit in the context without cutting off EOS.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from pa2.batching import tokenize_pair
from pa2.formatting import format_prompt

HH_HUMAN = "\n\nHuman:"
HH_ASSISTANT = "\n\nAssistant:"


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path):
    with Path(path).open() as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class LengthFilter:
    """Keep rows whose formatted prompt and full sequence fit the token budgets."""

    def __init__(self, tokenizer, max_length, max_prompt_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def prompt_ok(self, instruction):
        n = len(self.tokenizer.encode(format_prompt(instruction), add_special_tokens=False))
        return n <= self.max_prompt_length

    def pair_ok(self, instruction, response):
        # tokenize_pair appends EOS; untruncated length must fit max_length.
        ids, _ = tokenize_pair(self.tokenizer, format_prompt(instruction), response, 10**9)
        return len(ids) <= self.max_length


def _take(dataset, n, seed, convert, keep, strict=True):
    """Shuffle ``dataset`` with ``seed`` and collect ``n`` converted rows passing ``keep``.

    With ``strict=False`` a split that runs out keeps every passing row instead of
    raising (the manual asks for 1,000 validation pairs "when available").
    """
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    out, seen = [], 0
    for idx in order:
        seen += 1
        row = convert(dataset[idx])
        if row is None or not keep(row):
            continue
        row["source_index"] = idx
        out.append(row)
        if len(out) == n:
            break
    if len(out) < n and strict:
        raise RuntimeError(f"only {len(out)} of {n} rows passed filters")
    return out, seen


# ---------------------------------------------------------------- SFT (UltraChat)


def ultrachat_to_sft(example):
    """First user turn and first assistant reply of an UltraChat conversation."""
    messages = example["messages"]
    if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    instruction = messages[0]["content"].strip()
    response = messages[1]["content"].strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "response": response}


# ---------------------------------------------------------------- HH-RLHF


def split_hh(text):
    """Split a single-turn HH transcript into (instruction, response); None if multi-turn."""
    if text.count(HH_HUMAN) != 1 or text.count(HH_ASSISTANT) != 1:
        return None
    head, response = text.split(HH_ASSISTANT)
    if not head.startswith(HH_HUMAN):
        return None
    instruction = head[len(HH_HUMAN) :].strip()
    return instruction, response.strip()


def hh_to_pair(example):
    chosen = split_hh(example["chosen"])
    rejected = split_hh(example["rejected"])
    if chosen is None or rejected is None or chosen[0] != rejected[0]:
        return None
    instruction = chosen[0]
    if not instruction or not chosen[1] or not rejected[1] or chosen[1] == rejected[1]:
        return None
    return {"instruction": instruction, "chosen": chosen[1], "rejected": rejected[1]}


# ---------------------------------------------------------------- UltraFeedback


def ultrafeedback_to_pair(example):
    chosen, rejected = example["chosen"], example["rejected"]
    if len(chosen) != 2 or len(rejected) != 2:
        return None
    if chosen[1]["role"] != "assistant" or rejected[1]["role"] != "assistant":
        return None
    if example["score_chosen"] <= example["score_rejected"]:
        return None
    instruction = example["prompt"].strip()
    c, r = chosen[1]["content"].strip(), rejected[1]["content"].strip()
    if not instruction or not c or not r or c == r:
        return None
    return {
        "instruction": instruction,
        "chosen": c,
        "rejected": r,
        "score_chosen": example["score_chosen"],
        "score_rejected": example["score_rejected"],
    }


# ---------------------------------------------------------------- AlpacaEval


def alpaca_eval_subset(n=300, seed=42):
    """Fixed AlpacaEval subset: ``random.Random(seed).sample`` of row indices."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    rows = read_json(path)
    indices = random.Random(seed).sample(range(len(rows)), n)
    return [
        {
            "id": f"alpaca-eval-{seed}-{i:04d}",
            "instruction": rows[idx]["instruction"],
            "dataset": "alpaca_eval",
            "source_dataset": rows[idx].get("dataset"),
            "source_index": idx,
        }
        for i, idx in enumerate(indices)
    ]


# ---------------------------------------------------------------- entry point


def prepare_all(tokenizer, out_dir, cfg):
    """Build every training/validation subset and write them plus a manifest to ``out_dir``."""
    from datasets import load_dataset

    out_dir = Path(out_dir)
    seed = cfg.get("seed", 42)
    max_length = cfg.get("max_length", 512)
    max_prompt_length = cfg.get("max_prompt_length", 256)
    lf = LengthFilter(tokenizer, max_length, max_prompt_length)

    def sft_keep(row):
        return lf.prompt_ok(row["instruction"]) and lf.pair_ok(row["instruction"], row["response"])

    def pref_keep(row):
        return (
            lf.prompt_ok(row["instruction"])
            and lf.pair_ok(row["instruction"], row["chosen"])
            and lf.pair_ok(row["instruction"], row["rejected"])
        )

    # (name, repo, split, target rows, converter, filter, strict): validation splits may
    # come up short (UltraFeedback test_prefs has fewer than 1,000 pairs that fit 512 tokens).
    specs = [
        ("sft_train", "HuggingFaceH4/ultrachat_200k", "train_sft", cfg["sft_train"], ultrachat_to_sft, sft_keep, True),
        ("sft_val", "HuggingFaceH4/ultrachat_200k", "test_sft", cfg["sft_val"], ultrachat_to_sft, sft_keep, False),
        ("hh_train", "Anthropic/hh-rlhf", "train", cfg["dpo_train"], hh_to_pair, pref_keep, True),
        ("hh_val", "Anthropic/hh-rlhf", "test", cfg["dpo_val"], hh_to_pair, pref_keep, False),
        ("uf_train", "HuggingFaceH4/ultrafeedback_binarized", "train_prefs", cfg["dpo_train"], ultrafeedback_to_pair, pref_keep, True),
        ("uf_val", "HuggingFaceH4/ultrafeedback_binarized", "test_prefs", cfg["dpo_val"], ultrafeedback_to_pair, pref_keep, False),
    ]
    manifest = {"seed": seed, "max_length": max_length, "max_prompt_length": max_prompt_length, "subsets": {}}
    for name, repo, split, n, convert, keep, strict in specs:
        dataset = load_dataset(repo, split=split)
        rows, scanned = _take(dataset, n, seed, convert, keep, strict)
        write_jsonl(out_dir / f"{name}.jsonl", rows)
        manifest["subsets"][name] = {
            "source": f"{repo}:{split}",
            "source_rows": len(dataset),
            "rows_scanned": scanned,
            "rows_target": n,
            "rows_kept": len(rows),
        }
        print(f"{name}: kept {len(rows)} after scanning {scanned} of {len(dataset)}", flush=True)
    write_json(out_dir / "manifest.json", manifest)
    return manifest
