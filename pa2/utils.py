"""Shared helpers: config loading, seeding, metrics logging, LR schedules, checkpoints."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MetricsLogger:
    """Append one JSON object per line so partial progress survives interruptions."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.time()

    def log(self, **record):
        record.setdefault("elapsed_s", round(time.time() - self.start, 1))
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def lr_lambda(schedule, total_steps, warmup_steps):
    """Multiplier on the peak LR for optimizer step ``step`` (0-based)."""

    def fn(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        if schedule == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        raise ValueError(f"unknown schedule {schedule!r}")

    return fn


def param_groups(model, weight_decay):
    """Apply weight decay to matrices only, not to biases or norm weights."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def autocast_ctx(precision, device):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def save_checkpoint(model, tokenizer, path, extra=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    if extra is not None:
        with (path / "checkpoint_info.json").open("w") as f:
            json.dump(extra, f, indent=2)


def batches(rows, batch_size, seed=None):
    order = list(range(len(rows)))
    if seed is not None:
        random.Random(seed).shuffle(order)
    for i in range(0, len(order), batch_size):
        yield [rows[j] for j in order[i : i + batch_size]]


def env_report():
    import transformers

    info = {
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    try:
        import datasets

        info["datasets"] = datasets.__version__
    except ImportError:
        pass
    for name in ("trl", "peft"):
        try:
            info[name] = __import__(name).__version__
        except ImportError:
            pass
    info["hostname"] = os.uname().nodename
    return info
