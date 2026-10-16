"""Supervised fine-tuning with a hand-written training loop (no HF Trainer).

fp16 autocast + GradScaler with fp32 master weights, gradient accumulation,
gradient clipping, cosine schedule with warmup, response-only loss, periodic
validation loss, and ``best``/``final`` checkpoints.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from pa2.batching import build_sft_batch
from pa2.data import read_jsonl
from pa2.formatting import format_prompt
from pa2.losses import batch_response_logprobs, sft_loss_from_batch
from pa2.utils import (
    MetricsLogger,
    autocast_ctx,
    batches,
    load_config,
    lr_lambda,
    param_groups,
    save_checkpoint,
    set_seed,
)


def load_model_and_tokenizer(name_or_path, device, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=dtype).to(device)
    model.config.use_cache = False
    return model, tokenizer


def to_sft_rows(rows):
    return [(format_prompt(r["instruction"]), r["response"]) for r in rows]


@torch.no_grad()
def evaluate_sft(model, tokenizer, rows, cfg, device):
    """Token-level mean NLL over every response token in ``rows``."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for chunk in batches(rows, cfg["eval_batch_size"]):
        batch = build_sft_batch(
            tokenizer, [p for p, _ in chunk], [r for _, r in chunk], cfg["max_length"], pad_to="longest"
        )
        with autocast_ctx(cfg["precision"], device):
            logps, counts = batch_response_logprobs(model, batch)
        total_nll -= logps.sum().item()
        total_tokens += counts.sum().item()
    model.train()
    return total_nll / max(1, total_tokens)


def train(cfg, commit_fn=None):
    set_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["output_dir"])
    logger = MetricsLogger(out_dir / "metrics.jsonl")

    model, tokenizer = load_model_and_tokenizer(cfg["model_name"], device)
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    model.train()

    train_rows = to_sft_rows(read_jsonl(cfg["train_file"]))[: cfg.get("max_train_examples")]
    val_rows = to_sft_rows(read_jsonl(cfg["val_file"]))[: cfg.get("max_val_examples")]

    micro_bs, accum = cfg["per_device_batch_size"], cfg["gradient_accumulation_steps"]
    micro_steps = math.ceil(len(train_rows) / micro_bs) * cfg["epochs"]
    total_steps = math.ceil(micro_steps / accum)
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)

    optimizer = torch.optim.AdamW(
        param_groups(model, cfg["weight_decay"]), lr=cfg["learning_rate"], betas=(0.9, 0.95), eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(cfg["lr_schedule"], total_steps, warmup_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["precision"] == "fp16" and device.type == "cuda")
    logger.log(
        event="start", train_examples=len(train_rows), val_examples=len(val_rows),
        total_steps=total_steps, warmup_steps=warmup_steps, effective_batch_size=micro_bs * accum,
    )

    best_val, step, micro, window_loss, window_n = float("inf"), 0, 0, 0.0, 0
    skipped = 0
    window_tokens, window_start = 0, time.time()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(cfg["epochs"]):
        for chunk in batches(train_rows, micro_bs, seed=cfg["seed"] + epoch):
            batch = build_sft_batch(
                tokenizer, [p for p, _ in chunk], [r for _, r in chunk], cfg["max_length"], pad_to="longest"
            )
            with autocast_ctx(cfg["precision"], device):
                loss = sft_loss_from_batch(model, batch)
            scaler.scale(loss / accum).backward()
            window_loss += loss.item()
            window_n += 1
            window_tokens += int(batch["attention_mask"].sum())
            micro += 1
            if micro % accum != 0 and micro != micro_steps:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            skipped += int(not torch.isfinite(grad_norm))  # GradScaler skips overflow steps
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % cfg["log_every"] == 0:
                dt = time.time() - window_start
                logger.log(
                    event="train", step=step, epoch=epoch, train_loss=window_loss / window_n,
                    grad_norm=float(grad_norm), lr=scheduler.get_last_lr()[0], loss_scale=scaler.get_scale(),
                    skipped_steps=skipped,
                    tokens_per_s=round(window_tokens / dt), s_per_step=round(dt / cfg["log_every"], 3),
                    eta_min=round((total_steps - step) * dt / cfg["log_every"] / 60, 1),
                )
                window_loss, window_n, window_tokens, window_start = 0.0, 0, 0, time.time()

            if step % cfg["eval_every"] == 0 or step == total_steps:
                val_loss = evaluate_sft(model, tokenizer, val_rows, cfg, device)
                is_best = val_loss < best_val
                logger.log(event="eval", step=step, val_loss=val_loss, best=is_best)
                if is_best:
                    best_val = val_loss
                    save_checkpoint(model, tokenizer, out_dir / "best", {"step": step, "val_loss": val_loss})
                if commit_fn:
                    commit_fn()

    save_checkpoint(model, tokenizer, out_dir / "final", {"step": step, "best_val_loss": best_val})
    logger.log(event="done", step=step, best_val_loss=best_val)
    if commit_fn:
        commit_fn()
    return {"best_val_loss": best_val, "steps": step}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--output_dir")
    parser.add_argument("--max_train_examples", type=int)
    parser.add_argument("--max_val_examples", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config, {k: v for k, v in vars(args).items() if k != "config"})
    train(cfg)


if __name__ == "__main__":
    main()
