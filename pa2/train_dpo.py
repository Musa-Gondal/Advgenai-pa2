"""From-scratch DPO with a hand-written training loop (no TRL / HF Trainer).

The policy and a frozen reference both start from the SFT ``best`` checkpoint.
Each micro-step runs chosen+rejected through the policy in one forward pass and
through the reference under ``no_grad``. Validation tracks DPO loss, reward
accuracy (strict r_w > r_l), and reward margin; the checkpoint with the best
validation reward accuracy is saved as ``best``.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from pa2.batching import build_dpo_batch
from pa2.data import read_jsonl
from pa2.formatting import format_prompt
from pa2.losses import dpo_loss_from_batch
from pa2.train_sft import load_model_and_tokenizer
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

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def to_pref_rows(rows):
    return [(format_prompt(r["instruction"]), r["chosen"], r["rejected"]) for r in rows]


def make_batch(tokenizer, chunk, max_length):
    return build_dpo_batch(
        tokenizer,
        [p for p, _, _ in chunk],
        [c for _, c, _ in chunk],
        [r for _, _, r in chunk],
        max_length,
        pad_to="longest",
    )


@torch.no_grad()
def evaluate_dpo(policy, reference, tokenizer, rows, cfg, device):
    policy.eval()
    losses, chosen, rejected = [], [], []
    for chunk in batches(rows, cfg["eval_batch_size"]):
        batch = make_batch(tokenizer, chunk, cfg["max_length"])
        with autocast_ctx(cfg["precision"], device):
            loss, cr, rr = dpo_loss_from_batch(policy, reference, batch, cfg["beta"])
        losses.append(loss.item() * len(chunk))
        chosen.append(cr.float().cpu())
        rejected.append(rr.float().cpu())
    policy.train()
    chosen, rejected = torch.cat(chosen), torch.cat(rejected)
    return {
        "val_loss": sum(losses) / len(rows),
        "val_reward_accuracy": (chosen > rejected).float().mean().item(),
        "val_reward_margin": (chosen - rejected).mean().item(),
        "val_chosen_reward": chosen.mean().item(),
        "val_rejected_reward": rejected.mean().item(),
    }


def train(cfg, commit_fn=None):
    set_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_device = torch.device(cfg.get("reference_device") or device)
    out_dir = Path(cfg["output_dir"])
    logger = MetricsLogger(out_dir / "metrics.jsonl")

    policy, tokenizer = load_model_and_tokenizer(cfg["sft_checkpoint"], device, torch.float32)
    reference, _ = load_model_and_tokenizer(
        cfg["sft_checkpoint"], ref_device, DTYPES[cfg.get("reference_dtype", "fp32")]
    )
    reference.eval()
    reference.requires_grad_(False)
    if cfg.get("gradient_checkpointing"):
        policy.gradient_checkpointing_enable()
    policy.train()

    train_rows = to_pref_rows(read_jsonl(cfg["train_file"]))[: cfg.get("max_train_examples")]
    val_rows = to_pref_rows(read_jsonl(cfg["val_file"]))[: cfg.get("max_val_examples")]

    micro_bs, accum = cfg["per_device_batch_size"], cfg["gradient_accumulation_steps"]
    micro_steps = math.ceil(len(train_rows) / micro_bs) * cfg["epochs"]
    total_steps = math.ceil(micro_steps / accum)
    warmup_steps = cfg.get("warmup_steps", 0)

    params = param_groups(policy, cfg.get("weight_decay", 0.0))
    if cfg["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(params, lr=cfg["learning_rate"])
    else:
        optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(cfg.get("lr_schedule", "constant"), total_steps, warmup_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["precision"] == "fp16" and device.type == "cuda")
    logger.log(
        event="start", train_pairs=len(train_rows), val_pairs=len(val_rows), total_steps=total_steps,
        effective_batch_size=micro_bs * accum, beta=cfg["beta"], sft_checkpoint=str(cfg["sft_checkpoint"]),
    )

    metrics = evaluate_dpo(policy, reference, tokenizer, val_rows, cfg, device)
    logger.log(event="eval", step=0, **metrics)
    best_acc = -1.0

    step, micro, skipped = 0, 0, 0
    window = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0, "tokens": 0, "start": time.time()}
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(cfg["epochs"]):
        for chunk in batches(train_rows, micro_bs, seed=cfg["seed"] + epoch):
            batch = make_batch(tokenizer, chunk, cfg["max_length"])
            with autocast_ctx(cfg["precision"], device):
                loss, cr, rr = dpo_loss_from_batch(policy, reference, batch, cfg["beta"])
            scaler.scale(loss / accum).backward()
            window["loss"] += loss.item()
            window["acc"] += (cr > rr).float().mean().item()
            window["margin"] += (cr - rr).mean().item()
            window["n"] += 1
            window["tokens"] += int(batch["chosen"]["attention_mask"].sum() + batch["rejected"]["attention_mask"].sum())
            micro += 1
            if micro % accum != 0 and micro != micro_steps:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
            skipped += int(not torch.isfinite(grad_norm))  # GradScaler skips overflow steps
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % cfg["log_every"] == 0:
                n, dt = window["n"], time.time() - window["start"]
                logger.log(
                    event="train", step=step, epoch=epoch, train_loss=window["loss"] / n,
                    train_reward_accuracy=window["acc"] / n, train_reward_margin=window["margin"] / n,
                    grad_norm=float(grad_norm), lr=scheduler.get_last_lr()[0], loss_scale=scaler.get_scale(),
                    skipped_steps=skipped,
                    tokens_per_s=round(window["tokens"] / dt), s_per_step=round(dt / cfg["log_every"], 3),
                    eta_min=round((total_steps - step) * dt / cfg["log_every"] / 60, 1),
                )
                window = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0, "tokens": 0, "start": time.time()}

            if step % cfg["eval_every"] == 0 or step == total_steps:
                metrics = evaluate_dpo(policy, reference, tokenizer, val_rows, cfg, device)
                is_best = metrics["val_reward_accuracy"] > best_acc
                logger.log(event="eval", step=step, best=is_best, **metrics)
                if is_best:
                    best_acc = metrics["val_reward_accuracy"]
                    save_checkpoint(policy, tokenizer, out_dir / "best", {"step": step, **metrics})
                if commit_fn:
                    commit_fn()

    save_checkpoint(policy, tokenizer, out_dir / "final", {"step": step, **metrics})
    logger.log(event="done", step=step, best_val_reward_accuracy=best_acc)
    if commit_fn:
        commit_fn()
    return {"best_val_reward_accuracy": best_acc, "steps": step}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--sft_checkpoint")
    parser.add_argument("--max_train_examples", type=int)
    parser.add_argument("--max_val_examples", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config, {k: v for k, v in vars(args).items() if k != "config"})
    train(cfg)


if __name__ == "__main__":
    main()
