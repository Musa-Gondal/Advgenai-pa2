"""Task 5: TRL DPOTrainer + PEFT/LoRA persona tuning on SmolLM2-360M-Instruct.

This is the only part of the project that uses TRL. Rows from the persona file
are converted to TRL's conversational preference format.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pa2.data import read_json
from pa2.utils import load_config, set_seed


def to_conversational(row):
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def train(cfg, commit_fn=None):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DPOConfig, DPOTrainer

    set_seed(cfg["seed"])
    out_dir = Path(cfg["output_dir"])
    rows = read_json(cfg["data_file"])
    random.Random(cfg["seed"]).shuffle(rows)
    n_val = cfg["val_size"]
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch.float32)

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    args = DPOConfig(
        output_dir=str(out_dir / "trainer"),
        beta=cfg["beta"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_schedule"],
        warmup_ratio=cfg["warmup_ratio"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        per_device_eval_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["epochs"],
        max_length=cfg["max_length"],
        logging_steps=cfg["log_every"],
        eval_strategy="steps",
        eval_steps=cfg["eval_every"],
        save_strategy="no",
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        report_to="none",
        seed=cfg["seed"],
    )

    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    class JsonlLogger(TrainerCallback):
        def on_log(self, _args, state, control, logs=None, **kwargs):
            if logs:
                with metrics_path.open("a") as f:
                    f.write(json.dumps({"step": state.global_step, **logs}) + "\n")

    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list([to_conversational(r) for r in train_rows]),
        eval_dataset=Dataset.from_list([to_conversational(r) for r in val_rows]),
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[JsonlLogger()],
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    final_metrics = trainer.evaluate()
    trainer.model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    with (out_dir / "final_eval.json").open("w") as f:
        json.dump({"train_rows": len(train_rows), "val_rows": len(val_rows), **final_metrics}, f, indent=2)
    if commit_fn:
        commit_fn()
    return final_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_persona.yaml")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    train(load_config(args.config, {"output_dir": args.output_dir}))


if __name__ == "__main__":
    main()
