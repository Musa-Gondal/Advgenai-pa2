"""Batched generation on a fixed evaluation set, written as AlpacaEval-style JSON.

Decoding settings come from the config and are recorded in every output row so
SFT and DPO comparisons can be checked to use identical decoding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pa2.data import read_json, write_json
from pa2.formatting import clean_response, format_prompt
from pa2.utils import load_config


def build_prompts(rows, prompt_format, tokenizer):
    if prompt_format == "pa2_chatml":
        return [format_prompt(r["instruction"]) for r in rows]
    if prompt_format == "chat_template":
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": r["instruction"]}], tokenize=False, add_generation_prompt=True
            )
            for r in rows
        ]
    raise ValueError(f"unknown prompt_format {prompt_format!r}")


def decoding_kwargs(cfg):
    kwargs = {"max_new_tokens": cfg["max_new_tokens"], "do_sample": cfg.get("do_sample", False)}
    if kwargs["do_sample"]:
        kwargs.update(temperature=cfg["temperature"], top_p=cfg["top_p"])
    if cfg.get("repetition_penalty", 1.0) != 1.0:
        kwargs["repetition_penalty"] = cfg["repetition_penalty"]
    if cfg.get("no_repeat_ngram_size", 0):
        kwargs["no_repeat_ngram_size"] = cfg["no_repeat_ngram_size"]
    return kwargs


def load_for_generation(model_path, adapter_path=None, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(
    model_path, eval_file, output_file, generator_name, gen_cfg,
    adapter_path=None, limit=None, commit_fn=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_for_generation(model_path, adapter_path, device)
    rows = read_json(eval_file)[:limit]
    prompts = build_prompts(rows, gen_cfg["prompt_format"], tokenizer)

    stop_ids = {tokenizer.eos_token_id}
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
            stop_ids.add(token_id)
    kwargs = decoding_kwargs(gen_cfg)
    torch.manual_seed(gen_cfg.get("seed", 42))

    outputs = []
    bs = gen_cfg["batch_size"]
    for i in range(0, len(rows), bs):
        enc = tokenizer(prompts[i : i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            eos_token_id=sorted(stop_ids), pad_token_id=tokenizer.pad_token_id, **kwargs,
        )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        for row, tokens in zip(rows[i : i + bs], new_tokens):
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            hit_limit = not any(t in stop_ids for t in tokens.tolist())
            outputs.append(
                {
                    "id": row["id"],
                    "instruction": row["instruction"],
                    "output": clean_response(text),
                    "generator": generator_name,
                    "dataset": row.get("dataset"),
                    "category": row.get("category"),
                    "hit_max_new_tokens": hit_limit,
                }
            )
        print(f"{generator_name}: {len(outputs)}/{len(rows)}", flush=True)

    meta = {"generator": generator_name, "model_path": str(model_path), "adapter_path": adapter_path,
            "eval_file": str(eval_file), "decoding": kwargs, "prompt_format": gen_cfg["prompt_format"]}
    write_json(output_file, outputs)
    write_json(Path(output_file).with_suffix(".meta.json"), meta)
    if commit_fn:
        commit_fn()
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generation.yaml")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path")
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--prompt_format")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    gen_cfg = load_config(args.config, {"prompt_format": args.prompt_format})
    generate(args.model_path, args.eval_file, args.output_file, args.generator, gen_cfg,
             adapter_path=args.adapter_path, limit=args.limit)


if __name__ == "__main__":
    main()
