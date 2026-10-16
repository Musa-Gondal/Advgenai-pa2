"""End-to-end CPU run of the SFT/DPO loops, generation, pairing, and judging.

Uses a tiny randomly initialised Qwen2 model and a character-level tokenizer
built on the fly, so it needs no downloads or GPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parents[1]


def make_tiny_model(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?:;'\n_-|<>")
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    vocab = {tok: i for i, tok in enumerate(specials + chars)}
    vocab["[UNK]"] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|endoftext|>", unk_token="[UNK]",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    config = Qwen2Config(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        eos_token_id=0, pad_token_id=0, tie_word_embeddings=True,
    )
    Qwen2ForCausalLM(config).save_pretrained(path)
    tokenizer.save_pretrained(path)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("pipe")
    make_tiny_model(d / "tiny")
    sft = [{"instruction": f"say {w}", "response": f"{w} {w}."} for w in ["hi", "yo", "ok", "no", "go", "up"]]
    write_jsonl(d / "sft_train.jsonl", sft)
    write_jsonl(d / "sft_val.jsonl", sft[:2])
    pref = [{"instruction": f"pick {w}", "chosen": f"{w}!", "rejected": "meh"} for w in ["a", "b", "c", "d"]]
    write_jsonl(d / "pref_train.jsonl", pref)
    write_jsonl(d / "pref_val.jsonl", pref[:2])
    evalset = [{"id": f"e{i}", "instruction": f"say {w}", "dataset": "t", "category": "c"} for i, w in enumerate("xyz")]
    (d / "eval.json").write_text(json.dumps(evalset))
    return d


def base_cfg(d):
    return {"seed": 0, "max_length": 64, "eval_batch_size": 2, "max_grad_norm": 1.0, "epochs": 1,
            "precision": "fp32", "log_every": 1, "eval_every": 1}


def test_sft_then_dpo_then_generate_pairs_judge(workdir):
    from pa2 import train_dpo, train_sft
    from pa2.generate import generate
    from pa2.make_pairs import make_pairs

    d = workdir
    sft_cfg = {**base_cfg(d), "model_name": str(d / "tiny"), "train_file": str(d / "sft_train.jsonl"),
               "val_file": str(d / "sft_val.jsonl"), "output_dir": str(d / "sft"),
               "per_device_batch_size": 2, "gradient_accumulation_steps": 2, "learning_rate": 1e-3,
               "lr_schedule": "cosine", "warmup_ratio": 0.1, "weight_decay": 0.1}
    res = train_sft.train(sft_cfg)
    assert res["steps"] == 2
    assert (d / "sft/best/model.safetensors").exists() and (d / "sft/final/model.safetensors").exists()

    dpo_cfg = {**base_cfg(d), "sft_checkpoint": str(d / "sft/best"), "train_file": str(d / "pref_train.jsonl"),
               "val_file": str(d / "pref_val.jsonl"), "output_dir": str(d / "dpo"), "per_device_batch_size": 1,
               "gradient_accumulation_steps": 2, "beta": 0.1, "optimizer": "adamw", "learning_rate": 1e-3,
               "reference_dtype": "fp32"}
    train_dpo.train(dpo_cfg)
    evals = [json.loads(l) for l in (d / "dpo/metrics.jsonl").read_text().splitlines()]
    step0 = next(e for e in evals if e.get("event") == "eval" and e["step"] == 0)
    assert step0["val_reward_accuracy"] == 0.0 and abs(step0["val_reward_margin"]) < 1e-6
    assert abs(step0["val_loss"] - 0.693147) < 1e-4
    assert (d / "dpo/best/checkpoint_info.json").exists()

    gen_cfg = {"prompt_format": "pa2_chatml", "max_new_tokens": 8, "batch_size": 2, "do_sample": False}
    generate(str(d / "sft/best"), d / "eval.json", d / "gen/sft.json", "sft", gen_cfg)
    generate(str(d / "dpo/best"), d / "eval.json", d / "gen/dpo.json", "dpo", gen_cfg)
    pairs = make_pairs(d / "gen/dpo.json", d / "gen/sft.json", d / "pairs.jsonl")
    assert len(pairs) == 3
    for p in pairs:
        assert {p["hidden"]["model_a"], p["hidden"]["model_b"]} == {"sft", "dpo"}

    cmd = [sys.executable, str(ROOT / "scripts/judge_pairwise_kaggle.py"), "--pairs_path", str(d / "pairs.jsonl"),
           "--output_path", str(d / "judged.jsonl"), "--backend", "mock", "--batch_size", "2", "--sleep_seconds", "0"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    summary = json.loads((d / "judged_summary.json").read_text())
    assert summary["overall"]["n"] == 3 and summary["overall"]["failed"] == 0
    # resume: nothing left to judge
    out = subprocess.run(cmd + ["--reuse_judgments"], check=True, cwd=ROOT, capture_output=True, text=True).stdout
    assert "0 to go" in out
