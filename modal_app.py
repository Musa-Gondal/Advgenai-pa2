"""Modal app: data prep, training (H100), and generation (L40S).

Outputs (data subsets, checkpoints, metrics, generations, pairwise files) live
on the ``pa2-outputs`` Volume under /outputs. Judging runs separately on Kaggle
(scripts/judge_pairwise_kaggle.py) because the free judge model is only
reachable through kaggle_benchmarks there.

Typical usage (from the repo root):
  modal run modal_app.py::env_check
  modal run modal_app.py::prepare_data            # also writes eval_sets/alpaca_eval_300_seed42.json locally
  modal run modal_app.py::smoke                   # tiny end-to-end run of every stage
  modal run --detach modal_app.py::pipeline --skip-data   # full run: SFT -> DPO x2 + LoRA -> generations -> pairs
  modal volume get pa2-outputs runs/sft/metrics.jsonl outputs/runs/sft/
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# Training is overhead-bound for a 0.5B model; H100 + larger micro-batches keeps the
# full pipeline under ~1 h. Generation is decode-bound, so a cheaper L40S is enough.
TRAIN_GPU = os.environ.get("PA2_TRAIN_GPU", "H100")
GEN_GPU = os.environ.get("PA2_GEN_GPU", "L40S")
OUT = Path("/outputs")
ROOT = Path("/root")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.14.1",
        "transformers==4.57.6",
        "datasets==4.8.5",
        "peft==0.21.2",
        "trl==1.15.0",
        "huggingface_hub",
        "accelerate",
        "tqdm",
        "pyyaml",
        "sentencepiece",
        "safetensors",
        "matplotlib",
    )
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("pa2")
    .add_local_dir("configs", str(ROOT / "configs"))
    .add_local_dir("data", str(ROOT / "data"))
    .add_local_dir("eval_sets", str(ROOT / "eval_sets"))
)

app = modal.App("pa2-alignment", image=image)
outputs_vol = modal.Volume.from_name("pa2-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("pa2-hf-cache", create_if_missing=True)
VOLUMES = {str(OUT): outputs_vol, "/cache": hf_cache}
SECRETS = [modal.Secret.from_name("huggingface")] if os.environ.get("PA2_USE_HF_SECRET") else []

ALPACA_FILE = "alpaca_eval_300_seed42.json"
SMOKE = {"max_train_examples": 64, "max_val_examples": 16, "eval_every": 1, "log_every": 1}


def _cfg(name, **overrides):
    from pa2.utils import load_config

    return load_config(ROOT / "configs" / name, overrides)


def _commit():
    outputs_vol.commit()


@app.function(gpu=TRAIN_GPU, timeout=600)
def env_check():
    """Task 1 equivalent: GPU visibility and package versions."""
    import json
    import subprocess

    from pa2.utils import env_report

    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    report = env_report()
    print(json.dumps(report, indent=2))
    return report


@app.function(volumes=VOLUMES, secrets=SECRETS, timeout=3 * 3600, cpu=4, memory=16384)
def _prepare_data_remote():
    from transformers import AutoTokenizer

    from pa2.data import alpaca_eval_subset, prepare_all, write_json

    cfg = _cfg("data.yaml")
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    manifest = prepare_all(tokenizer, cfg["output_dir"], cfg)
    alpaca = alpaca_eval_subset(300, cfg["seed"])
    write_json(OUT / "eval_sets" / ALPACA_FILE, alpaca)
    _commit()
    hf_cache.commit()
    return manifest, alpaca


@app.local_entrypoint()
def prepare_data():
    import json

    manifest, alpaca = _prepare_data_remote.remote()
    Path("outputs/data").mkdir(parents=True, exist_ok=True)
    Path("outputs/data/manifest.json").write_text(json.dumps(manifest, indent=2))
    Path("eval_sets", ALPACA_FILE).write_text(json.dumps(alpaca, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2))


@app.function(gpu=TRAIN_GPU, volumes=VOLUMES, secrets=SECRETS, timeout=3 * 3600)
def sft(smoke: bool = False):
    from pa2.train_sft import train

    overrides = {**SMOKE, "output_dir": str(OUT / "smoke/sft")} if smoke else {}
    return train(_cfg("sft.yaml", **overrides), commit_fn=_commit)


@app.function(gpu=TRAIN_GPU, volumes=VOLUMES, secrets=SECRETS, timeout=3 * 3600)
def dpo(track: str, smoke: bool = False):
    from pa2.train_dpo import train

    assert track in ("hh", "uf"), track
    overrides = (
        {**SMOKE, "output_dir": str(OUT / f"smoke/dpo_{track}"), "sft_checkpoint": str(OUT / "smoke/sft/best")}
        if smoke
        else {}
    )
    outputs_vol.reload()
    return train(_cfg(f"dpo_{track}.yaml", **overrides), commit_fn=_commit)


@app.function(gpu=TRAIN_GPU, volumes=VOLUMES, secrets=SECRETS, timeout=2 * 3600)
def lora_dpo(smoke: bool = False, config: str = "lora_persona.yaml"):
    import os

    from pa2.train_lora_dpo import train

    os.chdir(ROOT)  # data_file is repo-relative
    overrides = {"output_dir": str(OUT / "smoke/lora_persona"), "epochs": 1} if smoke else {}
    return train(_cfg(config, **overrides), commit_fn=_commit)


# name -> (model path, adapter path, eval set, prompt format)
def _generation_jobs(prefix):
    alpaca = str(OUT / "eval_sets" / ALPACA_FILE)
    hh = str(ROOT / "eval_sets/hh_rlhf_eval_300_seed42.json")
    persona = str(ROOT / "eval_sets/persona_style_eval_40.json")
    runs = OUT / prefix
    smol = "HuggingFaceTB/SmolLM2-360M-Instruct"
    return {
        "base_alpaca": ("Qwen/Qwen2.5-0.5B", None, alpaca, "pa2_chatml", "qwen2.5-0.5b-base"),
        "sft_alpaca": (str(runs / "sft/best"), None, alpaca, "pa2_chatml", "qwen2.5-0.5b-sft"),
        "sft_hh": (str(runs / "sft/best"), None, hh, "pa2_chatml", "qwen2.5-0.5b-sft"),
        "dpo_uf_alpaca": (str(runs / "dpo_uf/best"), None, alpaca, "pa2_chatml", "qwen2.5-0.5b-dpo-uf"),
        "dpo_hh_hh": (str(runs / "dpo_hh/best"), None, hh, "pa2_chatml", "qwen2.5-0.5b-dpo-hh"),
        "smol_persona": (smol, None, persona, "chat_template", "smollm2-360m-instruct"),
        "lora_persona": (smol, str(runs / "lora_persona/adapter"), persona, "chat_template", "smollm2-360m-lora-dpo"),
        "lora_tuned_persona": (smol, str(runs / "lora_persona_tuned/adapter"), persona, "chat_template",
                               "smollm2-360m-lora-dpo-tuned"),
    }


@app.function(gpu=GEN_GPU, volumes=VOLUMES, secrets=SECRETS, timeout=2 * 3600)
def generate(job: str, limit: int = 0, smoke: bool = False):
    from pa2.generate import generate as run_generation

    outputs_vol.reload()
    prefix = "smoke" if smoke else "runs"
    model_path, adapter, eval_file, prompt_format, generator = _generation_jobs(prefix)[job]
    gen_cfg = _cfg("generation.yaml", prompt_format=prompt_format)
    tag = "sanity" if limit else "full"
    out = OUT / ("smoke" if smoke else "") / "generations" / tag / f"{job}.json"
    rows = run_generation(model_path, eval_file, out, generator, gen_cfg,
                          adapter_path=adapter, limit=limit or None, commit_fn=_commit)
    return rows[:5]


PAIRS = {
    # name: (candidate job, baseline job, eval kind)
    "dpo_uf_vs_sft": ("dpo_uf_alpaca", "sft_alpaca", "general"),
    "dpo_hh_vs_sft": ("dpo_hh_hh", "sft_hh", "safety"),
    "lora_vs_smol": ("lora_persona", "smol_persona", "persona"),
    "lora_tuned_vs_smol": ("lora_tuned_persona", "smol_persona", "persona"),
    "sft_vs_base": ("sft_alpaca", "base_alpaca", "general"),
}


@app.function(volumes=VOLUMES, timeout=600)
def make_pairs(smoke: bool = False, tag: str = "full"):
    from pa2.make_pairs import make_pairs as build

    outputs_vol.reload()
    root = OUT / ("smoke" if smoke else "")
    gen_dir = root / "generations" / tag
    for name, (cand, base, kind) in PAIRS.items():
        if (gen_dir / f"{cand}.json").exists() and (gen_dir / f"{base}.json").exists():
            build(gen_dir / f"{cand}.json", gen_dir / f"{base}.json", root / "eval" / tag / f"{name}_pairs.jsonl",
                  seed=42, eval_kind=kind)
            print(f"wrote {name}")
    _commit()


@app.function(volumes=VOLUMES, timeout=10 * 3600)
def pipeline(skip_data: bool = False, skip_sft: bool = False):
    """Full run inside Modal, so it survives the local client disconnecting (use --detach)."""
    if not skip_data:
        _prepare_data_remote.remote()
    if not skip_sft:
        sft.remote()
    calls = [dpo.spawn("hh"), dpo.spawn("uf"), lora_dpo.spawn()]
    base = generate.spawn("base_alpaca")
    smol = generate.spawn("smol_persona")
    for call in calls:
        print(call.get())
    jobs = ["sft_alpaca", "sft_hh", "dpo_uf_alpaca", "dpo_hh_hh", "lora_persona"]
    list(generate.map(jobs))
    base.get()
    smol.get()
    make_pairs.remote()


@app.function(volumes=VOLUMES, timeout=4 * 3600)
def _smoke_remote():
    sft.remote(smoke=True)
    dpo.remote("hh", smoke=True)
    lora_dpo.remote(smoke=True)
    for job in ("sft_hh", "dpo_hh_hh", "lora_persona", "smol_persona"):
        print(generate.remote(job, limit=4, smoke=True))
    make_pairs.remote(smoke=True, tag="sanity")


@app.local_entrypoint()
def smoke():
    _smoke_remote.remote()


@app.function(volumes=VOLUMES, timeout=2 * 3600)
def _lora_retry_remote():
    print(lora_dpo.remote(config="lora_persona_tuned.yaml"))
    print(generate.remote("lora_tuned_persona"))
    make_pairs.remote()


@app.local_entrypoint()
def lora_retry():
    """Performance-driven LoRA-DPO retry (configs/lora_persona_tuned.yaml): train, generate, pair."""
    _lora_retry_remote.remote()
