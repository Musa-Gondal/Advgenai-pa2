# PA2: SFT + DPO on Qwen2.5-0.5B

Supervised fine-tuning and from-scratch DPO (HH-RLHF and UltraFeedback) on
`Qwen/Qwen2.5-0.5B`, plus TRL `DPOTrainer` + LoRA persona tuning on
`SmolLM2-360M-Instruct`. Training and generation run on **Modal** (H100 for
training, L40S for generation). Pairwise judging runs on **Kaggle**, because that is where the free
`kaggle_benchmarks` judge models are.

## Layout

| Path | What |
|---|---|
| `submission_adapter.py` | Grading interface: thin wrappers around `pa2/` |
| `pa2/batching.py` | SFT/DPO batch building (EOS before truncation, prompt/pad labels = -100) |
| `pa2/losses.py` | Response-only log-probs, SFT loss (token mean), DPO loss |
| `pa2/formatting.py` | The single prompt format (ChatML, no system message) |
| `pa2/data.py` | Fixed subsets: UltraChat SFT, HH-RLHF, UltraFeedback, AlpacaEval-300 (seed 42) |
| `pa2/train_sft.py`, `pa2/train_dpo.py` | Hand-written training loops (no HF Trainer / TRL) |
| `pa2/train_lora_dpo.py` | Task 5: TRL `DPOTrainer` + PEFT LoRA |
| `pa2/generate.py`, `pa2/make_pairs.py` | Generations and blind A/B pairwise files |
| `scripts/judge_pairwise_kaggle.py` | Resumable batched judge (Kaggle proxy / Gemini API / mock) |
| `scripts/plot_metrics.py` | Training curves and judged-outcome chart |
| `scripts/summarize_judgments.py` | Evaluation summary (`outputs/eval/full/SUMMARY.md`) |
| `configs/*.yaml` | Every hyperparameter used |
| `modal_app.py` | Modal functions for every stage |
| `notebooks/kaggle_judge.ipynb` | Kaggle judging notebook |
| `eval_sets/` | Fixed evaluation prompts (HH eval set, AlpacaEval-300 seed 42, persona style set) |

## Tests (CPU, offline)

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q tests/
```

`tests/test_public_adapter_smoke.py` is the staff smoke suite.
`tests/test_own_edge_cases.py` covers truncation boundaries, tokenizers without
EOS or pad, mixed lengths, row order, comparison against a hand-computed
reference, and optimisation steps. `tests/test_pipeline_cpu.py` runs SFT → DPO →
generation → pairs → mock judge end to end on a tiny random Qwen2 model.

## Running on Modal

```bash
modal run modal_app.py::env_check          # GPU and package versions (Task 1)
modal run modal_app.py::prepare_data       # data subsets to the volume; writes eval_sets/alpaca_eval_300_seed42.json
modal run modal_app.py::smoke              # tiny run of every stage
modal run --detach modal_app.py::pipeline  # full run: SFT -> {DPO-HH, DPO-UF, LoRA} -> generations -> pairs
modal volume get pa2-outputs runs outputs/ # fetch metrics (weights are gitignored)
modal volume get pa2-outputs generations outputs/
modal volume get pa2-outputs eval outputs/
python scripts/plot_metrics.py
```

Each stage can also be run on its own, for example `modal run modal_app.py::dpo --track uf`
or `modal run modal_app.py::generate --job sft_alpaca --limit 5`.

### GPUs, runtime, cost

Training (SFT, both DPO runs, LoRA-DPO) runs on an **H100**, and generation on an **L40S**.
The DPO-HH, DPO-UF and LoRA runs, and the generation jobs, each run in parallel containers.
Approximate Modal list prices are H100 ~$3.95/h and L40S ~$1.95/h. The full pipeline is
expected to take about 45–55 min wall-clock and cost about $6–8. Override the GPUs with
`PA2_TRAIN_GPU` / `PA2_GEN_GPU`.

Resource-driven deviation from the guide: micro-batches are larger (SFT 16×2 instead of
2×16, DPO 8×4 instead of 1×32). The effective batch size stays at 32, and every other
hyperparameter is unchanged. Training logs `tokens_per_s`, `s_per_step` and `eta_min`.

## Judging on Kaggle

Upload `outputs/eval/full/*_pairs.jsonl` as a Kaggle dataset, open
`notebooks/kaggle_judge.ipynb` in a Kaggle benchmark notebook, and run it. Then
copy the `*_judged.jsonl`, `*_failures.jsonl` and `*_summary.json` files back
into `outputs/eval/full/`.

The submitted judgments were produced through the same Kaggle model proxy from
outside a notebook: `kaggle benchmarks auth -y --env-file kb.env` writes the proxy
credentials (keep that file out of git), then

```bash
set -a; . kb.env; set +a
export LLM_DEFAULT=google/gemini-3.6-flash LLMS_AVAILABLE=google/gemini-3.6-flash
python scripts/judge_pairwise_kaggle.py --backend kaggle --judge_model google/gemini-3.6-flash \
  --pairs_path outputs/eval/full/dpo_hh_vs_sft_pairs.jsonl \
  --output_path outputs/eval/full/dpo_hh_vs_sft_judged.jsonl \
  --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments
```

Persona pairs can also be judged for style only or usefulness/safety only with
`--eval_kind persona_style` / `--eval_kind persona_quality`. `python scripts/summarize_judgments.py`
writes the evaluation summary `outputs/eval/full/SUMMARY.md` (win/tie/loss, Wilson CIs, sign tests,
and run-to-run agreement with the earlier judging run in `outputs/eval/judge_rerun_sleep2/`).

The LoRA-DPO retry with a higher learning rate (`configs/lora_persona_tuned.yaml`) runs with
`modal run modal_app.py::lora_retry` (train, generate, rebuild pairs).

## Report

`docs/report/report.pdf` (6 pages), built from `docs/report/report.html` with headless
Chromium: `chrome --headless --no-pdf-header-footer --print-to-pdf=report.pdf report.html`.
A LaTeX copy with the same content is in `docs/report/latex/` (`report.tex`, `figures/`,
compiled `report.pdf`; build with `pdflatex report.tex` twice, or upload the folder to Overleaf).
Figures come from `python scripts/plot_metrics.py`.

---

## Original starter README

This branch is the student starter package for the SFT + DPO alignment assignment. It is intentionally minimal: it provides the tested environment metadata, assignment PDF, fixed evaluation/data files, and no completed training implementation.

## What Is Included

```text
requirements.txt
pyproject.toml
submission_adapter.py
tests/test_public_adapter_smoke.py
tests/toy_lm.py
docs/assignment_manual.pdf
data/persona_lora/ramsay_style_preferences.json
data/persona_lora/README.md
eval_sets/hh_rlhf_eval_300_seed42.json
```

## Recommended Kaggle Setup

Use a Kaggle notebook with the `GPU T4 x2` accelerator. After cloning this repository/branch, run:

```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest
python -m pip install -q -e . --no-deps
```

This keeps Kaggle's preinstalled CUDA-compatible PyTorch intact, avoids known stale `torchao` / PEFT conflicts, and makes the local repository importable without replacing the GPU stack.

## What You Need To Build

You will add your own training and evaluation code on top of this starter. At minimum, your project should implement:

- SFT training on 30,000 examples.
- From-scratch DPO on HH-RLHF-style preference data.
- From-scratch DPO on UltraFeedback-style preference data.
- TRL `DPOTrainer` + LoRA training on the provided persona preference data.
- Generation and DPO-vs-SFT evaluation outputs.
- A short final report with plots and analysis.

Read `docs/assignment_manual.pdf` for the full requirements and suggested workflow.

## Submission Adapter And Public Tests

Keep the six function signatures in `submission_adapter.py` unchanged. Implement
each function as a small wrapper around your own SFT/DPO code. The grader uses
this stable interface so your internal file and class organization can remain
your choice.

Run the public tests from the repository root:

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

The tests use a tiny local tokenizer and model, so they do not download a model,
load the assignment datasets, or require a GPU. The starter adapter raises
`NotImplementedError` until you connect it to your implementation. Passing the
public tests checks the interface and a few basic invariants; the private grader
also checks numerical correctness and harder edge cases described in the manual.

For `compute_sft_loss`, the assignment accepts either the mean of per-example
summed response negative log probabilities or the mean over all unmasked response
tokens. Use one reduction consistently and document it in the report.

## Fixed Files

- `eval_sets/hh_rlhf_eval_300_seed42.json` is the fixed synthetic helpful/harmless evaluation set.
- For AlpacaEval-style evaluation, use 300 prompts with seed `42` so comparisons are consistent.
- `data/persona_lora/ramsay_style_preferences.json` is the small preference dataset for the LoRA-DPO persona task.

## Saving Work On Kaggle

Kaggle sessions are temporary. Write important outputs under `/kaggle/working`, save notebook versions, and archive checkpoints/generations as zip files or Kaggle output datasets before ending a session.
