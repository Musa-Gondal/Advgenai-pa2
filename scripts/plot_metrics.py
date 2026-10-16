"""Plot training curves from metrics.jsonl files (Task 8 figures).

Usage: python scripts/plot_metrics.py --runs_dir outputs/runs --out_dir outputs/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def series(rows, event, key):
    pts = [(r["step"], r[key]) for r in rows if r.get("event") == event and key in r]
    return [p[0] for p in pts], [p[1] for p in pts]


def selected(rows):
    """Step and metrics of the saved best checkpoint: the last eval flagged best."""
    best = [r for r in rows if r.get("event") == "eval" and r.get("best")]
    return best[-1] if best else None


def smooth(values, window):
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_sft(rows, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    x, y = series(rows, "train", "train_loss")
    ax.plot(x, y, alpha=0.3, color="C0", label="train (raw, per log step)")
    ax.plot(x, smooth(y, 10), color="C0", label="train (moving avg, 10)")
    x, y = series(rows, "eval", "val_loss")
    ax.plot(x, y, "o-", color="C1", label="validation")
    best = selected(rows)
    if best:
        ax.plot(best["step"], best["val_loss"], "*", color="C3", markersize=16, zorder=5,
                label=f"selected checkpoint (step {best['step']}, best val loss)")
    ax.set(xlabel="optimizer step", ylabel="response-token NLL", title="SFT loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_dpo(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, (name, rows) in enumerate(runs.items()):
        color = f"C{i}"
        x, y = series(rows, "train", "train_loss")
        axes[0].plot(x, smooth(y, 10), color=color, alpha=0.5, label=f"{name} train (avg 10)")
        x, y = series(rows, "eval", "val_loss")
        axes[0].plot(x, y, "o-", color=color, label=f"{name} val")
        x, y = series(rows, "eval", "val_reward_accuracy")
        axes[1].plot(x, y, "o-", color=color, label=name)
        x, y = series(rows, "eval", "val_reward_margin")
        axes[2].plot(x, y, "o-", color=color, label=name)
        best = selected(rows)
        if best:
            axes[1].plot(best["step"], best["val_reward_accuracy"], "*", color=color, markersize=16, zorder=5,
                         markeredgecolor="black", label=f"{name} selected (step {best['step']})")
    axes[0].set(xlabel="optimizer step", ylabel="DPO loss", title="DPO loss")
    axes[1].set(xlabel="optimizer step", ylabel="accuracy (r_w > r_l)",
                title="Validation reward accuracy (star = best-accuracy checkpoint)")
    axes[2].set(xlabel="optimizer step", ylabel="mean r_w - r_l", title="Validation reward margin")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_lora(runs, out):
    """LoRA-DPO runs side by side: DPO loss, held-out margin, held-out chosen/rejected log-probs."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, (name, rows) in enumerate(runs.items()):
        color = f"C{i}"
        train = [r for r in rows if "loss" in r and "eval_loss" not in r]
        evals = [r for r in rows if "eval_loss" in r]
        axes[0].plot([r["step"] for r in train], [r["loss"] for r in train], color=color, alpha=0.5,
                     label=f"{name} train")
        axes[0].plot([r["step"] for r in evals], [r["eval_loss"] for r in evals], "o-", color=color, ms=3,
                     label=f"{name} held-out")
        axes[1].plot([r["step"] for r in evals], [r["eval_rewards/margins"] for r in evals], "o-", color=color,
                     ms=3, label=name)
        axes[2].plot([r["step"] for r in evals], [r["eval_logps/chosen"] for r in evals], "o-", color=color,
                     ms=3, label=f"{name} chosen")
        axes[2].plot([r["step"] for r in evals], [r["eval_logps/rejected"] for r in evals], "s--", color=color,
                     ms=3, label=f"{name} rejected")
    axes[0].set(xlabel="optimizer step", ylabel="DPO loss", title="LoRA-DPO (TRL) loss; final adapters used")
    axes[1].set(xlabel="optimizer step", ylabel="mean r_w - r_l", title="Held-out reward margin (20 pairs)")
    axes[2].set(xlabel="optimizer step", ylabel="summed log-prob", title="Held-out policy log-probs")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


COMPARISONS = [  # (summary file stem, label)
    ("sft_vs_base", "SFT vs base\n(AlpacaEval-300)"),
    ("dpo_uf_vs_sft", "DPO-UF vs SFT\n(AlpacaEval-300)"),
    ("dpo_hh_vs_sft", "DPO-HH vs SFT\n(HH eval-300)"),
    ("lora_vs_smol", "LoRA-DPO vs SmolLM2\n(persona-40)"),
    ("lora_tuned_vs_smol", "LoRA-DPO tuned vs SmolLM2\n(persona-40)"),
]


def plot_judged(eval_dir, out):
    rows = [(label, json.loads((eval_dir / f"{stem}_judged_summary.json").read_text())["overall"])
            for stem, label in COMPARISONS if (eval_dir / f"{stem}_judged_summary.json").exists()]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (label, o) in enumerate(rows):
        n = o["n"]
        left = 0.0
        for key, color in (("win", "C2"), ("tie", "0.75"), ("loss", "C3")):
            frac = o[key] / n
            ax.barh(i, frac, left=left, color=color, label=key if i == 0 else None)
            if frac > 0.05:
                ax.text(left + frac / 2, i, str(o[key]), ha="center", va="center", fontsize=8)
            left += frac
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.5, color="black", lw=0.8, ls="--")
    ax.set(xlim=(0, 1), xlabel="fraction of prompts (win/loss = first-named model wins/loses)",
           title="Pairwise judgments, Gemini 3.6 Flash (counts in bars)")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="outputs/runs")
    parser.add_argument("--out_dir", default="outputs/plots")
    parser.add_argument("--eval_dir", default="outputs/eval/full")
    args = parser.parse_args()
    runs, out = Path(args.runs_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (runs / "sft/metrics.jsonl").exists():
        plot_sft(load(runs / "sft/metrics.jsonl"), out / "sft_loss.png")
    dpo = {n: load(runs / f"dpo_{n}/metrics.jsonl") for n in ("hh", "uf") if (runs / f"dpo_{n}/metrics.jsonl").exists()}
    if dpo:
        plot_dpo({f"DPO-{k.upper()}": v for k, v in dpo.items()}, out / "dpo_curves.png")
    lora = {label: load(runs / d / "metrics.jsonl") for d, label in
            (("lora_persona", "guide (5e-5, 3 ep)"), ("lora_persona_tuned", "tuned (2e-4, 5 ep)"))
            if (runs / d / "metrics.jsonl").exists()}
    if lora:
        plot_lora(lora, out / "lora_dpo_curves.png")
    plot_judged(Path(args.eval_dir), out / "judged_outcomes.png")
    print("plots written to", out)


if __name__ == "__main__":
    main()
