"""Write a Markdown evaluation summary from the judged pairwise files (Task 7 / submission checklist).

Usage: python scripts/summarize_judgments.py --eval_dir outputs/eval/full \
           --rerun_dir outputs/eval/judge_rerun_sleep2 --out outputs/eval/full/SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
from math import comb, sqrt
from pathlib import Path

COMPARISONS = [  # (file stem, label, eval set)
    ("dpo_hh_vs_sft", "DPO-HH vs SFT", "HH eval-300"),
    ("dpo_uf_vs_sft", "DPO-UF vs SFT", "AlpacaEval-300 (seed 42)"),
    ("sft_vs_base", "SFT vs base (context)", "AlpacaEval-300 (seed 42)"),
    ("lora_vs_smol", "LoRA-DPO vs SmolLM2 (style + usefulness)", "persona-40"),
    ("lora_vs_smol_style", "LoRA-DPO vs SmolLM2 (style only)", "persona-40"),
    ("lora_vs_smol_quality", "LoRA-DPO vs SmolLM2 (usefulness/safety only)", "persona-40"),
    ("lora_tuned_vs_smol", "LoRA-DPO tuned vs SmolLM2 (style + usefulness)", "persona-40"),
    ("lora_tuned_vs_smol_style", "LoRA-DPO tuned vs SmolLM2 (style only)", "persona-40"),
    ("lora_tuned_vs_smol_quality", "LoRA-DPO tuned vs SmolLM2 (usefulness/safety only)", "persona-40"),
]


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def sign_test(wins, n):
    """Two-sided exact binomial test of wins vs losses (ties excluded)."""
    if n == 0:
        return None
    k = min(wins, n - wins)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def wilson(wins, n, z=1.96):
    if n == 0:
        return None, None
    p = wins / n
    centre, half = p + z * z / (2 * n), z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="outputs/eval/full")
    parser.add_argument("--rerun_dir", default="outputs/eval/judge_rerun_sleep2")
    parser.add_argument("--out", default="outputs/eval/full/SUMMARY.md")
    args = parser.parse_args()
    eval_dir, rerun_dir = Path(args.eval_dir), Path(args.rerun_dir)

    lines = [
        "# Pairwise evaluation summary",
        "",
        "Judge: google/gemini-3.6-flash via the Kaggle model proxy, temperature 0, 5 comparisons per request, "
        "5 s sleep, up to 3 retries per batch. Win/loss is for the first-named model. Win rate excludes ties; "
        "CI is a 95% Wilson interval; p is a two-sided sign test on decisive pairs.",
        "",
        "| Comparison | Eval set | n | Win | Tie | Loss | Unknown | Failed (unresolved) | Failure rows logged "
        "| Win rate excl. ties [95% CI] | Ties as 1/2 | p |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for stem, label, eval_set in COMPARISONS:
        summary_path = eval_dir / f"{stem}_judged_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        o = summary["overall"]
        decisive = o["win"] + o["loss"]
        lo, hi = wilson(o["win"], decisive)
        p = sign_test(o["win"], decisive)
        logged = len(read_jsonl(eval_dir / f"{stem}_judged_failures.jsonl"))
        rate = f"{o['win_rate_excl_ties']:.3f} [{lo:.3f}, {hi:.3f}]" if decisive else "n/a"
        lines.append(
            f"| {label} | {eval_set} | {o['n']} | {o['win']} | {o['tie']} | {o['loss']} | {o['unknown']} "
            f"| {summary['unresolved_failures']} | {logged} | {rate} | {o['win_rate_ties_half']:.3f} "
            f"| {p:.3g} |" if p is not None else "| n/a |"
        )

    lines += ["", "## Per category (W-L-T)", ""]
    for stem, label, _ in COMPARISONS:
        summary_path = eval_dir / f"{stem}_judged_summary.json"
        if not summary_path.exists():
            continue
        cats = json.loads(summary_path.read_text())["per_category"]
        if len(cats) > 1:
            body = ", ".join(f"{k} {v['win']}-{v['loss']}-{v['tie']}" for k, v in cats.items())
            lines.append(f"- **{label}**: {body}")

    if rerun_dir.exists():
        lines += [
            "",
            "## Judge run-to-run agreement",
            "",
            f"Same pairs judged earlier with a 2 s sleep ({rerun_dir}); verdicts compared per pair.",
            "",
            "| Comparison | Same outcome | Win/loss flips | Earlier win rate excl. ties |",
            "|---|---|---|---|",
        ]
        for stem, label, _ in COMPARISONS:
            new = {j["pair_id"]: j["outcome"] for j in read_jsonl(eval_dir / f"{stem}_judged.jsonl")}
            old = {j["pair_id"]: j["outcome"] for j in read_jsonl(rerun_dir / f"{stem}_judged.jsonl")}
            shared = set(new) & set(old)
            if not shared:
                continue
            same = sum(new[k] == old[k] for k in shared)
            flips = sum({new[k], old[k]} == {"win", "loss"} for k in shared)
            old_rate = json.loads((rerun_dir / f"{stem}_judged_summary.json").read_text())["overall"]["win_rate_excl_ties"]
            lines.append(f"| {label} | {same}/{len(shared)} ({same / len(shared):.0%}) | {flips} | {old_rate:.3f} |")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
