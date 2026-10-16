"""Build blind pairwise comparison files from two generation files.

For each prompt, the two outputs are placed as Response A / Response B in a
random order (seeded). The judge only ever sees ``instruction``,
``response_a``, ``response_b``; the mapping back to model identity is kept in
``hidden`` and used only after judging.
"""

from __future__ import annotations

import argparse
import random

from pa2.data import read_json, write_jsonl


def make_pairs(candidate_file, baseline_file, output_file, seed=42, eval_kind="general"):
    candidate = {r["id"]: r for r in read_json(candidate_file)}
    baseline = {r["id"]: r for r in read_json(baseline_file)}
    if candidate.keys() != baseline.keys():
        raise ValueError("candidate and baseline files cover different prompt ids")
    rng = random.Random(seed)
    pairs = []
    for pid in baseline:  # baseline file order = eval set order
        c, b = candidate[pid], baseline[pid]
        if c["instruction"] != b["instruction"]:
            raise ValueError(f"instruction mismatch for {pid}")
        candidate_is_a = rng.random() < 0.5
        a, bb = (c, b) if candidate_is_a else (b, c)
        pairs.append(
            {
                "pair_id": pid,
                "instruction": c["instruction"],
                "category": c.get("category"),
                "eval_kind": eval_kind,
                "response_a": a["output"],
                "response_b": bb["output"],
                "hidden": {
                    "model_a": a["generator"],
                    "model_b": bb["generator"],
                    "candidate": c["generator"],
                    "baseline": b["generator"],
                },
            }
        )
    write_jsonl(output_file, pairs)
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_kind", default="general", choices=["general", "safety", "persona"])
    args = parser.parse_args()
    make_pairs(args.candidate, args.baseline, args.output, args.seed, args.eval_kind)


if __name__ == "__main__":
    main()
