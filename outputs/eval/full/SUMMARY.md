# Pairwise evaluation summary

Judge: google/gemini-3.6-flash via the Kaggle model proxy, temperature 0, 5 comparisons per request, 5 s sleep, up to 3 retries per batch. Win/loss is for the first-named model. Win rate excludes ties; CI is a 95% Wilson interval; p is a two-sided sign test on decisive pairs.

| Comparison | Eval set | n | Win | Tie | Loss | Unknown | Failed (unresolved) | Failure rows logged | Win rate excl. ties [95% CI] | Ties as 1/2 | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DPO-HH vs SFT | HH eval-300 | 300 | 127 | 73 | 100 | 0 | 0 | 0 | 0.559 [0.494, 0.623] | 0.545 | 0.0842 |
| DPO-UF vs SFT | AlpacaEval-300 (seed 42) | 300 | 121 | 65 | 114 | 0 | 0 | 80 | 0.515 [0.451, 0.578] | 0.512 | 0.696 |
| SFT vs base (context) | AlpacaEval-300 (seed 42) | 300 | 229 | 11 | 60 | 0 | 0 | 0 | 0.792 [0.742, 0.835] | 0.782 | 2.01e-24 |
| LoRA-DPO vs SmolLM2 (style + usefulness) | persona-40 | 40 | 19 | 4 | 17 | 0 | 0 | 0 | 0.528 [0.370, 0.680] | 0.525 | 0.868 |
| LoRA-DPO vs SmolLM2 (style only) | persona-40 | 40 | 1 | 39 | 0 | 0 | 0 | 0 | 1.000 [0.207, 1.000] | 0.512 | 1 |
| LoRA-DPO vs SmolLM2 (usefulness/safety only) | persona-40 | 40 | 15 | 5 | 20 | 0 | 0 | 0 | 0.429 [0.280, 0.591] | 0.438 | 0.5 |
| LoRA-DPO tuned vs SmolLM2 (style + usefulness) | persona-40 | 40 | 0 | 0 | 40 | 0 | 0 | 0 | 0.000 [-0.000, 0.088] | 0.000 | 1.82e-12 |
| LoRA-DPO tuned vs SmolLM2 (style only) | persona-40 | 40 | 32 | 8 | 0 | 0 | 0 | 0 | 1.000 [0.893, 1.000] | 0.900 | 4.66e-10 |
| LoRA-DPO tuned vs SmolLM2 (usefulness/safety only) | persona-40 | 40 | 2 | 0 | 38 | 0 | 0 | 0 | 0.050 [0.014, 0.165] | 0.050 | 1.49e-09 |

## Per category (W-L-T)

- **DPO-HH vs SFT**: benign_helpfulness 10-11-9, calibration_and_uncertainty 9-13-8, cyber_abuse 14-10-6, harmful_instructions 8-8-14, hate_or_harassment 11-7-12, legal_financial_safety 18-6-6, medical_safety 18-10-2, privacy_and_doxxing 7-19-4, safe_transformation 16-5-9, self_harm_support 16-11-3
- **LoRA-DPO vs SmolLM2 (style + usefulness)**: coding 4-5-1, cooking 7-2-1, general_help 3-5-2, safety_sensitive 5-5-0
- **LoRA-DPO vs SmolLM2 (style only)**: coding 0-0-10, cooking 1-0-9, general_help 0-0-10, safety_sensitive 0-0-10
- **LoRA-DPO vs SmolLM2 (usefulness/safety only)**: coding 2-5-3, cooking 6-3-1, general_help 2-8-0, safety_sensitive 5-4-1
- **LoRA-DPO tuned vs SmolLM2 (style + usefulness)**: coding 0-10-0, cooking 0-10-0, general_help 0-10-0, safety_sensitive 0-10-0
- **LoRA-DPO tuned vs SmolLM2 (style only)**: coding 10-0-0, cooking 10-0-0, general_help 6-0-4, safety_sensitive 6-0-4
- **LoRA-DPO tuned vs SmolLM2 (usefulness/safety only)**: coding 0-10-0, cooking 0-10-0, general_help 0-10-0, safety_sensitive 2-8-0

## Judge run-to-run agreement

Same pairs judged earlier with a 2 s sleep (outputs/eval/judge_rerun_sleep2); verdicts compared per pair.

| Comparison | Same outcome | Win/loss flips | Earlier win rate excl. ties |
|---|---|---|---|
| DPO-HH vs SFT | 246/300 (82%) | 18 | 0.554 |
| DPO-UF vs SFT | 260/300 (87%) | 12 | 0.543 |
| SFT vs base (context) | 283/300 (94%) | 9 | 0.810 |
| LoRA-DPO vs SmolLM2 (style + usefulness) | 33/40 (82%) | 3 | 0.474 |
