# Green Fairness: A Multi-Objective Trade-off Analysis of Inference-Time Strategies in Bangla Language Processing

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20(2%C3%97%20Tesla%20T4)-20BEFF)
![Task](https://img.shields.io/badge/Task-Bias%20%26%20Energy%20Evaluation-green)

Code, results, and human ratings for our paper on bias mitigation for Bangla at inference time, measured against its **energy cost**.

**Authors:** Mohua Akter, Mubashir Jawad, Syeda Sayma Sultana, Ahmed Abdal Shafi Rasel
Dept. of CSE, East West University, Dhaka, Bangladesh

**Dataset (Kaggle):** https://www.kaggle.com/datasets/mohuamridha/bias-datasetcse407

---

## Overview

Large language models can amplify stereotypes, and for low-resource languages such as Bangla, mitigation methods are rarely evaluated against their computational cost. We compare three inference-time strategies on **250 Bangla fill-in-the-blank bias prompts** from BanStereoSet, using `meta-llama/Llama-3.2-3B-Instruct` as the generator:

| Strategy | Idea |
|---|---|
| **Greedy Decoding** | Deterministic baseline. |
| **INCLINE** | Hidden-state rotation toward English on the last three decoder layers, `h + alpha * (h W)`, with `alpha` in {0, 0.3, 0.5, 0.7, 1.0}. |
| **Best-of-5 + CHOPS-style judge** | One greedy and four sampled completions (T = 0.7); a single judge call picks the least biased. |

Each output is scored on four objectives: **fairness** (Stereotype Score and an LLM-judge rating), **fluency** (BanglaBERT pseudo-perplexity and a grammar score), **cultural accuracy** (keyword-based cultural-bias score and concept preservation), and **energy** (GPU joules measured with NVML, plus latency and tokens). We analyse them with paired statistics, a 35-combination weight-sensitivity sweep, and Pareto dominance, and validate the automatic scores with **ten human raters**.

## Key Findings

| Metric (mean over 250 prompts) | Greedy | INCLINE (alpha = 0.3) | Best-of-5 |
|---|---|---|---|
| Stereotype Score (0.5 = unbiased) | 0.502 | 0.500 | 0.502 |
| LLM-judge score (1-10) | 6.704 | 6.592 | 6.600 |
| Energy per query, gross GPU joules | 106.7 | 106.3 | 546.3 |
| Latency (s) | 1.107 | 1.132 | 5.610 |

- **Best-of-5 uses about 5.1x the energy** of Greedy and INCLINE, without a detectable fairness gain (Friedman test: SS p = 0.554, judge score p = 0.191).
- **The Stereotype Score is a tie for about 99% of outputs.** A ten-rater human validation shows it misses most stereotypical fills (it detected 2 of 13 fills that raters marked as stereotypical or anti-stereotypical).
- **No strategy is Pareto-dominated.** Across 35 objective weightings, Greedy wins 25, INCLINE 8, and Best-of-5 2, with wide bootstrap intervals.
- **Human validation:** Fleiss' kappa = 0.712 between raters; automatic SS vs. human majority Cohen's kappa = 0.233; judge vs. human fairness Spearman rho = 0.287.
- Greedy is the practical energy-aware default under the metrics used here.

## Repository Structure

```
.
├── Code/
│   ├── nb1-greedy-v2.ipynb        # Greedy decoding: generation, scoring, energy
│   ├── nb2-incline-v2.ipynb       # INCLINE: FLORES-200 alignment, alpha sweep
│   ├── nb3-bestofn-v2.ipynb       # Best-of-5 + CHOPS-style judge, 3 seed sets
│   ├── nb4-analysis-v2.ipynb      # Statistics, weight sweep, Pareto, human validation, figures
│   ├── nb5-ablation-v2.ipynb      # Cross-model ablation (Qwen, Phi-3, Gemma)
│   ├── rejudge_all_outputs.py     # Re-judges all outputs with one judge prompt
│   └── Result_sheets/             # Result CSV sheets produced by the notebooks
├── Figures/                       # Figures used in the paper
├── Human_rating/                  # Human-validation rating sheets
└── README.md
```

## Data

**Prompts.** `prompts_fixed.csv` (in the Kaggle dataset above) holds 250 prompts drawn from **BanStereoSet** (Kamruzzaman et al., *Findings of ACL 2025*), a Bangla benchmark of 1,194 sentences in nine bias categories. The subset is fixed, so every strategy receives identical prompts.

| Category | Prompts | Type |
|---|---|---|
| Profession | 65 | Universal |
| Gender | 65 | Universal |
| Religion | 56 | Culturally specific |
| Region/Caste | 64 | Culturally specific |

Columns: `prompt_id`, `prompt_text` (with a `BLANK` placeholder), `bias_category`, `stereotype_target`, `anti_stereotype`, `unrelated`, `source`.

**Result CSVs.** Each strategy's file has one row per prompt with the output and all scores: `fill_text`, `output_text`, `ss_score`, `judge_score`, `perplexity`, `grammar_score`, `cultural_bias_score`, `concept_preservation_score`, `joules` (net), `joules_gross`, `latency_seconds`, `tokens_generated`, `idle_w`, plus `alpha` (INCLINE) or `seed_set` (Best-of-5). The paper reports **gross** energy.

## How to Reproduce

All experiments ran as **Kaggle notebooks on 2x Tesla T4 GPUs** (Python 3.13, PyTorch 2.11, Transformers 5.20, Accelerate 1.15, Datasets 4.8, CodeCarbon 3.3).

1. **Accounts and access.** Request access to the gated Hugging Face models you will use (`meta-llama/Llama-3.2-3B-Instruct`, and for the ablation `google/gemma-2-2b-it`). Add your token as a Kaggle secret named `HF_TOKEN1`.
2. **Attach the dataset.** Add the Kaggle dataset linked above to each notebook.
3. **Install.** The first cell of each notebook installs the libraries:
   `transformers accelerate codecarbon datasets sentencepiece scipy scikit-learn nvidia-ml-py` (`statsmodels`, `matplotlib`, `seaborn` are preinstalled on Kaggle).
4. **Run in order** (Save & Run All works):
   1. `nb1-greedy-v2.ipynb` produces the Greedy results.
   2. `nb2-incline-v2.ipynb` builds the INCLINE rotation from 997 FLORES-200 sentence pairs and runs the alpha sweep.
   3. `nb3-bestofn-v2.ipynb` runs Best-of-5 for three seed sets.
   4. `rejudge_all_outputs.py` re-scores all outputs with a single judge prompt (attach the outputs of nb1-nb3 as inputs).
   5. `nb4-analysis-v2.ipynb` reads the re-judged files and produces the tables, statistics, weight sweep, Pareto analysis, human-validation statistics, and figures.
   6. `nb5-ablation-v2.ipynb` runs the cross-model ablation on Qwen2.5-3B, Phi-3-mini, and Gemma-2-2B.

### Energy measurement protocol

- Energy is **GPU energy measured with NVML**: a background thread samples board power every 20 ms on all visible GPUs and integrates it into a joule counter. Query energy is the counter difference across the generation call (for Best-of-5, the five generations plus the selection call). Tokenization and scoring are outside the window.
- Idle draw of the two GPUs (56-63 W) is recorded; **gross** energy is the headline metric.
- Each run starts with five discarded warm-up generations. Greedy and INCLINE are deterministic, so each query is generated three times and averaged; Best-of-5 is run as three independent seed sets.
- The **serving-stack benchmark** (32 prompts, batch sizes 1/4/8, with and without KV-cache) uses CodeCarbon, so its absolute joules are not comparable to the NVML values. Best-of-5 serving energy is **projected** as five times a single call and excludes the judge call.

## Models and Resources

- Generator and judge: `meta-llama/Llama-3.2-3B-Instruct`
- Fluency scorer: `sagorsarker/bangla-bert-base` (BanglaBERT pseudo-perplexity)
- Ablation generators: `Qwen/Qwen2.5-3B-Instruct`, `microsoft/Phi-3-mini-4k-instruct`, `google/gemma-2-2b-it`
- INCLINE alignment data: FLORES-200 (`facebook/flores`), Bangla-English dev sentences

## Responsible Use

The prompts and outputs contain stereotypical statements about professions, genders, religions, regions, and castes. They are provided for research on measuring and mitigating bias and should not be used to generate or promote discriminatory content.

## Contact

Mohua Akter, East West University, Dhaka, Bangladesh: mohuaakter1022002@gmail.com
