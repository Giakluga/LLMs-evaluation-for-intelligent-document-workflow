# LLM Evaluation for Intelligent Document Workflows

**NLP for Business and Finance — Invoices**  
Giacomo Lugana, Roberto Stoian · May 2026

## Overview

Comparative evaluation of open-source vision-language models (VLMs) for two core invoice-processing tasks:

- **Document classification** — routing incoming documents to the correct downstream process (RVL-CDIP dataset, 16 classes)
- **Key-field extraction** — populating structured databases from receipts (SROIE dataset)

We benchmark zero-shot and few-shot prompting strategies across six models, and fine-tune DeiT-small and PaliGemma2-3B (LoRA) on the classification task. All models run **fully locally** — no cloud APIs, no data leaving the execution environment.

### Key results

- Fine-tuned PaliGemma2-3B reaches **90.0% accuracy** and **0.882 invoice F1**, approaching specialist DiT-large at a fraction of the parameters
- Qwen2-VL-7B achieves the strongest extraction performance among off-the-shelf models
- Few-shot prompting consistently hurts classification but benefits large-model extraction

## Repository structure

```
NLPv2/
├── 01_explore/                   # Dataset exploration
│   └── 01_explore_dataset_v2.ipynb
├── zero_cla/                     # Zero-shot classification
│   └── zero-shot_classification.ipynb
├── few_cla/                      # Few-shot classification
│   └── 02_classification_fewshot_v2.ipynb
├── extraction/                   # Information extraction (SROIE)
│   ├── zero-shot-information-extraction-sroie.ipynb
│   ├── few-shot-information-extraction-sroie.ipynb
│   └── rescore-zero-and-few-shot.ipynb
├── fine-tuning/                  # Fine-tuning experiments
│   ├── 03_deit_finetuning_v2.ipynb       # DeiT-small fine-tuning
│   ├── 05_paligemma_finetuning.ipynb     # PaliGemma2-3B + LoRA
│   └── 06_deit_comparison.ipynb          # DeiT comparison
├── slides_images/                # Slide assets
├── report.tex                    # LaTeX report source
└── report.pdf                    # Compiled report
```

## Models evaluated

| Model | Task |
|---|---|
| Qwen2-VL-7B | Classification & Extraction |
| PaliGemma2-3B | Classification & Extraction |
| SmolVLM | Classification & Extraction |
| DeiT-small (fine-tuned) | Classification |
| PaliGemma2-3B + LoRA (fine-tuned) | Classification |

## Datasets

- **[RVL-CDIP](https://huggingface.co/datasets/aharley/rvl_cdip)** — 400k scanned document images, 16 classes
- **[SROIE](https://huggingface.co/datasets/darentang/sroie)** — 973 scanned receipts with annotated key fields

## Requirements

Notebooks were developed with Python 3.10+. Main dependencies:

```
torch
transformers
peft
timm
Pillow
datasets
evaluate
```

Install with:

```bash
pip install torch transformers peft timm Pillow datasets evaluate
```

## Report

See [`LLM_Evaluation_for_Intelligent_Document_Workflows.pdf`](LLM_Evaluation_for_Intelligent_Document_Workflows.pdf) for the full write-up.
