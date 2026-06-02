# LLM Evaluation for Intelligent Document Workflows

> **NLP for Business and Finance** — University Project, May 2026  
> Giacomo Lugana, Roberto Stoian

---

## Overview

This project benchmarks open-source **vision-language models (VLMs)** on two core invoice-processing tasks, motivated by the need for a fully local Swiss document platform (in partnership with Sacco AI Consulting) where no data leaves the execution environment.

| Task | Dataset | Metric |
|---|---|---|
| Document classification | RVL-CDIP (16 classes, 400k images) | Accuracy, Macro F1, Invoice F1 |
| Key-field extraction | SROIE (973 receipts) | Exact match, F1 |

We evaluate **zero-shot and few-shot** prompting across six models, and **fine-tune** DeiT-small and PaliGemma2-3B (LoRA) on the classification task.

---

## Key Results

- 🏆 **Fine-tuned PaliGemma2-3B** reaches **90.0% accuracy** and **0.882 invoice F1**, approaching specialist DiT-large at a fraction of the parameters
- 🔍 **Qwen2-VL-7B** achieves the strongest extraction performance among off-the-shelf models
- ⚠️ Few-shot prompting consistently **hurts classification** but **benefits large-model extraction** — optimal strategy is task- and model-dependent

---

## Repository Structure

```
├── 01_explore/          # Dataset exploration (RVL-CDIP visual properties)
├── zero_cla/            # Zero-shot document classification
├── few_cla/             # Few-shot document classification
├── extraction/          # Zero-shot & few-shot key-field extraction (SROIE)
├── fine-tuning/         # DeiT-small and PaliGemma2-3B + LoRA fine-tuning
└── LLM_Evaluation_for_Intelligent_Document_Workflows.pdf   # Full report
```

---

## Models

| Model | Parameters | Tasks |
|---|---|---|
| Qwen2-VL-7B | 7B | Classification, Extraction |
| PaliGemma2-3B | 3B | Classification, Extraction |
| PaliGemma2-3B + LoRA | 3B | Classification (fine-tuned) |
| SmolVLM-500M | 500M | Classification, Extraction |
| Qwen2.5-VL-3B | 3B | Extraction |
| DeiT-small | 22M | Classification (fine-tuned) |

---

## Setup

Python 3.10+

```bash
pip install torch transformers peft timm Pillow datasets evaluate
```

> Fine-tuning experiments were run on GPU. For inference-only notebooks, CPU is sufficient for small models.

---

## Report

The full write-up is available in [`LLM_Evaluation_for_Intelligent_Document_Workflows.pdf`](LLM_Evaluation_for_Intelligent_Document_Workflows.pdf).
