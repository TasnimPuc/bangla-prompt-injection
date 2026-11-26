---
license: apache-2.0
language: [bn, en, de]
base_model: microsoft/mdeberta-v3-base
datasets: [deepset/prompt-injections]
pipeline_tag: text-classification
tags: [prompt-injection, llm-security, bangla, banglish, code-mixing]
---
# Bangla / Banglish prompt-injection detector

mDeBERTa-v3-base fine-tuned to flag prompt-injection attacks in **Bangla, Banglish (romanised Bangla) and
code-mixed** text, plus English/German. Labels: `LEGIT` (0), `INJECTION` (1).

Training data: `deepset/prompt-injections` (Apache-2.0) machine-translated to Bangla with M2M100 (MIT),
rule-based Banglish transliteration, sentence-level code-mixing, and hand-written native red-team prompts.
Results per language variant are in the GitHub repo (`results/REPORT.md`).

**Limitations:** small source dataset; machine translation and rule-based transliteration do not cover every
writing style; new attack styles will be missed. Use it as one layer of defence, not the only one.
