---
license: apache-2.0
language: [bn, en, de]
task_categories: [text-classification]
tags: [prompt-injection, bangla, banglish, code-mixing]
---
# Bangla / Banglish prompt injections

Augmented version of `deepset/prompt-injections` (Apache-2.0): each example in its original language,
machine-translated to Bangla (M2M100), transliterated to Banglish, and code-mixed. It also includes
hand-written native red-team prompts. Columns: `text`, `label` (1 = injection), `variant`, `src_id`
(all variants of one source example share it), `split`.
