"""Builds notebooks/kaggle_run.ipynb from the repo (self-contained). Run: python tools/build_notebook.py"""
from pathlib import Path

import sys

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbsteps import cells as step_cells  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Bangla / Banglish Prompt-Injection Detector (Kaggle)

**Before Run All:** GPU **T4** (one is enough) · Internet **On**.
If you ran an older version of this notebook in this session: Run → **Restart & clear cell outputs** first.
If you ran an older version of this notebook: Run → **Restart & clear cell outputs** first.
Optional: Add-ons → Secrets → `HF_TOKEN` (a *write* token from huggingface.co/settings/tokens) to publish the
model, dataset and demo Space automatically.

Pipeline (~20–30 min): `deepset/prompt-injections` from Hugging Face → Bangla translation on the GPU (M2M100) →
Banglish + code-mixed variants + native red-team prompts → three detectors compared per language → report,
`.pth`, HF model folder."""

INSTALL = """# keep Kaggle's transformers happy: never upgrade huggingface_hub past what transformers allows
!pip install -q langid sentencepiece protobuf "huggingface_hub>=1.3,<2.0" 2>&1 | tail -1
import importlib.metadata as md
print({p: md.version(p) for p in ("transformers", "huggingface_hub", "accelerate", "datasets", "torch")})"""

CONFIG = """import sys
sys.path.insert(0, "/kaggle/working/repo")
from src.config import Config

HF_USER = "YOUR_HF_USERNAME"          # <-- for publishing (leave as is to skip publishing)
publish = HF_USER != "YOUR_HF_USERNAME"
cfg = Config(
    work_dir="/kaggle/working/bpi",
    hub_model_repo=f"{HF_USER}/bangla-prompt-injection-mdeberta" if publish else "",
    hub_dataset_repo=f"{HF_USER}/bangla-prompt-injections" if publish else "",
    hub_space_repo=f"{HF_USER}/bangla-prompt-injection-demo" if publish else "",
    project_id="mimic-cxr-training",   # optional: also back up report + .pth to your GCS bucket ("" to skip)
)"""


SHOW = """import re, shutil, glob
from IPython.display import Markdown, Image, display
rep = f"{cfg.work_dir}/report"
md = open(f"{rep}/REPORT.md", encoding="utf-8").read()
display(Markdown(re.sub(r"!\\[.*?\\]\\(figures/.*?\\)", "", md)))
for f in sorted(glob.glob(f"{rep}/figures/*.png")):
    display(Image(f))
shutil.make_archive("/kaggle/working/results_for_github", "zip", rep)
print("Results zip  : /kaggle/working/results_for_github.zip -> unzip into the repo as results/")
print("Model (.pth) :", f"{cfg.work_dir}/model/bangla_injection_mdeberta.pth")
print("HF folder    :", f"{cfg.work_dir}/model/hf")"""


def main():
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_markdown_cell("## 1 · Setup"), nbf.v4.new_code_cell(INSTALL),
             nbf.v4.new_markdown_cell("## 2 · Project code\nThese cells write the package to `/kaggle/working/repo`. "
                                      "No need to edit them; the steps further down call this code."),
             nbf.v4.new_code_cell("!mkdir -p /kaggle/working/repo/src /kaggle/working/repo/app /kaggle/working/repo/space /kaggle/working/repo/data")]
    files = sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "app").glob("*.py")) + \
        sorted((ROOT / "space").glob("*")) + [ROOT / "data" / "redteam_seed.jsonl"]
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        cells.append(nbf.v4.new_code_cell(f"%%writefile /kaggle/working/repo/{rel}\n" + f.read_text(encoding="utf-8")))
    cells += [nbf.v4.new_markdown_cell("## 3 · Config\nChange only this cell."), nbf.v4.new_code_cell(CONFIG)]
    steps, n = step_cells(ROOT / "src" / "pipeline.py", 4)
    cells += steps + [nbf.v4.new_markdown_cell(f"## {n} · Results report\nAggregate results only (safe for GitHub)."),
                      nbf.v4.new_code_cell(SHOW)]
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}
    nbf.write(nb, ROOT / "notebooks" / "kaggle_run.ipynb")
    print("wrote notebooks/kaggle_run.ipynb")


if __name__ == "__main__":
    main()
