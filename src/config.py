from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # --- data (Hugging Face, Apache-2.0) ---
    hf_dataset: str = "deepset/prompt-injections"
    redteam_path: str = "data/redteam_seed.jsonl"   # own Bangla/Banglish/code-mixed attacks + hard negatives
    native_train_frac: float = 0.5                  # half of the red-team seeds go to training, half held out
    val_frac: float = 0.12
    seed: int = 42

    # --- augmentation ---
    mt_model: str = "facebook/m2m100_1.2B"          # MIT licence, local GPU translation en/de -> bn
    mt_batch: int = 16
    mt_max_len: int = 384

    # --- classifier ---
    model_name: str = "microsoft/mdeberta-v3-base"  # multilingual DeBERTa-v3 (MIT); English-only tokenizers fail on Bangla
    baseline_hub_model: str = "deepset/deberta-v3-base-injection"   # off-the-shelf English detector (zero-shot)
    max_len: int = 256
    epochs: int = 8
    lr: float = 2e-5
    batch_size: int = 16
    threshold: float = 0.5

    # --- outputs ---
    work_dir: str = "/kaggle/working/bpi"
    hub_model_repo: str = ""       # e.g. "your-hf-user/bangla-prompt-injection-mdeberta" (needs HF_TOKEN secret)
    hub_dataset_repo: str = ""     # e.g. "your-hf-user/bangla-prompt-injections"
    hub_space_repo: str = ""       # e.g. "your-hf-user/bangla-prompt-injection-demo"
    hf_token_secret: str = "HF_TOKEN"

    # --- optional GCS backup (same bucket as the other projects) ---
    project_id: str = ""
    bucket: str = ""
    bucket_location: str = "US"
    gcs_prefix: str = "bangla-prompt-injection"
    run_name: str = "main"
    local_store: str = ""

    def to_dict(self):
        return asdict(self)
