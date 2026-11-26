"""Publish model / dataset / Gradio demo Space to the Hugging Face Hub."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def push_all(cfg, out: Path, token: str) -> dict:
    from huggingface_hub import HfApi
    api, res = HfApi(token=token), {}
    if cfg.hub_model_repo:
        api.create_repo(cfg.hub_model_repo, exist_ok=True)
        api.upload_folder(folder_path=str(out / "model" / "hf"), repo_id=cfg.hub_model_repo,
                          ignore_patterns=["checkpoint-*", "*.pt", "optimizer*", "scheduler*", "rng_state*"])
        api.upload_file(path_or_fileobj=str(out / "model" / "bangla_injection_mdeberta.pth"),
                        path_in_repo="bangla_injection_mdeberta.pth", repo_id=cfg.hub_model_repo)
        res["model"] = f"https://huggingface.co/{cfg.hub_model_repo}"
    if cfg.hub_dataset_repo:
        api.create_repo(cfg.hub_dataset_repo, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=str(out / "data" / "dataset.parquet"), path_in_repo="data/dataset.parquet",
                        repo_id=cfg.hub_dataset_repo, repo_type="dataset")
        api.upload_file(path_or_fileobj=str(REPO / "space" / "DATASET_CARD.md"), path_in_repo="README.md",
                        repo_id=cfg.hub_dataset_repo, repo_type="dataset")
        res["dataset"] = f"https://huggingface.co/datasets/{cfg.hub_dataset_repo}"
    if cfg.hub_space_repo:
        api.create_repo(cfg.hub_space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(folder_path=str(REPO / "space"), repo_id=cfg.hub_space_repo, repo_type="space",
                          ignore_patterns=["MODEL_CARD.md", "DATASET_CARD.md"])
        if cfg.hub_model_repo:
            api.add_space_variable(cfg.hub_space_repo, "MODEL_ID", cfg.hub_model_repo)
        res["space"] = f"https://huggingface.co/spaces/{cfg.hub_space_repo}"
    return res
