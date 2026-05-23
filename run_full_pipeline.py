"""
End-to-end retrain that saves the user_id -> W-row mapping for follow-up
mechanistic / demographic interpretability work.

Calls the APA pipeline's own functions for everything that matters
(download, parse, embed, group, train). Adds ~30 lines of glue that derive
the seen_user_ids ordering from the saved embeddings pickle, joins each
user's PRISM survey demographics, and writes both as JSON next to the new
W_seen_K8.pt checkpoint.

The patched apa.load_prism._generate_embeddings now checkpoints every 200
examples, so a Colab runtime crash during the long embedding step resumes
from disk on the next launch instead of restarting from scratch.

Run:
    python run_full_pipeline.py
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch

HERE        = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "apa"))

OUTPUT_DIR     = HERE / "checkpoints_retrained"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
PRISM_DATA_DIR = OUTPUT_DIR / "prism_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"


def step_1_embed() -> None:
    """Download PRISM + generate embeddings via Skywork-Reward-Llama-8B on GPU."""
    from apa.config import configure_environment
    from apa.load_prism import prepare_prism_data, _generate_embeddings
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    configure_environment()

    print("Step 1a: download + parse PRISM, build train/test parquet files...")
    train_path, test_path = prepare_prism_data(output_dir=PRISM_DATA_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Step 1b: load embedding model {EMBEDDING_MODEL} on {device}...")
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL, torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="eager", num_labels=1, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1c: generate train embeddings (checkpointed every 200 examples)...")
    train_ds = load_dataset("parquet", data_files=str(train_path))["train"]
    _generate_embeddings(train_ds, model, tokenizer, device, EMBEDDINGS_DIR / "train.pkl")

    print("Step 1d: generate test embeddings (checkpointed every 200 examples)...")
    test_ds = load_dataset("parquet", data_files=str(test_path))["train"]
    _generate_embeddings(test_ds, model, tokenizer, device, EMBEDDINGS_DIR / "test.pkl")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Step 1 complete.")


def step_2_save_metadata() -> None:
    """
    Derive the sorted seen_user_ids list (matches the row order
    apa.load_prism.group_embeddings_by_user produces) and join each user
    with their PRISM survey demographics.
    """
    print("Step 2: derive sorted user_ids + join demographics...")
    train_embeddings = torch.load(EMBEDDINGS_DIR / "train.pkl", weights_only=False)

    seen_user_ids: set[str] = set()
    for ex in train_embeddings:
        extra = ex.get("extra_info", {})
        if (extra.get("seen") is True
            and extra.get("split") == "train"
            and extra.get("chosen_conv_embedding") is not None
            and extra.get("rejected_conv_embedding") is not None):
            uid = extra.get("user_id")
            if uid:
                seen_user_ids.add(uid)
    sorted_user_ids = sorted(seen_user_ids)
    print(f"  {len(sorted_user_ids)} seen users with valid train embeddings")

    survey_path = PRISM_DATA_DIR / "survey.jsonl"
    demographics: dict[str, dict] = {}
    with open(survey_path) as f:
        for line in f:
            d = json.loads(line)
            uid = d.get("user_id")
            if uid in seen_user_ids:
                demographics[uid] = {
                    "preference":          d.get("preference", []),
                    "age":                 d.get("age", ""),
                    "gender":              d.get("gender", ""),
                    "education":           d.get("education", ""),
                    "employment":          d.get("employment", ""),
                    "marital":             d.get("marital", ""),
                    "english_proficiency": d.get("english_proficiency", ""),
                    "self_description":    d.get("self_description", ""),
                }

    with open(OUTPUT_DIR / "seen_user_ids.json", "w") as f:
        json.dump(sorted_user_ids, f, indent=2)
    with open(OUTPUT_DIR / "user_demographics.json", "w") as f:
        json.dump(demographics, f, indent=2)
    print(f"  Wrote seen_user_ids.json ({len(sorted_user_ids)} ids) and user_demographics.json to {OUTPUT_DIR}")


def step_3_train() -> None:
    """Train LoRe at K=8, alpha=10000 (published APA settings)."""
    print("Step 3: train LoRe K=8 alpha=10000...")
    from apa.config import configure_environment
    from apa.load_prism import group_embeddings_by_user
    from apa.train_lore_bases import run_regularized
    from transformers import AutoModel

    configure_environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_embeddings = torch.load(EMBEDDINGS_DIR / "train.pkl", weights_only=False)
    test_embeddings  = torch.load(EMBEDDINGS_DIR / "test.pkl",  weights_only=False)
    train_seen, train_unseen, test_seen, test_unseen = group_embeddings_by_user(
        train_embeddings, test_embeddings, device)

    print(f"  train_seen={len(train_seen)}, train_unseen={len(train_unseen)}")

    print("Loading reward model to extract V_final...")
    rm = AutoModel.from_pretrained(
        EMBEDDING_MODEL, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager", num_labels=1, low_cpu_mem_usage=True,
    )
    last_linear = None
    for _, module in rm.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_linear = module
    V_final = last_linear.weight[:, 0].to(device).to(torch.float32).reshape(-1, 1)
    del rm
    gc.collect()

    run_regularized(
        K_list=[8], alpha_list=[10000.0], V_final=V_final,
        train_features=train_seen, test_features_sparse=test_seen,
        train_features_unseen=train_unseen, test_features_sparse_unseen=test_unseen,
        N=len(train_seen), N_unseen=len(train_unseen), device=device,
        checkpoint_dir=OUTPUT_DIR, num_iterations=20000, learning_rate=0.5,
        few_shot_iterations=500, few_shot_lr=0.5, log_interval=2000,
    )
    print("Step 3 complete.")


if __name__ == "__main__":
    step_1_embed()
    step_2_save_metadata()
    step_3_train()
    print(f"All done. Outputs in {OUTPUT_DIR}")
