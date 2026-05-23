"""
End-to-end retrain that saves the user_id -> W-row mapping for follow-up
mechanistic / demographic interpretability work.

Uses the APA pipeline's own functions for everything *except* the long
embedding step. For that step we run a local checkpointed copy of her
embedding loop (see _embed_with_checkpoint below) so a Colab runtime
crash during the 2-4h Skywork forward pass does not lose hours of work.
Her apa/ source is not modified.

Also writes every print() to a persistent log file at
checkpoints_retrained/run.log, so if the run dies you can scroll back
through the full history (including any exception traceback).

To resume after a crash: just re-run this script. It will pick up from
the last .partial.pt file. Final files (train.pkl, test.pkl, V_K8.pt,
W_seen_K8.pt, *.json) are written only on successful completion.

Run:
    python run_full_pipeline.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

HERE        = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "apa"))

OUTPUT_DIR     = HERE / "checkpoints_retrained"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
PRISM_DATA_DIR = OUTPUT_DIR / "prism_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL  = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
CHECKPOINT_EVERY = 200          # save .partial.pt every N examples
LOG_PATH         = OUTPUT_DIR / "run.log"


class _Tee:
    """File-like object that writes to both stdout and a log file."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass
    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass


def _setup_logging() -> None:
    """Mirror all stdout/stderr to LOG_PATH."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_PATH, "a", buffering=1)  # line-buffered
    log_file.write(f"\n{'=' * 70}\n[{datetime.now().isoformat()}] run_full_pipeline.py starting\n{'=' * 70}\n")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)


def _embed_with_checkpoint(dataset, model, tokenizer, device: str, output_path: Path) -> None:
    """
    Local checkpoint-aware copy of apa.load_prism._generate_embeddings.

    Saves embeddings_data to {output_path}.partial.pt every CHECKPOINT_EVERY
    examples (overwriting, so disk usage is bounded). On re-run, resumes from
    the partial file. On successful completion writes the final output_path
    and deletes the partial. If the final output_path already exists, skips.
    """
    from tqdm import tqdm

    dataset_size = len(dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(".partial.pt")

    if output_path.exists():
        print(f"Final {output_path} already exists, skipping.")
        return

    if partial_path.exists():
        embeddings_data = torch.load(partial_path, weights_only=False)
        resume_from = len(embeddings_data)
        print(f"Resuming from {partial_path}: {resume_from}/{dataset_size} done")
    else:
        embeddings_data = []
        resume_from = 0

    print(f"Generating embeddings for {dataset_size} examples to {output_path}")
    print(f"Checkpoint every {CHECKPOINT_EVERY} examples -> {partial_path}")
    start_time = time.time()

    for idx, entry in enumerate(tqdm(dataset, total=dataset_size), 1):
        if idx <= resume_from:
            continue

        if not isinstance(entry, dict):
            entry = dict(entry)
        else:
            entry = dict(entry)
            if "extra_info" in entry and isinstance(entry["extra_info"], dict):
                entry["extra_info"] = dict(entry["extra_info"])

        extra_info = entry.get("extra_info", {})
        if not isinstance(extra_info, dict):
            extra_info = {}
        entry["extra_info"] = extra_info

        prompt             = entry.get("prompt", [])
        chosen_utterance   = extra_info.get("chosen_utterance", "")
        rejected_utterance = extra_info.get("rejected_utterance", "")
        chosen_conv        = prompt + [{"content": chosen_utterance,   "role": "assistant"}]
        rejected_conv      = prompt + [{"content": rejected_utterance, "role": "assistant"}]

        # Chosen embedding
        try:
            tokenized = tokenizer.apply_chat_template(
                chosen_conv, tokenize=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                output = model(tokenized)
                embedding = output.last_hidden_state[0, -1].cpu()
            entry["extra_info"]["chosen_conv_embedding"] = embedding
            del tokenized, output
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as e:
            err = str(e).lower()
            if "out of memory" in err or ("cuda" in err and "memory" in err):
                print(f"CUDA OOM at example {idx} (chosen), skipping...")
                entry["extra_info"]["chosen_conv_embedding"] = None
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                raise

        # Rejected embedding
        try:
            tokenized = tokenizer.apply_chat_template(
                rejected_conv, tokenize=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                output = model(tokenized)
                embedding = output.last_hidden_state[0, -1].cpu()
            entry["extra_info"]["rejected_conv_embedding"] = embedding
            del tokenized, output
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as e:
            err = str(e).lower()
            if "out of memory" in err or ("cuda" in err and "memory" in err):
                print(f"CUDA OOM at example {idx} (rejected), skipping...")
                entry["extra_info"]["rejected_conv_embedding"] = None
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            else:
                raise

        embeddings_data.append(entry)

        if idx % CHECKPOINT_EVERY == 0:
            torch.save(embeddings_data, partial_path)
            elapsed = time.time() - start_time
            done    = idx - resume_from
            rate    = done / elapsed if elapsed > 0 else 0
            eta     = (dataset_size - idx) / rate if rate > 0 else 0
            print(f"[ckpt] {idx}/{dataset_size} -> {partial_path} | {rate:.1f} ex/s | ETA {eta/60:.1f} min")

        if idx % 100 == 0 and device.startswith("cuda"):
            torch.cuda.empty_cache()

    torch.save(embeddings_data, output_path)
    print(f"Saved {output_path}")
    if partial_path.exists():
        partial_path.unlink()
        print(f"Removed {partial_path}")


def step_1_embed() -> None:
    """Download PRISM + generate embeddings via Skywork-Reward-Llama-8B on GPU."""
    from apa.config import configure_environment
    from apa.load_prism import prepare_prism_data
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

    print("Step 1c: generate train embeddings (checkpoint every 200)...")
    train_ds = load_dataset("parquet", data_files=str(train_path))["train"]
    _embed_with_checkpoint(train_ds, model, tokenizer, device, EMBEDDINGS_DIR / "train.pkl")

    print("Step 1d: generate test embeddings (checkpoint every 200)...")
    test_ds = load_dataset("parquet", data_files=str(test_path))["train"]
    _embed_with_checkpoint(test_ds, model, tokenizer, device, EMBEDDINGS_DIR / "test.pkl")

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
    _setup_logging()
    try:
        step_1_embed()
        step_2_save_metadata()
        step_3_train()
        print(f"All done. Outputs in {OUTPUT_DIR}")
    except Exception:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"[{datetime.now().isoformat()}] EXCEPTION — pipeline crashed")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        traceback.print_exc()
        raise
