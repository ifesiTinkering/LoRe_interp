"""
LoRe (Low-Rank Reward) model training and embedding utilities.

This module provides:
- LoReRewardModel: Holds the V basis matrix and provides scoring
- LoReTrainer: Implements alternating minimization training
- PersonalizeBatch: Few-shot learning for new users
- Embedding utilities for Skywork-Reward model
- CLI for training on PRISM dataset

CLI Usage:
    python -m apa.train_lore_bases --K_list 0,1
    python -m apa.train_lore_bases --K_list 0,1,5 --alpha 10000.0
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from apa._logging import log


# =============================================================================
# Device Utilities
# =============================================================================

def get_device() -> torch.device:
    """Get the default device (CUDA if available)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Embedding Model
# =============================================================================

# Global cache for embedding model and tokenizer
_EMBEDDING_MODEL = None
_EMBEDDING_TOKENIZER = None
_EMBEDDING_MODEL_NAME = None


def get_embedding_model(
    model_name: str = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
    device: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[Any, Any]:
    """
    Get or load the Skywork-Reward embedding model and tokenizer.

    Uses a global cache to avoid reloading the model multiple times.
    """
    global _EMBEDDING_MODEL, _EMBEDDING_TOKENIZER, _EMBEDDING_MODEL_NAME

    if _EMBEDDING_MODEL is not None and _EMBEDDING_MODEL_NAME == model_name:
        return _EMBEDDING_MODEL, _EMBEDDING_TOKENIZER

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_dir = os.environ.get("HF_HOME", "/nas/XXXX-9/XXXX-1/APA/hf_cache")

    print(f"Loading embedding model: {model_name}")
    print(f"Device: {device}, dtype: {torch_dtype}")

    _EMBEDDING_TOKENIZER = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    _EMBEDDING_MODEL = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device,
        cache_dir=cache_dir, num_labels=1,
    )
    _EMBEDDING_MODEL.eval()
    _EMBEDDING_MODEL_NAME = model_name

    return _EMBEDDING_MODEL, _EMBEDDING_TOKENIZER


def _format_for_embedding(prompt: str, response: str, tokenizer: Any) -> str:
    """Format prompt and response as a chat conversation for embedding."""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def _extract_embedding(model: Any, tokenizer: Any, text: str, device: str = "cuda") -> np.ndarray:
    """Extract embedding from the last token's hidden state (LoRe methodology)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[-1][0, -1, :]
    return embedding.float().cpu().numpy()


def embed_text(
    text: str,
    model: Any | None = None,
    tokenizer: Any | None = None,
    model_name: str = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
) -> np.ndarray:
    """Embed a single text string."""
    if model is None or tokenizer is None:
        model, tokenizer = get_embedding_model(model_name)
    device = next(model.parameters()).device
    return _extract_embedding(model, tokenizer, text, str(device))


def embed_texts(
    texts: list[str],
    model: Any | None = None,
    tokenizer: Any | None = None,
    model_name: str = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
    batch_size: int = 4,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed multiple text strings. Returns array of shape (n_texts, 4096)."""
    if model is None or tokenizer is None:
        model, tokenizer = get_embedding_model(model_name)

    device = next(model.parameters()).device
    embeddings = []

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Embedding texts")

    for i in iterator:
        batch_texts = texts[i:i + batch_size]
        for text in batch_texts:
            emb = _extract_embedding(model, tokenizer, text, str(device))
            embeddings.append(emb)

    return np.array(embeddings)


# =============================================================================
# LoRe Reward Model
# =============================================================================

class LoReRewardModel:
    """
    LoRe reward model for scoring responses based on learned preferences.

    The reward is computed as: reward(x) = x @ V @ w
    where V is the shared basis matrix and w is the user-specific weight vector.
    """

    def __init__(self, V: torch.Tensor):
        self.V = V

    @classmethod
    def load(cls, checkpoint_path: str, device: str = 'cpu') -> "LoReRewardModel":
        """Load a LoRe model from checkpoint."""
        V = torch.load(checkpoint_path, map_location=device)
        if isinstance(V, dict):
            V = V.get('V', V.get('basis_matrix', V))
        return cls(V)

    def score(self, embedding: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Score an embedding using user weights."""
        V = self.V.to(embedding.device)
        w = w.to(embedding.device)
        return embedding @ V @ w

    @property
    def rank(self) -> int:
        return self.V.shape[1]

    @property
    def embedding_dim(self) -> int:
        return self.V.shape[0]


# =============================================================================
# LoRe Trainer
# =============================================================================

class LoReTrainer(nn.Module):
    """
    LoRe trainer implementing alternating minimization.

    Key features:
    - Alternating optimization for W (user weights) and V (basis)
    - Alpha warmup from 20% to 80% of training
    - Cosine similarity regularization toward reference model V_sft
    - Dimension filtering based on softmax threshold
    """

    def __init__(
        self,
        V_sft: torch.Tensor,
        alpha: float,
        num_classes: int,
        num_features: int,
        num_basis_vectors: int,
        num_iterations: int,
        learning_rate: float,
        logits_scale: float = 100.0,
        threshold: float = 1e-2,
        logger: Any = None,
        log_interval: int = 1000,
    ):
        super().__init__()
        device = get_device()

        self.V_sft = V_sft.to(device)
        self.V_sft_norm = F.normalize(self.V_sft, dim=0)
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_basis_vectors = num_basis_vectors
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.logits_scale = logits_scale
        self.threshold = threshold
        self.logger = logger
        self.log_interval = log_interval

        self.training_history = {
            "steps": [], "nll_W": [], "nll_V": [], "reg": [],
            "alpha_curr": [], "grad_norm_W": [], "grad_norm_V": [],
        }

        self.W = nn.Parameter(torch.rand(num_classes, num_basis_vectors, device=device))
        self.V = nn.Parameter(torch.randn(num_features, num_basis_vectors, device=device))

    @staticmethod
    def _prepare_batch(X: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch from list of per-user feature tensors."""
        x_list, y_list = [], []
        for i, x in enumerate(X):
            x_list.append(x)
            y_list.append(torch.full((x.shape[0],), i, device=x.device, dtype=torch.long))
        return torch.cat(x_list, dim=0), torch.cat(y_list, dim=0)

    def _forward_from_packed(self, X_cat: torch.Tensor, y: torch.Tensor, alpha_curr: float) -> tuple[torch.Tensor, float, float]:
        W_row = F.softmax(self.W, dim=1)
        Vw = self.V @ W_row.T
        logits_all = (X_cat @ Vw) / self.logits_scale
        logits = logits_all.gather(1, y.unsqueeze(1)).squeeze(1)
        nll = -F.logsigmoid(logits).mean()

        reg = 0.0
        if alpha_curr > 0:
            V_norm = F.normalize(self.V, dim=0)
            V_sft_norm = F.normalize(self.V_sft, dim=0)
            cos_sim = (V_norm * V_sft_norm).sum(dim=0)
            reg = torch.mean(1 - cos_sim)

        return nll, reg, 0.0

    def _alpha_at_step(self, step: int) -> float:
        """Compute alpha with warmup (0 for first 20%, linear to full at 80%)."""
        warmup_start = int(0.2 * self.num_iterations)
        warmup_end = int(0.8 * self.num_iterations)
        if step < warmup_start:
            return 0.0
        if step >= warmup_end:
            return float(self.alpha)
        return float(self.alpha) * (step - warmup_start) / (warmup_end - warmup_start)

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def train_model(self, X: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Train the model using alternating minimization.

        Returns:
            W_kept: User weight matrix after filtering
            V_kept: Basis matrix after filtering
        """
        device = get_device()
        self.to(device)

        X_cat, y = self._prepare_batch(X)
        X_cat = X_cat.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer_W = optim.Adam([self.W], lr=self.learning_rate)
        optimizer_V = optim.Adam([self.V], lr=self.learning_rate)

        for key in self.training_history:
            self.training_history[key] = []

        for step in range(self.num_iterations):
            alpha_curr = self._alpha_at_step(step)

            # Update W
            optimizer_W.zero_grad()
            nll_W, _, _ = self._forward_from_packed(X_cat, y, alpha_curr=0.0)
            nll_W.backward()
            grad_norm_W = self.W.grad.norm().item() if self.W.grad is not None else 0.0
            optimizer_W.step()

            # Update V
            optimizer_V.zero_grad()
            nll_V, reg, _ = self._forward_from_packed(X_cat, y, alpha_curr=alpha_curr)
            total_loss_V = nll_V + alpha_curr * reg
            total_loss_V.backward()
            grad_norm_V = self.V.grad.norm().item() if self.V.grad is not None else 0.0
            optimizer_V.step()

            if step % self.log_interval == 0 or step == self.num_iterations - 1:
                self.training_history["steps"].append(step)
                self.training_history["nll_W"].append(nll_W.item())
                self.training_history["nll_V"].append(nll_V.item())
                self.training_history["reg"].append(float(reg))
                self.training_history["alpha_curr"].append(alpha_curr)
                self.training_history["grad_norm_W"].append(grad_norm_W)
                self.training_history["grad_norm_V"].append(grad_norm_V)

                if self.logger and step % self.log_interval == 0:
                    self._log(f"  Step {step:5d}/{self.num_iterations}: NLL={nll_V.item():.4f}, "
                             f"Reg={float(reg):.4f}, Alpha={alpha_curr:.1f}")

            if (step + 1) == self.num_iterations:
                W_sm = F.softmax(self.W, dim=1)
                print(f"W mean per dim: {W_sm.mean(dim=0).detach().cpu().numpy()}")
                print(f"W std  per dim: {W_sm.std(dim=0).detach().cpu().numpy()}")
                with torch.no_grad():
                    V_param_norms = torch.linalg.vector_norm(self.V, ord=2, dim=0)
                print(f"||V[:, i]|| (param): {V_param_norms.detach().cpu().numpy()}")
                print(f"Step {step}: NLL(W)={nll_W.item():.4f}, NLL(V)={nll_V.item():.4f}, "
                      f"Reg={float(reg):.4f}, Alpha={alpha_curr:.4f}")

        # Filter dimensions based on threshold
        W_probs = F.softmax(self.W, dim=1)
        max_per_basis = W_probs.max(dim=0).values
        print(max_per_basis)
        mask = (max_per_basis >= self.threshold)

        W_kept = W_probs[:, mask]
        V_kept = self.V[:, mask]
        num_kept = int(mask.sum().item())
        print(f"Num dimensions kept: {num_kept}/{self.num_basis_vectors} (threshold={self.threshold})")
        print(f"W mean per dim: {W_kept.mean(dim=0).detach().cpu().numpy()}")
        print(f"W std  per dim: {W_kept.std(dim=0).detach().cpu().numpy()}")

        return W_kept, V_kept


# =============================================================================
# Few-Shot Personalization
# =============================================================================

class PersonalizeBatch(nn.Module):
    """Few-shot personalization for new users with fixed V basis."""

    def __init__(
        self,
        num_classes: int,
        num_features: int,
        num_basis_vectors: int,
        num_iterations: int,
        learning_rate: float,
        logits_scale: float = 100.0,
    ):
        super().__init__()
        device = get_device()

        self.num_classes = num_classes
        self.num_features = num_features
        self.num_basis_vectors = num_basis_vectors
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.logits_scale = logits_scale

        self.w = nn.ParameterList([
            nn.Parameter(torch.randn(num_basis_vectors, device=device))
            for _ in range(num_classes)
        ])
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, X: list[torch.Tensor], V: torch.Tensor) -> torch.Tensor:
        nll = 0
        for i, x in enumerate(X):
            V_w = V @ F.softmax(self.w[i], dim=0)
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=V.device)
            elif x.device != V.device:
                x = x.to(V.device)
            logits = x @ V_w / self.logits_scale
            log_likelihood = torch.log(torch.sigmoid(logits))
            nll += ((-log_likelihood.sum()) / len(x))
        return nll

    def train_model(self, X: list[torch.Tensor], V: torch.Tensor) -> list[torch.Tensor]:
        for _ in range(self.num_iterations):
            self.optimizer.zero_grad()
            loss = self.forward(X, V)
            loss.backward()
            self.optimizer.step()
        return [F.softmax(self.w[i], dim=0).detach() for i in range(len(X))]


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_model(
    X: list[torch.Tensor] | torch.Tensor | np.ndarray,
    V: torch.Tensor,
    w: torch.Tensor,
) -> float:
    """Evaluate model accuracy on preference pairs."""
    if isinstance(X, list):
        X = torch.cat(X, dim=0)
    X = torch.tensor(X, dtype=torch.float32, device=V.device)
    result = X @ V @ w
    num_positive = (result > 0).sum().item()
    return num_positive / result.numel()


def eval_multiple(
    W_list: list[torch.Tensor],
    V_list: list[torch.Tensor],
    test_features: list[torch.Tensor],
) -> list[float]:
    """Evaluate accuracy for multiple users."""
    N = len(test_features)
    accuracies = [evaluate_model(test_features[i], V_list[i], W_list[i]) for i in range(N)]
    print(f"Average accuracy: {np.mean(accuracies):.4f}")
    print(f"Standard deviation of accuracy: {np.std(accuracies):.4f}")
    return accuracies


def learn_multiple_few_shot(
    train_features: list[torch.Tensor],
    V: torch.Tensor,
    num_iterations: int = 500,
    learning_rate: float = 0.5,
) -> list[torch.Tensor]:
    """Learn user weights for multiple users with few-shot data."""
    device = get_device()
    N = len(train_features)
    num_features = train_features[0].shape[-1] if len(train_features) > 0 else V.shape[0]
    fitw = PersonalizeBatch(N, num_features, V.shape[1], num_iterations, learning_rate).to(device)
    return fitw.train_model(train_features, V)


# =============================================================================
# Training Runner
# =============================================================================

def run_regularized(
    K_list: list[int],
    alpha_list: list[float],
    V_final: torch.Tensor,
    train_features: list[torch.Tensor],
    test_features_sparse: list[torch.Tensor],
    train_features_unseen: list[torch.Tensor],
    test_features_sparse_unseen: list[torch.Tensor],
    N: int,
    N_unseen: int,
    device: torch.device,
    checkpoint_dir: Path,
    num_iterations: int = 20000,
    learning_rate: float = 0.5,
    few_shot_iterations: int = 500,
    few_shot_lr: float = 0.5,
    log_interval: int = 2000,
):
    """Compute accuracies for joint and few-shot learning."""
    train_accuracies_joint = []
    seen_user_unseen_prompts_accuracies_joint = []
    few_shot_train_accuracies_few_shot = []
    unseen_user_unseen_prompts_accuracies_few_shot = []
    train_accuracies_joint_std = []
    seen_user_unseen_prompts_accuracies_joint_std = []
    few_shot_train_accuracies_few_shot_std = []
    unseen_user_unseen_prompts_accuracies_few_shot_std = []

    for alpha in alpha_list:
        log(f"Alpha: {alpha}")

        for K in K_list:
            log("")
            log("=" * 50)
            log(f"Training K={K}, alpha={alpha}")
            log("=" * 50)

            if K == 0:
                V_joint = V_final
                W_joint = [torch.tensor([1.0]).to(device) for _ in range(N)]
            else:
                trainer = LoReTrainer(
                    V_sft=V_final, alpha=alpha, num_classes=N, num_features=4096,
                    num_basis_vectors=K, num_iterations=num_iterations,
                    learning_rate=learning_rate, log_interval=log_interval,
                )
                W_joint, V_joint = trainer.train_model(train_features)

                if trainer.training_history:
                    log("")
                    log(f"Training Summary for K={K}:")
                    log(f"  Initial NLL: {trainer.training_history['nll_V'][0]:.4f}")
                    log(f"  Final NLL:   {trainer.training_history['nll_V'][-1]:.4f}")

                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                v_path = checkpoint_dir / f"V_K{K}.pt"
                torch.save(V_joint.detach().cpu(), v_path)
                log(f"Saved V to {v_path}")

                w_path = checkpoint_dir / f"W_seen_K{K}.pt"
                torch.save(W_joint.detach().cpu(), w_path)
                log(f"Saved W to {w_path}")

            log("Train Performance")
            accuracies_train = eval_multiple(W_joint, [V_joint.detach() for _ in range(N)], train_features)
            train_accuracies_joint.append(np.mean(accuracies_train))
            train_accuracies_joint_std.append(np.std(accuracies_train))

            log("Seen User Unseen Prompts")
            accuracies_seen_unseen = eval_multiple(W_joint, [V_joint.detach() for _ in range(N)], test_features_sparse)
            seen_user_unseen_prompts_accuracies_joint.append(np.mean(accuracies_seen_unseen))
            seen_user_unseen_prompts_accuracies_joint_std.append(np.std(accuracies_seen_unseen))

            if K <= 1:
                W_few_shot = [torch.tensor([1.0]).to(device) for _ in range(N_unseen)]
            else:
                W_few_shot = learn_multiple_few_shot(train_features_unseen, V_joint.detach(), few_shot_iterations, few_shot_lr)

            log("Few Shot Train Performance")
            accuracies_few_shot_train = eval_multiple(W_few_shot, [V_joint.detach() for _ in range(N_unseen)], train_features_unseen)
            few_shot_train_accuracies_few_shot.append(np.mean(accuracies_few_shot_train))
            few_shot_train_accuracies_few_shot_std.append(np.std(accuracies_few_shot_train))

            log("Unseen User Unseen Prompts")
            accuracies_unseen_unseen = eval_multiple(W_few_shot, [V_joint.detach() for _ in range(N_unseen)], test_features_sparse_unseen)
            unseen_user_unseen_prompts_accuracies_few_shot.append(np.mean(accuracies_unseen_unseen))
            unseen_user_unseen_prompts_accuracies_few_shot_std.append(np.std(accuracies_unseen_unseen))

    fac = 0.25
    return (
        np.array(train_accuracies_joint),
        np.array(seen_user_unseen_prompts_accuracies_joint),
        np.array(few_shot_train_accuracies_few_shot),
        np.array(unseen_user_unseen_prompts_accuracies_few_shot),
        fac * np.array(train_accuracies_joint_std),
        fac * np.array(seen_user_unseen_prompts_accuracies_joint_std),
        fac * np.array(few_shot_train_accuracies_few_shot_std),
        fac * np.array(unseen_user_unseen_prompts_accuracies_few_shot_std),
    )


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """CLI entry point for LoRe training on PRISM."""
    from apa.config import configure_environment, EMBEDDINGS_DIR, MODELS_DIR
    from apa.load_prism import group_embeddings_by_user

    parser = argparse.ArgumentParser(
        description="Train LoRe model on PRISM dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--K_list", type=str, default="0,1", help="Comma-separated list of ranks")
    parser.add_argument("--alpha", type=float, default=10000.0, help="Regularization coefficient")
    parser.add_argument("--num_iterations", type=int, default=20000, help="Training iterations")
    parser.add_argument("--learning_rate", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--few_shot_iterations", type=int, default=500, help="Few-shot iterations")
    parser.add_argument("--few_shot_lr", type=float, default=0.5, help="Few-shot learning rate")
    parser.add_argument("--log_interval", type=int, default=2000, help="Log every N iterations")
    parser.add_argument("--embeddings_dir", type=str, default=None, help="Embeddings directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_plot", action=argparse.BooleanOptionalAction, default=True,
                        help="Save accuracy plot (use --no-save_plot to disable)")
    parser.add_argument("--embedding_model", type=str, default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
    args = parser.parse_args()

    configure_environment()

    K_list = [int(k.strip()) for k in args.K_list.split(",")]
    alpha_list = [args.alpha]

    script_start = time.time()
    log("=" * 60)
    log("Starting PRISM basis training")
    log("=" * 60)
    log(f"K values: {K_list}")
    log(f"Alpha: {args.alpha}")
    log(f"Iterations: {args.num_iterations}")
    log(f"Device: {args.device}")

    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else EMBEDDINGS_DIR
    checkpoint_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR

    log(f"Embeddings dir: {embeddings_dir}")
    log(f"Checkpoint dir: {checkpoint_dir}")

    log("Loading embeddings...")
    load_start = time.time()
    train_embeddings = torch.load(embeddings_dir / "train.pkl")
    test_embeddings = torch.load(embeddings_dir / "test.pkl")
    log(f"Loaded embeddings in {time.time() - load_start:.1f}s")
    log(f"  Train embeddings: {len(train_embeddings)} examples")
    log(f"  Test embeddings: {len(test_embeddings)} examples")

    device = args.device
    train_seen, train_unseen, test_seen, test_unseen = group_embeddings_by_user(train_embeddings, test_embeddings, device)

    N = len(train_seen)
    N_unseen = len(train_unseen)
    log(f"Dataset statistics:")
    log(f"  Train seen users: {N}")
    log(f"  Train unseen users: {N_unseen}")
    log(f"  Test seen users: {len(test_seen)}")
    log(f"  Test unseen users: {len(test_unseen)}")

    log("=" * 60)
    log("Loading reward model on CPU to extract V_final...")
    from transformers import AutoModel

    rm = AutoModel.from_pretrained(
        args.embedding_model, torch_dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager", num_labels=1, low_cpu_mem_usage=True,
    )

    last_linear_layer = None
    for name, module in rm.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_linear_layer = module

    V_final = last_linear_layer.weight[:, 0].to(device).to(torch.float32).reshape(-1, 1)
    log(f"  V_final shape: {V_final.shape}")

    del rm
    gc.collect()

    log("=" * 60)
    log("Starting training...")
    log("=" * 60)
    training_start = time.time()

    results = run_regularized(
        K_list=K_list, alpha_list=alpha_list, V_final=V_final,
        train_features=train_seen, test_features_sparse=test_seen,
        train_features_unseen=train_unseen, test_features_sparse_unseen=test_unseen,
        N=N, N_unseen=N_unseen, device=torch.device(device),
        checkpoint_dir=checkpoint_dir, num_iterations=args.num_iterations,
        learning_rate=args.learning_rate, few_shot_iterations=args.few_shot_iterations,
        few_shot_lr=args.few_shot_lr, log_interval=args.log_interval,
    )

    (train_acc, seen_unseen_acc, few_shot_train_acc, unseen_unseen_acc,
     train_std, seen_unseen_std, few_shot_train_std, unseen_unseen_std) = results

    log("=" * 60)
    log(f"Training completed in {time.time() - training_start:.1f}s")
    log("=" * 60)

    log("")
    log("Final Results:")
    log("-" * 80)
    log(f"{'Rank':<6} {'Train Acc':<12} {'Seen/Unseen':<14} {'Few-Shot':<12} {'Unseen/Unseen':<14}")
    log("-" * 80)
    for i, K in enumerate(K_list):
        log(f"{K:<6} {train_acc[i]*100:>10.2f}%  {seen_unseen_acc[i]*100:>12.2f}%  "
            f"{few_shot_train_acc[i]*100:>10.2f}%  {unseen_unseen_acc[i]*100:>12.2f}%")
    log("-" * 80)

    results_path = checkpoint_dir / f"results_alpha_{args.alpha}.json"
    results_dict = {
        "K_list": K_list, "alpha": args.alpha,
        "train_accuracy": train_acc.tolist(),
        "seen_unseen_accuracy": seen_unseen_acc.tolist(),
        "few_shot_train_accuracy": few_shot_train_acc.tolist(),
        "unseen_unseen_accuracy": unseen_unseen_acc.tolist(),
    }
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    log(f"Saved results to {results_path}")

    if args.save_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            plt.plot(K_list, seen_unseen_acc, marker='o', label="Seen Users")
            plt.plot(K_list, unseen_unseen_acc, marker='o', label="Unseen Users")
            plt.plot(K_list, train_acc, marker='o', label="Train Seen Users")
            plt.plot(K_list, few_shot_train_acc, marker='o', label="Train Unseen Users Fewshot")
            plt.xlabel('Rank')
            plt.ylabel('Accuracy')
            plt.title('Generalization Accuracy vs. Rank')
            plt.xticks(K_list, labels=["ref" if k == 0 else str(k) for k in K_list])
            plt.legend()

            plot_path = checkpoint_dir / f"accuracy_vs_rank_alpha_{args.alpha}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            log(f"Plot saved to {plot_path}")
            plt.close()
        except ImportError:
            log("matplotlib not available, skipping plot")

    log("=" * 60)
    log(f"All done! Total runtime: {time.time() - script_start:.1f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
