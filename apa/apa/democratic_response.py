"""
Democratic inference pipeline.

CLI:
    python -m apa.democratic_response --query "What is AI?"
    python -m apa.democratic_response --query "..." --methods borda_count,copeland
    python -m apa.democratic_response --query "..." --responses_file candidates.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from apa.config import (
    HF_CACHE_DIR,
    InferenceConfig,
    InferenceLLMConfig,
    MODELS_DIR,
    configure_environment,
)
from apa.levers.slate_generation import temperature_sampling
from apa.levers.voter_aggregation import AGGREGATION_METHODS
from apa.levers.voter_sampling import (
    _normalize_source_label,
    parse_jury_source_spec,
    per_group_sampling,
    random_sampling,
    stratified_sampling,
    temporal_mix_sampling,
    weighted_sampling,
)
from apa._logging import log as _log
from apa.lore_adapt import LoReScorer

SAMPLING_METHODS: dict[str, Callable[..., list[str]]] = {
    "random": random_sampling,
    "stratified": stratified_sampling,
    "weighted": weighted_sampling,
    "temporal_mix": temporal_mix_sampling,
}


# =============================================================================
# LLM loading and response generation
# =============================================================================

_MODEL = None
_TOKENIZER = None
_MODEL_NAME = None


def load_inference_llm(
    model_name: str | None = None,
    device_map: str = "auto",
    cache_dir: str | None = None,
) -> tuple[Any, Any]:
    """Load the base LLM for response generation (cached across calls)."""
    global _MODEL, _TOKENIZER, _MODEL_NAME
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_name is None:
        model_name = InferenceLLMConfig().model_name

    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL, _TOKENIZER

    configure_environment()
    if cache_dir is None:
        cache_dir = str(HF_CACHE_DIR)

    _log(f"Loading inference LLM: {model_name}")
    _TOKENIZER = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True,
    )
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    _MODEL_NAME = model_name
    return _MODEL, _TOKENIZER


def generate_responses(
    query: str,
    k: int,
    model: Any,
    tokenizer: Any,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
) -> list[str]:
    """Generate k diverse responses via temperature sampling."""
    cfg = InferenceLLMConfig()
    config = {
        "temperature": cfg.temperature if temperature is None else temperature,
        "max_new_tokens": cfg.max_new_tokens if max_new_tokens is None else max_new_tokens,
    }
    return temperature_sampling(model, tokenizer, query, k, config)


# =============================================================================
# Response file loaders
# =============================================================================

@dataclass
class QueryCase:
    """One query + its candidate responses, loaded from a response file."""
    query: str | None = None
    responses: list[str] = field(default_factory=list)
    query_id: Any = None


def _extract_response_text(item: Any) -> str:
    """Pull a response string out of a raw file item (str or dict)."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "response", "content"):
            if key in item:
                return item[key]
        raise ValueError(
            f"Response object missing 'text'/'response'/'content': {sorted(item)}"
        )
    raise ValueError(f"Unsupported response item type: {type(item)}")


def _case_from_obj(obj: dict) -> QueryCase:
    """Convert a dict (one row of a rich jsonl or a wrapping json) to a QueryCase."""
    return QueryCase(
        query=obj.get("query"),
        responses=[_extract_response_text(r) for r in obj.get("responses", [])],
        query_id=obj.get("query_id"),
    )


def load_query_cases(path: str | Path) -> list[QueryCase]:
    """
    Load one or more (query, responses) cases from a file.

    Supported formats:

    - **Rich .jsonl** (preferred, see experiments/query_responses.jsonl):
      one JSON object per line with ``query_id``, ``query``, and
      ``responses`` (a list of ``{response_id, text}`` objects or bare
      strings). One ``QueryCase`` is returned per line.

    - **Flat .jsonl**: each line is ``{"response": "..."}``, with an
      optional ``"query"`` field on the first line. All lines collapse
      into one ``QueryCase``.

    - **.json**: either a single wrapping object
      ``{"query": ..., "responses": [...]}``, a list of such objects,
      a list of strings, or a list of ``{"response": ...}`` dicts.

    - **.txt**: one response per line; blank lines ignored. Single case.

    Response items may carry their text under ``"text"``, ``"response"``,
    or ``"content"``.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        # Accept both strict JSONL (one object per line) and a concatenated
        # stream of pretty-printed JSON objects (as in experiments/query_responses.jsonl).
        text = Path(path).read_text()
        rows: list[dict] = []
        decoder = json.JSONDecoder()
        idx = 0
        n = len(text)
        while idx < n:
            while idx < n and text[idx].isspace():
                idx += 1
            if idx >= n:
                break
            obj, end = decoder.raw_decode(text, idx)
            rows.append(obj)
            idx = end
        if not rows:
            return []
        if any("responses" in r for r in rows):
            return [_case_from_obj(r) for r in rows if "responses" in r]
        # Flat format: one response per line.
        case = QueryCase()
        for row in rows:
            if case.query is None and "query" in row:
                case.query = row["query"]
            if "response" in row or "text" in row or "content" in row:
                case.responses.append(_extract_response_text(row))
        return [case]

    if suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [_case_from_obj(data)] if "responses" in data else [
                QueryCase(responses=[_extract_response_text(data)]),
            ]
        if isinstance(data, list):
            # List of wrapping objects, or list of raw items.
            if data and isinstance(data[0], dict) and "responses" in data[0]:
                return [_case_from_obj(d) for d in data]
            return [QueryCase(responses=[_extract_response_text(item) for item in data])]
        raise ValueError(f"Unsupported JSON top-level type: {type(data)}")

    if suffix == ".txt":
        with open(path) as f:
            responses = [line.strip() for line in f if line.strip()]
        return [QueryCase(responses=responses)]

    raise ValueError(f"Unsupported response file format: {suffix}")


# =============================================================================
# Jury construction
# =============================================================================

_CENTURY_RE = re.compile(r"C(\d{2,3})")


def _period_for_user(user_id: str, origin: str) -> str:
    """
    Derive a coarse 'period' tag for a voter.

    - PRISM users → 'original' (the dataset the V basis was trained on).
    - Adapted users whose IDs encode a century (e.g. 'hist_C013_01') →
      '13C', '14C', ... (strip the leading zeros).
    - Anything else → 'other'.
    """
    if origin == "prism":
        return "original"
    m = _CENTURY_RE.search(user_id)
    if m:
        return f"{int(m.group(1))}C"
    return "other"


def build_default_jury(
    K: int = 8,
    V_checkpoint: Path | str | None = None,
    prism_users: Path | str | None = None,
    adapted_users: Path | str | list[Path | str] | None = None,
) -> tuple[LoReScorer, dict[str, dict], dict[str, Any]]:
    """
    Build a default jury from checkpoints under MODELS_DIR.

    Sources loaded (all optional — silently skipped if missing):
      - V basis: V_K{K}.pt
      - PRISM users: W_seen_K{K}.pt (+ user_to_idx.json if present)
      - Adapted users: W_adapted_hist_top3_K{K}_K{K}.pt by default
        (the curated 27-user "top-3 per century" adaptation); callers
        may override with a specific path or list of paths.

    Per-voter metadata carries:
      - 'period': 'original' for PRISM, 'NC' for adapted (e.g. '13C').
      - 'ID': the user ID (traceable back to preference files).
      - 'origin': 'prism' | 'adapted' (used for stratified sampling).
      - 'checkpoint': path the W vector was loaded from.

    Returns:
        (scorer, user_metadata, sources_summary)
    """
    V_checkpoint = Path(V_checkpoint) if V_checkpoint else MODELS_DIR / f"V_K{K}.pt"
    if not V_checkpoint.exists():
        raise FileNotFoundError(
            f"V basis checkpoint not found: {V_checkpoint}. "
            f"Train LoRe first: python -m apa.train_lore_bases"
        )

    scorer = LoReScorer.from_checkpoint(V_checkpoint)
    user_metadata: dict[str, dict] = {}
    sources: list[dict] = []

    before = set(scorer.get_user_ids())
    prism_path = Path(prism_users) if prism_users else MODELS_DIR / f"W_seen_K{K}.pt"
    if prism_path.exists():
        mapping_path = prism_path.parent / "user_to_idx.json"
        n = scorer.load_prism_users(
            prism_path, user_mapping_path=mapping_path if mapping_path.exists() else None,
        )
        new = set(scorer.get_user_ids()) - before
        for uid in new:
            user_metadata[uid] = {
                "period": _period_for_user(uid, "prism"),
                "ID": uid,
                "origin": "prism",
                "checkpoint": str(prism_path),
            }
        sources.append({"origin": "prism", "path": str(prism_path), "n_loaded": n})
        before = set(scorer.get_user_ids())

    if adapted_users is None:
        default_adapted = MODELS_DIR / f"W_adapted_hist_top3_K{K}_K{K}.pt"
        adapted_paths = [default_adapted] if default_adapted.exists() else []
    elif isinstance(adapted_users, (str, Path)):
        adapted_paths = [Path(adapted_users)]
    else:
        adapted_paths = [Path(p) for p in adapted_users]

    for path in adapted_paths:
        if not path.exists():
            continue
        n = scorer.load_adapted_users(path)
        new = set(scorer.get_user_ids()) - before
        for uid in new:
            user_metadata[uid] = {
                "period": _period_for_user(uid, "adapted"),
                "ID": uid,
                "origin": "adapted",
                "checkpoint": str(path),
            }
        sources.append({"origin": "adapted", "path": str(path), "n_loaded": n})
        before = set(scorer.get_user_ids())

    summary = {"V": str(V_checkpoint), "K": K, "sources": sources}
    return scorer, user_metadata, summary


# =============================================================================
# Result type
# =============================================================================

@dataclass
class InferenceResult:
    """Fully reconstructable record of a democratic vote."""

    timestamp: str
    query: str | None
    query_id: Any
    responses_source: str  # "llm" or path to response file
    responses: list[str]
    response_embeddings_hash: str
    jury_manifest: dict[str, Any]
    sampled_user_ids: list[str]
    sampled_user_metadata: dict[str, dict]
    per_voter_scores: dict[str, list[float]]
    per_voter_rankings: dict[str, list[int]]
    average_ranks: list[float]  # mean 1-indexed rank per response (lower = preferred)
    aggregations: dict[str, dict]  # {method: {"ranking": [...], "winner_idx": int}}
    config: dict[str, Any]

    @property
    def winner_by_method(self) -> dict[str, str]:
        return {
            method: self.responses[info["winner_idx"]]
            for method, info in self.aggregations.items()
        }

    @property
    def primary_winner(self) -> tuple[str, int, str]:
        """(method_name, winner_idx, winner_response) for the first method."""
        method = next(iter(self.aggregations))
        idx = self.aggregations[method]["winner_idx"]
        return method, idx, self.responses[idx]

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Embedding helper
# =============================================================================

def _embed_responses(
    query: str | None,
    responses: list[str],
    scorer: LoReScorer,
) -> torch.Tensor:
    """Embed (query, response) pairs once for all voters."""
    if query is not None:
        texts = [f"{query}\n\n{r}" for r in responses]
    else:
        texts = list(responses)
    return scorer.embed_texts(texts)


# =============================================================================
# Orchestrator
# =============================================================================

class DemocraticInference:
    """Inference-time democratic voting pipeline."""

    def __init__(
        self,
        scorer: LoReScorer,
        user_metadata: dict[str, dict],
        jury_manifest: dict[str, Any],
        k_responses: int | None = None,
        m_voters: int | None = None,
        methods: str | Sequence[str] | None = None,
        sampling: str = "stratified",
        sampling_config: dict | None = None,
        jury_sources: list[str] | None = None,
        seed: int | None = None,
        model: Any = None,
        tokenizer: Any = None,
    ):
        cfg = InferenceConfig()
        self.scorer = scorer
        self.user_metadata = user_metadata
        self.jury_manifest = jury_manifest
        self.k_responses = cfg.k_responses if k_responses is None else k_responses
        self.m_voters = cfg.m_voters if m_voters is None else m_voters

        if methods is None:
            methods = [cfg.aggregate_strategy]
        elif isinstance(methods, str):
            methods = [methods]
        for name in methods:
            if name not in AGGREGATION_METHODS:
                raise ValueError(
                    f"Unknown aggregation method: {name}. "
                    f"Available: {list(AGGREGATION_METHODS)}"
                )
        self.methods: list[str] = list(methods)

        if sampling not in SAMPLING_METHODS:
            raise ValueError(
                f"Unknown sampling method: {sampling}. "
                f"Available: {list(SAMPLING_METHODS)}"
            )
        self.sampling = sampling
        # Stratify on 'origin' by default (prism vs adapted → half/half jury).
        self.sampling_config = (
            dict(sampling_config) if sampling_config is not None else {"stratify_by": "origin"}
        )
        # When jury_sources is set, we'll override sampling to draw from a
        # period-filtered candidate pool. Each entry is (label, count|None);
        # count=None means "include all available voters in this group", in
        # which case the legacy stratified split (m_voters // n_groups) is
        # used. When any count is explicit, per-group sampling kicks in and
        # m_voters is ignored.
        self.jury_sources: list[tuple[str, int | None]] | None = None
        if jury_sources:
            parsed: list[tuple[str, int | None]] = []
            for s in jury_sources:
                if isinstance(s, tuple):
                    label, count = s
                    parsed.append((_normalize_source_label(label), count))
                else:
                    parsed.append(parse_jury_source_spec(s))
            self.jury_sources = parsed
        self.seed = seed
        self.model = model
        self.tokenizer = tokenizer

    def _ensure_llm(self) -> None:
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = load_inference_llm()

    def __call__(
        self,
        query: str | None = None,
        responses: list[str] | None = None,
        responses_source: str = "llm",
        query_id: Any = None,
    ) -> InferenceResult:
        """
        Run one democratic vote.

        Either `query` (generates responses via the LLM) or `responses`
        (skips generation) must be provided. When both are provided, the
        supplied responses are used and `query` is included in the embedding.
        """
        if query is None and responses is None:
            raise ValueError("Must provide either query or responses")

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # 1. Get responses.
        generated_by_llm = responses is None
        if generated_by_llm:
            self._ensure_llm()
            _log(f"Generating {self.k_responses} responses...")
            responses = generate_responses(
                query, self.k_responses, self.model, self.tokenizer,
            )
        else:
            _log(f"Using {len(responses)} pre-supplied responses (source={responses_source}).")

        # 2. Embed responses once.
        _log("Embedding responses...")
        embeddings = _embed_responses(query, responses, self.scorer)
        emb_hash = hashlib.sha256(embeddings.numpy().tobytes()).hexdigest()

        # 3. Sample jury.
        all_user_ids = self.scorer.get_user_ids()
        if not all_user_ids:
            raise RuntimeError("Jury is empty — no users loaded into the scorer.")

        if self.jury_sources:
            sampled_user_ids, sampling_strategy, sample_config = per_group_sampling(
                all_user_ids,
                self.user_metadata,
                self.jury_sources,
                m_voters_fallback=self.m_voters,
            )
        else:
            m = min(self.m_voters, len(all_user_ids))
            sample_fn = SAMPLING_METHODS[self.sampling]
            sample_config = dict(self.sampling_config)
            sampling_strategy = self.sampling
            _log(f"Sampling {m} voters from {len(all_user_ids)} via '{self.sampling}'...")
            sampled_user_ids = sample_fn(all_user_ids, self.user_metadata, m, sample_config)

        # 4. Per-voter scores and rankings.
        _log("Scoring per voter...")
        per_voter_scores: dict[str, list[float]] = {}
        per_voter_rankings: dict[str, list[int]] = {}
        for uid in sampled_user_ids:
            scores = [
                self.scorer.score_embedding(uid, embeddings[i])
                for i in range(embeddings.shape[0])
            ]
            per_voter_scores[uid] = scores
            ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            per_voter_rankings[uid] = ranking

        # Average 1-indexed rank per response across sampled voters
        # (response i's rank = position_in_ranking + 1).
        n_responses = embeddings.shape[0]
        if per_voter_rankings:
            rank_sums = [0.0] * n_responses
            for ranking in per_voter_rankings.values():
                for pos, resp_idx in enumerate(ranking):
                    rank_sums[resp_idx] += pos + 1
            n_voters = len(per_voter_rankings)
            average_ranks = [s / n_voters for s in rank_sums]
        else:
            average_ranks = [float("nan")] * n_responses

        # 5. Aggregate for each method.
        _log(f"Aggregating via methods: {self.methods}")
        aggregations: dict[str, dict] = {}
        for method in self.methods:
            fn = AGGREGATION_METHODS[method]
            ranking = fn(per_voter_rankings, {})
            aggregations[method] = {
                "ranking": list(ranking),
                "winner_idx": int(ranking[0]),
            }

        sampled_meta = {
            uid: self.user_metadata.get(uid, {"source": "unknown"})
            for uid in sampled_user_ids
        }

        result = InferenceResult(
            timestamp=datetime.now().isoformat(),
            query=query,
            query_id=query_id,
            responses_source=responses_source,
            responses=list(responses),
            response_embeddings_hash=emb_hash,
            jury_manifest=self.jury_manifest,
            sampled_user_ids=list(sampled_user_ids),
            sampled_user_metadata=sampled_meta,
            per_voter_scores=per_voter_scores,
            per_voter_rankings=per_voter_rankings,
            average_ranks=average_ranks,
            aggregations=aggregations,
            config={
                "k_responses": self.k_responses,
                "m_voters": self.m_voters,
                "methods": list(self.methods),
                "sampling": sampling_strategy,
                "sampling_config": dict(sample_config),
                # Always serialised as a list of {label, count} dicts (or
                # None when no jury_sources were set). count=None means
                # "include every available voter in that group".
                "jury_sources": (
                    [
                        {"label": label, "count": count}
                        for label, count in self.jury_sources
                    ]
                    if self.jury_sources else None
                ),
                "seed": self.seed,
                "inference_llm": (
                    InferenceLLMConfig().model_name if generated_by_llm else None
                ),
            },
        )
        return result

    @classmethod
    def with_default_jury(
        cls,
        K: int = 8,
        V_checkpoint: Path | str | None = None,
        prism_users: Path | str | None = None,
        adapted_users: Path | str | list[Path | str] | None = None,
        **kwargs,
    ) -> "DemocraticInference":
        scorer, user_metadata, sources = build_default_jury(
            K=K,
            V_checkpoint=V_checkpoint,
            prism_users=prism_users,
            adapted_users=adapted_users,
        )
        return cls(
            scorer=scorer,
            user_metadata=user_metadata,
            jury_manifest=sources,
            **kwargs,
        )


# =============================================================================
# CLI
# =============================================================================

def _print_result_summary(result: InferenceResult, show_all: bool) -> None:
    print("\n" + "=" * 60)
    print("DEMOCRATIC VOTE")
    print("=" * 60)
    print(f"Timestamp: {result.timestamp}")
    if result.query_id is not None:
        print(f"Query ID: {result.query_id}")
    if result.query:
        q = result.query if len(result.query) < 120 else result.query[:117] + "..."
        print(f"Query: {q}")
    print(f"Responses source: {result.responses_source}")
    print(f"Responses: {len(result.responses)} | Voters: {len(result.sampled_user_ids)}")
    print("Average rank per response (1 = best): "
          + ", ".join(f"#{i + 1}={r:.2f}" for i, r in enumerate(result.average_ranks)))
    print()
    for method, info in result.aggregations.items():
        idx = info["winner_idx"]
        text = result.responses[idx]
        preview = text if len(text) <= 200 else text[:197] + "..."
        print(f"[{method}] winner = #{idx + 1} | ranking = {info['ranking']}")
        print(f"    {preview}")
        print()

    if show_all:
        print("\n--- All responses ---")
        for i, r in enumerate(result.responses):
            preview = r if len(r) <= 400 else r[:397] + "..."
            print(f"\n[{i + 1}] {preview}")
        print("\n--- Per-voter rankings ---")
        for uid, ranking in result.per_voter_rankings.items():
            print(f"  {uid} ({result.sampled_user_metadata[uid].get('source')}): {ranking}")


def _default_log_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = MODELS_DIR.parent / "logs"
    return logs_dir / f"democratic_vote_{ts}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run democratic inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", type=str, default=None,
                        help="Query to run inference on. Required unless --responses_file includes one.")
    parser.add_argument("--responses_file", type=str, default=None,
                        help="Candidate responses file (.jsonl/.json/.txt). Skips LLM generation.")
    parser.add_argument("--k", type=int, default=InferenceConfig().k_responses,
                        help="Number of responses to generate (ignored when --responses_file is set).")
    parser.add_argument("--m", type=int, default=InferenceConfig().m_voters,
                        help="Number of voters to sample from the jury.")
    parser.add_argument("--methods", type=str, default=InferenceConfig().aggregate_strategy,
                        help="Comma-separated aggregation methods (borda_count,plurality,copeland,instant_runoff).")
    parser.add_argument("--sampling", type=str, default="stratified",
                        choices=list(SAMPLING_METHODS),
                        help="Voter sampling strategy (default: stratified on 'origin' — half PRISM / half adapted).")
    parser.add_argument("--K", type=int, default=8, help="LoRe rank for default jury.")
    parser.add_argument("--V_checkpoint", type=str, default=None, help="Path to V basis checkpoint.")
    parser.add_argument("--prism_users", type=str, default=None, help="Path to PRISM W_seen checkpoint.")
    parser.add_argument("--adapted_users", type=str, default=None,
                        help="Path to a W_adapted_*.pt checkpoint (comma-separated for multiple).")
    parser.add_argument("--jury_sources", type=str, default=None,
                        help="Comma-separated jury source labels (e.g. 'prism,C21,C17'). "
                             "When set, the jury is drawn evenly from these groups only; "
                             "overrides --sampling. Omit to use the default half PRISM / "
                             "half adapted composition.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--log_file", type=str, default=None,
                        help=f"Path to write audit log JSON. Default: {_default_log_path().parent}/democratic_vote_<ts>.json")
    parser.add_argument("--show_all", action="store_true", help="Print every response and per-voter ranking.")
    args = parser.parse_args()

    if args.query is None and args.responses_file is None:
        print("Error: provide --query, --responses_file, or both.", file=sys.stderr)
        sys.exit(1)

    configure_environment()

    # Parse compound args.
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    adapted_users: list[Path] | None = None
    if args.adapted_users:
        adapted_users = [Path(p.strip()) for p in args.adapted_users.split(",") if p.strip()]
    jury_sources: list[str] | None = None
    if args.jury_sources:
        jury_sources = [s.strip() for s in args.jury_sources.split(",") if s.strip()]

    # Build jury.
    _log(f"Building jury (K={args.K})...")
    inference = DemocraticInference.with_default_jury(
        K=args.K,
        V_checkpoint=args.V_checkpoint,
        prism_users=args.prism_users,
        adapted_users=adapted_users,
        k_responses=args.k,
        m_voters=args.m,
        methods=methods,
        sampling=args.sampling,
        jury_sources=jury_sources,
        seed=args.seed,
    )
    n_voters = len(inference.scorer.get_user_ids())
    _log(f"Jury size: {n_voters} voters")
    for src in inference.jury_manifest.get("sources", []):
        _log(f"  - {src['origin']}: {src['n_loaded']} from {src['path']}")
    if n_voters == 0:
        print("Error: jury is empty. Train LoRe and/or run lore_adapt first.", file=sys.stderr)
        sys.exit(1)

    # Build the list of query cases to vote on.
    if args.responses_file:
        cases = load_query_cases(args.responses_file)
        responses_source = str(Path(args.responses_file).resolve())
        _log(f"Loaded {len(cases)} query case(s) from {args.responses_file}")
        if args.query:
            # CLI --query overrides the per-case query (applies to all cases).
            for c in cases:
                c.query = args.query
    else:
        cases = [QueryCase(query=args.query)]
        responses_source = "llm"

    if not cases:
        print("Error: no query cases to vote on.", file=sys.stderr)
        sys.exit(1)

    # Run one vote per case.
    results: list[InferenceResult] = []
    for i, case in enumerate(cases, 1):
        if len(cases) > 1:
            _log(f"--- Case {i}/{len(cases)} (query_id={case.query_id}) ---")
        result = inference(
            query=case.query,
            responses=case.responses or None,
            responses_source=responses_source,
            query_id=case.query_id,
        )
        results.append(result)
        _print_result_summary(result, show_all=args.show_all)

    # Write audit log (always a JSON list, even for a single case, for consistency).
    log_path = Path(args.log_file) if args.log_file else _default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    _log(f"Audit log ({len(results)} case(s)) written to {log_path}")


if __name__ == "__main__":
    main()
