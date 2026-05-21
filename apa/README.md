# APA: Aggregated Preference Alignment

Code for democratic preference aggregation with personalized reward
models. The pipeline:

1. learns individual user reward models on PRISM using **LoRe**
   (low-rank reward modeling),
2. simulates historical users via the **ProgressGym HistLlama**
   century-conditioned models,
3. forms a jury of those users and aggregates their per-response rankings
   into a single democratic ranking over candidate responses.

## Citation

If you use this code, please cite the corresponding paper: 

```bibtex
@misc{2026apa,
      title={Adaptive Pluralistic Alignment: A pipeline for dynamic artificial democracy}, 
      author={XXXX-1 XXXX-5},
      year={2026},
      eprint={2605.01642},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.01642}, 
}
```

This code adapts the LoRe implementation from `https://github.com/facebookresearch/LoRe`. If you use this portion of the code, please also cite the LoRe paper: 

```bibtex
@misc{bose2025lore,
      title={LoRe: Personalizing LLMs via Low-Rank Reward Modeling}, 
      author={Avinandan Bose and Zhihan Xiong and Yuejie Chi and Simon Shaolei Du and Lin Xiao and Maryam Fazel},
      year={2025},
      eprint={2504.14439},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.14439}, 
}
```

## Installation

```bash
cd /home/XXXX-1/APA

# Set up the environment
source setup_uv.sh
uv sync

# Smoke test
uv run python -m apa.config           # imports
uv run pytest -m "not slow" -q        # ~30s, all fast tests
```

## Reproducing the paper

### Exact reproduction from checkpoints

You can quickly and exactly reproduce the paper vote results by reusing the basis functions and PRISM user weights that we pre-compute by running:

```bash
bash experiments/scripts/train_user_weights_C016_C020.sh
bash experiments/scripts/reproduce_paper_votes.sh
```

That wrapper runs both votes — `run_vote_C016_C020.sh` (regular slate)
and `run_vote_C016_C020_simple.sh` (yes/no slate) — against our pre-computed checkpoints:

```bash
experiments/checkpoints/V_K8.pt                            # LoRe basis 
experiments/checkpoints/W_seen_K8.pt                       # PRISM jury voters
```

### Full pipeline

You can also reproduce the full pipeline by learning the reward bases and PRISM weights and regenerated historical user preferences yourself. This will take a few hours with 4× A100/A6000 GPUs. The results will be comparable, but not identical, to the checkpoints above and results reported in the paper.

**Prerequisites**:
- PRISM raw data (downloaded automatically by `apa.load_prism` from
  HuggingFace).
- 1× A100/A6000 (≥40 GB) for steps 1 and 3–6.
- 4× A100/A6000 (each ≥35 GB free) for step 2 (70B HistLlama via vLLM
  with `tensor_parallel_size=4`).

#### Step 1 — Train the LoRe basis on PRISM

```bash
uv run python -m apa.load_prism --split both
uv run python -m apa.train_lore_bases --K_list 0,1,8
```

Writes `V_K{0,1,8}.pt` and `W_seen_K{0,1,8}.pt` under `$NAS_BASE/models/`.
`V_K8.pt` is the basis the rest of the pipeline reuses.

#### Step 2 — Generate synthetic preferences for C016 and C020

```bash
bash experiments/scripts/generate_prefs_C016_C020.sh
```

Reads:
- `experiments/chosen_questions.jsonl` — 500 PRISM questions where moral
  consensus is expected to vary across centuries (produced by
  `scripts/select_time_varying_questions.py`).
- `experiments/profiles_C016_C020.jsonl` — the 20 paper personas (10
  from C016 + 10 from C020).

Writes `hist_prefs_C{016,020}.jsonl` (eval_prefs format) and
`hist_prefs_C{016,020}_raw.json` (full reasoning + logprobs) under
`experiments/synthetic_prefs_C016_C020/`. Each preference pair is judged
via a two-stage CoT chat-template flow (stage 1 reasoning, stage 2
guided-decode commit to `{"X","Y"}`); both orderings are averaged to
cancel the model's ordering bias.

#### Step 3 — Filter the synthetic preferences

```bash
uv run python -m experiments.filter_output filter \
    --input  experiments/synthetic_prefs_C016_C020/hist_prefs_C016_raw.json \
             experiments/synthetic_prefs_C016_C020/hist_prefs_C020_raw.json \
    --output experiments/synthetic_prefs_C016_C020/hist_prefs_all_filtered.jsonl \
    --min-records-per-user 5
```

#### Step 4 — Few-shot LoRe adaptation: fit per-user W vectors

```bash
bash experiments/scripts/train_user_weights_C016_C020.sh
```

Defaults to fitting against the pre-computed `experiments/checkpoints/V_K8.pt`. Override
with `V_CHECKPOINT=$NAS_BASE/models/V_K8.pt bash …` to use the V you
just trained in step 1.

#### Step 5 — Hold the democratic vote

```bash
bash experiments/scripts/run_vote_C016_C020.sh
bash experiments/scripts/run_vote_C016_C020_simple.sh
```

Each script defaults to the repo-tracked checkpoints; override with the
`V_CHECKPOINT` / `PRISM_USERS` / `ADAPTED` env vars to use freshly
trained ones.

## Project structure

```
APA/
├── apa/
│   ├── _logging.py                   # Shared timestamped logger
│   ├── config.py                     # Paths and dataclass configs
│   ├── load_prism.py                 # PRISM load + Skywork embedding
│   ├── train_lore_bases.py           # LoRe basis (V) training
│   ├── lore_adapt.py                 # Few-shot W adaptation + LoReScorer
│   ├── democratic_response.py        # Jury → vote orchestrator
│   ├── vote_analysis.py              # Audit-log post-processing + reports
│   ├── synthetic_prefs/
│   │   ├── historical_prefs.py       # HistLlama-driven preference generation
│   │   ├── eval_prefs.py             # LoRe suitability metrics
│   │   ├── sample_data.py            # PRISM/random baseline samplers
│   │   ├── profiles.jsonl            # Canonical 90 personas (10 / century)
│   │   └── curated_questions.txt     # Curated value-laden PRISM questions
│   └── levers/                       # Pluggable strategies (see below)
│       ├── slate_generation.py
│       ├── voter_sampling.py
│       ├── voter_aggregation.py
│       └── query_selection.py
├── scripts/                          # Generic / validation pipelines
│   ├── select_time_varying_questions.py
│   ├── compare_metrics.py
│   ├── run_hist_prefs_full.sh
│   ├── run_all_centuries_70b.sh
│   └── run_hist_adapt.sh
├── experiments/                      # Paper experiment + figures
│   ├── scripts/
│   │   ├── generate_prefs_C016_C020.sh
│   │   ├── train_user_weights_C016_C020.sh
│   │   ├── run_vote_C016_C020.sh
│   │   └── run_vote_C016_C020_simple.sh
│   ├── filter_output.py              # 3-stage preference filter
│   ├── figs.py                       # Paper figures
│   ├── utils.py                      # extract-question-ids helper
│   ├── chosen_questions.jsonl        # 500 time-varying PRISM questions (input)
│   ├── profiles_C016_C020.jsonl      # 20 paper personas (input)
│   ├── query_responses.jsonl         # Vote slate (input)
│   ├── query_responses_simple.jsonl  # Yes/no vote slate (input)
│   ├── synthetic_prefs_C016_C020/    # Step-2/3 outputs (tracked)
│   ├── vote_C016_C020/               # Step-5 outputs (tracked)
│   ├── vote_C016_C020_simple/        # Step-5 outputs (tracked)
│   └── figs/                         # Step-6 outputs (tracked)
├── tests/                            # Fast (~30s) + 2 slow (~30 min total)
├── pyproject.toml
├── setup_uv.sh
└── README.md
```

## Levers (pluggable strategies)

The four lever modules under `apa/levers/` factor out the strategy
choices made by the pipeline. Production code dispatches by name where
applicable.

| Lever | Module | Strategies | Used by |
|-------|--------|------------|---------|
| Response generation | `slate_generation.py` | `temperature_sampling` | `democratic_response` |
| Voter sampling (jury composition) | `voter_sampling.py` | `random`, `stratified`, `weighted`, `temporal_mix`, `per_group_sampling` | `democratic_response` |
| Ranking aggregation | `voter_aggregation.py` | `borda_count`, `plurality`, `copeland`, `instant_runoff` | `democratic_response` |
| Question selection | `query_selection.py` | `random_subset`, `select_by_ids` | `historical_prefs` (which questions to put to a persona) |

`per_group_sampling` is the lever that backs the `--jury_sources` flag —
e.g. `--jury_sources "C16,C20,prism:10"` means *all C016 voters + all
C020 voters + 10 randomly-sampled PRISM voters*. It also exposes
`parse_jury_source_spec` for parsing those flag tokens.

## Testing

```bash
uv run pytest -m "not slow" -q   # ~30s
uv run pytest -q                 # ~30 min (includes test_lore.py + test_suitability.py)
```

## External resources

- [APA paper](https://arxiv.org/abs/2605.01642)
- [LoRe paper](https://arxiv.org/abs/2504.14439)
- [LoRe code](https://github.com/facebookresearch/LoRe)
- [PRISM dataset](https://github.com/HannahKirk/prism-alignment)
- [ProgressGym](https://github.com/PKU-Alignment/ProgressGym)
- [HistLlama models](https://huggingface.co/collections/PKU-Alignment/progressgym-666735fcf3e4efa276226eaa)
