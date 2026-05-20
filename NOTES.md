# Fork notes

This is a fork of [facebookresearch/LoRe](https://github.com/facebookresearch/LoRe) used for a fellowship application exploring interpretability of personalized reward model bases on the PRISM dataset.

## Changes from upstream

- **`PRISM/prepare.py`**: removed the `IPython.core.ultratb` exception hook (lines 18–20 in upstream). It's an interactive-debugger hook that doesn't help in headless / Colab runs and emits noisy tracebacks when something goes wrong. No functional change.
- **`.gitignore`**: added. Excludes `.env`, generated data (`PRISM/data/`, `*.pkl`, `*.parquet`, `*.safetensors`), Python caches, and trained checkpoints. The repo no longer accidentally tracks the ~500 MB PRISM JSONLs or the ~16 GB embedding files when you run the pipeline.
- **`NOTES.md`**: this file.

No changes to the LoRe algorithm itself (`utils.py`), the embedding script (`PRISM/generate-prism-embeddings.py`), or the training script (`PRISM/train_basis.py`). Everything upstream still runs.

## How to use this fork

In Colab (after setting `HF_TOKEN` as a notebook secret):

```python
!git clone https://github.com/YOUR_USERNAME/LoRe.git
%cd LoRe/PRISM
!python prepare.py                       # downloads PRISM, builds train/test parquets
!python generate-prism-embeddings.py     # embeds via Skywork-Reward-Llama-3.1-8B (~45 min on A100)
```

Then call `solve_regularized_simplex` from `utils.py` directly to train a LoRe basis at your chosen K and extract the `W`, `V` matrices for analysis (the upstream `train_basis.py` only saves accuracy curves).

## License

Upstream LICENSE (CC-BY-NC 4.0) applies unchanged. See `LICENSE`.
