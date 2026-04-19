# logits-SAM for DPO (ICLR 2026) - Official Implementation

Official implementation of the ICLR 2026 paper:

> **SHARPNESS-AWARE MINIMIZATION IN LOGIT SPACE EFFICIENTLY ENHANCES DIRECT PREFERENCE OPTIMIZATION**

This codebase is built on top of the Hugging Face H4 **[alignment-handbook](https://github.com/huggingface/alignment-handbook)** and follows its recipe-driven training flow.

## What This Adds

- A logits-SAM trainer for DPO: `LogitsSAMTrainer` (`src/alignment/sam_trainer.py`)
- A lightweight switch in `scripts/dpo.py` to use the SAM trainer via CLI flags

Compared with the version used at paper submission time, the current codebase includes some improvements in efficiency and readability. If you want to reproduce the paper numbers as closely as possible, we recommend using `src/alignment/old_version/sam_trainer.py`.

## Quickstart

Set up the environment first. The dependency stack for this repo is:

- `torch==2.5.1+cu121`
- `transformers==4.56.1`
- `trl==0.23.0`
- `deepspeed==0.17.5`

### Environment Setup

Create a Python virtual environment:

```bash
python3.11 -m venv handbook && source handbook/bin/activate && pip install --upgrade pip
```

Install PyTorch `v2.5.1`:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

The exact PyTorch version matters for compatibility. If your hardware requires a different build, check the [PyTorch installation page](https://pytorch.org/get-started/locally/).

Install the remaining package dependencies from the repo root:

```bash
pip install .
```

The following steps are optional:

Optional: install `flash-attn` after the main package install:

```bash
pip install "flash-attn==2.7.4.post1" --no-build-isolation
```

Optional: log into your Hugging Face account:

```bash
huggingface-cli login
```

Optional: install Git LFS if you plan to push models to the Hugging Face Hub:

```bash
sudo apt-get install git-lfs
```

Then run the original DPO recipe command and append the extra flags:

- `--use_sam_trainer`: enable `LogitsSAMTrainer`
- `--sam_rho <float>`: SAM radius (default: `0.05`)

### Example: 4-GPU / DDP Full Fine-Tuning (Recipe + Extra Flags)

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/ddp.yaml \
  --num_processes=4 \
  scripts/dpo.py \
  --config recipes/zephyr-7b-beta/dpo/config_full.yaml \
  --use_sam_trainer \
  --sam_rho 1e-3
```

### Example: Full Fine-Tuning with DeepSpeed ZeRO-3 (Recipe + Extra Flags)

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  scripts/dpo.py \
  --config recipes/zephyr-7b-beta/dpo/config_full.yaml \
  --use_sam_trainer \
  --sam_rho 1e-3
```

If you run into GPU OOM issues, you can consider enabling the chunked `delta_logits` implementation for the ZeRO-3 non-adaptive no-gather path by appending `--sam_use_chunked_delta_logits --sam_vocab_chunk_size 4096` (or another chunk size that fits your memory budget).

You can use any other alignment-handbook recipe config the same way (e.g. `recipes/smollm2/dpo/config.yaml`).

## Notes on DeepSpeed ZeRO-3

logits-SAM support for DeepSpeed ZeRO-3 is still being optimized for efficiency.

## Code Pointers

- `src/alignment/sam_trainer.py`: `LogitsSAMTrainer` implementation (logits-SAM for DPO)
- `src/alignment/old_version/sam_trainer.py`: older implementation closer to the paper submission version
- `scripts/dpo.py`: CLI flags and trainer selection (`--use_sam_trainer`, `--sam_rho`)
