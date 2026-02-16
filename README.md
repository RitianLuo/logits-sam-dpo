# Logits-Space SAM for DPO (ICLR 2026) - Official Implementation

Official implementation of the ICLR 2026 paper:

> **SHARPNESS-AWARE MINIMIZATION IN LOGIT SPACE EFFICIENTLY ENHANCES DIRECT PREFERENCE OPTIMIZATION**

This codebase is built on top of the Hugging Face H4 **alignment-handbook** and follows its recipe-driven training flow.

## What This Adds

- A logits-space SAM trainer for DPO: `LogitsSAMTrainer` (`scripts/sam_trainer.py`)
- A lightweight switch in `scripts/dpo.py` to use the SAM trainer via CLI flags

## Quickstart

Follow the upstream alignment-handbook installation instructions first (Python env, dependencies, Hugging Face auth, etc.).

Then run the original DPO recipe command and append the extra flags:

- `--use_sam_trainer`: enable `LogitsSAMTrainer`
- `--sam_rho <float>`: SAM radius (default: `0.05`)

### Example: Full Fine-Tuning with DeepSpeed ZeRO-3 (Recipe + Extra Flags)

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero3.yaml \
  scripts/dpo.py \
  --config recipes/zephyr-7b-beta/dpo/config_full.yaml \
  --use_sam_trainer \
  --sam_rho 1e-3
```

### Example: 2-GPU / DDP Full Fine-Tuning (Recipe + Extra Flags)

```bash
CUDA_VISIBLE_DEVICES=0,1 ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/ddp.yaml \
  --num_processes=2 \
  scripts/dpo.py \
  --config recipes/zephyr-7b-beta/dpo/config_full.yaml \
  --use_sam_trainer \
  --sam_rho 1e-3
```

You can use any other alignment-handbook recipe config the same way (e.g. `recipes/smollm2/dpo/config.yaml`).

## Notes on DeepSpeed ZeRO-3

DeepSpeed ZeRO-3 support for the logits-only SAM path is still being optimized. 

## Code Pointers

- `scripts/sam_trainer.py`: `LogitsSAMTrainer` implementation (logits-only SAM for DPO)
- `scripts/dpo.py`: CLI flags and trainer selection (`--use_sam_trainer`, `--sam_rho`)
