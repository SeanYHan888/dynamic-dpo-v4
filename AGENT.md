# AGENT.md

This file is a working guide for engineers and coding agents operating on this repository.

## Purpose

This repo is a preference-training codebase centered on:

- SimPO
- alpha-DPO
- beta-DPO
- margin-DPO

The project uses a shared text-to-token preference pipeline and custom trainers layered on top of Hugging Face `Trainer`.

## Environment

- Python: `>=3.10,<3.12`
- Package manager: `uv`
- Default install command:

```bash
uv sync
```

`wandb` is part of the standard dependency set.
`hf-transfer` is also part of the standard dependency set, and the repo enables `HF_HUB_ENABLE_HF_TRANSFER=1` by default through the shared model-loading path.

Optional extras:

- `--extra ultra`: FlashAttention on Linux only
- `--extra deepspeed`: DeepSpeed runtime
- `--extra quantization`: bitsandbytes / low-bit loading
- `--extra dev`: pytest

## Core File Map

### Entry points

- [scripts/run_simpo.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/run_simpo.py)
- [scripts/run_alpha_dpo.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/run_alpha_dpo.py)
- [scripts/run_beta_dpo.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/run_beta_dpo.py)
- [scripts/run_margin_dpo.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/run_margin_dpo.py)

### Shared runtime / preprocessing

- [utils/runtime.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/utils/runtime.py)
- [utils/checkpoint_io.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/utils/checkpoint_io.py)
- [utils/preprocessing_cache.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/utils/preprocessing_cache.py)
- [scripts/trainer_configs.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/trainer_configs.py)
- [scripts/tokenized_dpo_trainer.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/tokenized_dpo_trainer.py)

### Operator tools

- [tools/pretokenize_preferences.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/tools/pretokenize_preferences.py)
- [tools/export_fsdp_checkpoint.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/tools/export_fsdp_checkpoint.py)
- [tools/validate_and_upload.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/tools/validate_and_upload.py)

### Algorithm-specific trainers

- [scripts/simpo_trainer.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/simpo_trainer.py)
- [scripts/alpha_dpo_trainer.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/alpha_dpo_trainer.py)
- [scripts/beta_dpo_trainer.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/beta_dpo_trainer.py)
- [scripts/margin_dpo_trainer.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/margin_dpo_trainer.py)

### Configs

- [training_configs](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/training_configs)
- [accelerate_configs](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/accelerate_configs)

## Architecture

### 1. Dataset formatting

The run scripts all delegate preference dataset preparation to `utils/runtime.py`.

The high-level flow is:

1. Load raw preference data with `alignment.get_datasets(...)`
2. Keep only preference-relevant columns
3. Apply chat templating
4. Normalize rows into textual:
   - `prompt`
   - `chosen`
   - `rejected`

### 2. Tokenization and batching

The shared trainer base in `tokenized_dpo_trainer.py` owns:

- prompt/response boundary-safe tokenization
- BOS/EOS insertion
- prompt-first truncation
- chosen/rejected label construction
- concatenated chosen+rejected batching
- sequence log-prob extraction

This is the standard preference data path for beta-DPO and margin-DPO. The shared tokenization helper is intentionally grouped inside `tokenized_dpo_trainer.py` for now so it can be split later without being lost in generic utilities.

### 3. Runner-side orchestration

`utils/runtime.py` owns:

- logging setup
- checkpoint resume detection
- dataset loading + chat formatting orchestration
- model init kwargs
- final train/save/eval/push flow

`utils/checkpoint_io.py` owns:

- validated HF artifact saving
- model artifact upload
- margin summary dataset publishing

### 4. Algorithm layer

Each algorithm-specific trainer only owns the math that should differ:

- `simpo_trainer.py`: SimPO objective
- `alpha_dpo_trainer.py`: adaptive margin objective
- `beta_dpo_trainer.py`: reward-gap EMA, global subset sampling, adaptive beta
- `margin_dpo_trainer.py`: DPO-style margin logging and f-divergence projection

## Important Design Rules

### Keep the shared data path shared

If you change:

- `tokenize_row`
- `concatenated_inputs`
- `concatenated_forward`
- `get_batch_logps`

make the change in [scripts/tokenized_dpo_trainer.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/tokenized_dpo_trainer.py), not separately inside beta/margin.

### Keep algorithm logic local

If you change:

- beta-DPO masking / EMA / adaptive beta
- margin-DPO divergence projection or logging

make the change in the algorithm-specific trainer only.

### Do not reintroduce `simpo_config`

The repo now uses [scripts/trainer_configs.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/trainer_configs.py) as the real config module.

## Default Operating Procedure

### Smoke run first

Always validate on single GPU before distributed runs.

Beta-DPO smoke:

```bash
uv run python scripts/run_beta_dpo.py \
  training_configs/llama-3-8b-base-beta-dpo-smoke.yaml \
  --attn_implementation=sdpa
```

Margin-DPO smoke:

```bash
uv run python scripts/run_margin_dpo.py \
  training_configs/llama-3-8b-base-margin-dpo-smoke.yaml \
  --attn_implementation=sdpa
```

### Distributed smoke next

Use this to validate DDP behavior, especially for beta-DPO global mask logic:

```bash
uv run accelerate launch \
  --config_file accelerate_configs/ddp.yaml \
  --num_processes=4 \
  scripts/run_beta_dpo.py \
  training_configs/llama-3-8b-base-beta-dpo-smoke.yaml \
  --attn_implementation=sdpa
```

### Real FSDP run on 4 H100s

```bash
uv run accelerate launch \
  --config_file accelerate_configs/fsdp.yaml \
  --num_processes=4 \
  scripts/run_beta_dpo.py \
  training_configs/llama-3-8b-base-beta-dpo.yaml
```

and

```bash
uv run accelerate launch \
  --config_file accelerate_configs/fsdp.yaml \
  --num_processes=4 \
  scripts/run_margin_dpo.py \
  training_configs/llama-3-8b-base-margin-dpo.yaml
```

## FlashAttention Guidance

- Smoke runs: prefer `--attn_implementation=sdpa`
- Real H100 runs: `flash_attention_2` is reasonable if the environment supports it

Do not debug trainer logic and FlashAttention installation at the same time.

## Quantization Guidance

Quantization in this repo means low-bit model loading via `bitsandbytes`.

Use it only if you explicitly want:

- reduced memory footprint
- 4-bit / 8-bit loading
- QLoRA-style workflows

Do not use quantization by default for:

- smoke runs
- standard 4x H100 FSDP runs
## Validation Checklist

Minimum safe validation after code changes:

1. Syntax check:

```bash
uv run python -m py_compile \
  scripts/run_alpha_dpo.py \
  scripts/run_beta_dpo.py \
  scripts/run_margin_dpo.py \
  scripts/run_simpo.py \
  scripts/alpha_dpo_trainer.py \
  scripts/beta_dpo_trainer.py \
  scripts/margin_dpo_trainer.py \
  scripts/simpo_trainer.py \
  scripts/tokenized_dpo_trainer.py \
  scripts/trainer_configs.py \
  utils/runtime.py \
  utils/checkpoint_io.py \
  utils/dtypes.py \
  tools/pretokenize_preferences.py \
  tools/export_fsdp_checkpoint.py \
  tools/validate_and_upload.py
```

2. Single-GPU smoke run for the changed path
3. If beta-DPO distributed logic changed, run a 4-GPU beta-DPO smoke
4. Save the validation summary to [tests/restructure_validation.md](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/tests/restructure_validation.md)

## Known Caveat

This environment has previously shown an OpenMP shared-memory failure during `import torch`.

If you see:

- `OMP: Error #179`
- `Can't open SHM2 failed`

that is an environment/runtime issue, not necessarily a trainer logic issue.

## When Editing This Repo

- Prefer modifying shared behavior in `tokenized_dpo_trainer.py` or `utils/runtime.py`
- Keep config definitions centralized in `trainer_configs.py`
- Keep run commands and README instructions aligned with actual config filenames
- If you change training entrypoints or standard commands, update [README.md](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/README.md) too
