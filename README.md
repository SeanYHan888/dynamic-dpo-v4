# Dynamic DPO v4

This repository contains preference-training code for:

- SimPO
- alpha-DPO
- beta-DPO
- margin-DPO

The main training entrypoints live in [scripts](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts), and the runnable YAML configs live in [training_configs](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/training_configs).

## Setup

From the repo root:

```bash
uv sync
```

`wandb` is part of the standard dependency set, so a plain `uv sync` installs it.

If you are using Hugging Face Hub or Weights & Biases, make sure you are already authenticated in the shell environment where you launch runs.

## Repo Layout

- [scripts/run_simpo.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/run_simpo.py): SimPO training entrypoint
- [scripts/run_alpha_dpo.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/run_alpha_dpo.py): SimPO / alpha-DPO entrypoint
- [scripts/run_beta_dpo.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/run_beta_dpo.py): beta-DPO entrypoint
- [scripts/run_margin_dpo.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts/run_margin_dpo.py): margin-DPO entrypoint
- [training_configs/llama-3-8b-base-beta-dpo-smoke.yaml](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/training_configs/llama-3-8b-base-beta-dpo-smoke.yaml): beta-DPO smoke config
- [training_configs/llama-3-8b-base-margin-dpo-smoke.yaml](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/training_configs/llama-3-8b-base-margin-dpo-smoke.yaml): margin-DPO smoke config
- [training_configs/llama-3-8b-base-beta-dpo.yaml](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/training_configs/llama-3-8b-base-beta-dpo.yaml): beta-DPO full config
- [training_configs/llama-3-8b-base-margin-dpo.yaml](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/training_configs/llama-3-8b-base-margin-dpo.yaml): margin-DPO full config
- [accelerate_configs/fsdp.yaml](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/accelerate_configs/fsdp.yaml): base FSDP accelerate config

## Step-by-Step Usage

### 1. Sanity Check the Environment

Run this once after syncing dependencies:

```bash
uv run python -m py_compile \
  scripts/run_beta_dpo.py \
  scripts/run_margin_dpo.py \
  scripts/beta_dpo_trainer.py \
  scripts/margin_dpo_trainer.py
```

This does a syntax-level check without starting a real training run.

### 2. Run a Single-GPU Smoke Test

Use single GPU first. The goal is to verify:

- config parsing works
- dataset loading and chat templating work
- tokenization works
- trainer construction works
- forward/backward starts successfully

For smoke runs, prefer `sdpa` instead of FlashAttention to reduce runtime complexity.

#### Beta-DPO smoke run

```bash
uv run python scripts/run_beta_dpo.py \
  training_configs/llama-3-8b-base-beta-dpo-smoke.yaml \
  --attn_implementation=sdpa
```

#### Margin-DPO smoke run

```bash
uv run python scripts/run_margin_dpo.py \
  training_configs/llama-3-8b-base-margin-dpo-smoke.yaml \
  --attn_implementation=sdpa
```

Smoke configs are already reduced:

- 10% dataset fraction
- `max_steps: 20`
- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 1`
- `do_eval: false`
- `save_strategy: "no"`

### 3. Inspect Smoke Outputs

Expected output locations:

- [outputs/llama-3-8b-base-beta-dpo-smoke](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/outputs/llama-3-8b-base-beta-dpo-smoke)
- [outputs/llama-3-8b-base-margin-dpo-smoke](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/outputs/llama-3-8b-base-margin-dpo-smoke)

For margin-DPO, also check:

- [outputs/llama-3-8b-base-margin-dpo-smoke/margin_logs/margins.jsonl](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/outputs/llama-3-8b-base-margin-dpo-smoke/margin_logs/margins.jsonl)

### 4. Run a Distributed Smoke Test

After the single-GPU smoke passes, validate distributed behavior.

This is more important for beta-DPO than margin-DPO because beta-DPO uses:

- cross-rank reward-gap gathering
- synchronized subset masking
- local slicing of a global mask

#### 4-GPU beta-DPO smoke

```bash
uv run accelerate launch \
  --config_file accelerate_configs/ddp.yaml \
  --num_processes=4 \
  scripts/run_beta_dpo.py \
  training_configs/llama-3-8b-base-beta-dpo-smoke.yaml \
  --attn_implementation=sdpa
```

#### 4-GPU margin-DPO smoke

```bash
uv run accelerate launch \
  --config_file accelerate_configs/ddp.yaml \
  --num_processes=4 \
  scripts/run_margin_dpo.py \
  training_configs/llama-3-8b-base-margin-dpo-smoke.yaml \
  --attn_implementation=sdpa
```

## Real FSDP Runs on 4 H100s

For real training on 4 H100s, use `accelerate` with the FSDP config and override the process count to 4.

The checked-in [accelerate_configs/fsdp.yaml](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/accelerate_configs/fsdp.yaml) has `num_processes: 8`, so pass `--num_processes=4` explicitly for a 4-GPU run.

For real runs, keeping `flash_attention_2` is reasonable on H100 unless you are debugging environment issues.

### Beta-DPO FSDP run

```bash
uv run accelerate launch \
  --config_file accelerate_configs/fsdp.yaml \
  --num_processes=4 \
  scripts/run_beta_dpo.py \
  training_configs/llama-3-8b-base-beta-dpo.yaml
```

### Margin-DPO FSDP run

```bash
uv run accelerate launch \
  --config_file accelerate_configs/fsdp.yaml \
  --num_processes=4 \
  scripts/run_margin_dpo.py \
  training_configs/llama-3-8b-base-margin-dpo.yaml
```

## Recommended Execution Order

Use this order:

1. `uv sync`
2. Single-GPU beta-DPO smoke
3. Single-GPU margin-DPO smoke
4. 4-GPU beta-DPO smoke
5. 4-GPU margin-DPO smoke if you want distributed confirmation
6. Real 4x H100 FSDP beta-DPO or margin-DPO run

## Notes

### FlashAttention

For smoke runs:

- prefer `--attn_implementation=sdpa`

For real H100 training:

- default `flash_attention_2` is appropriate if the environment supports it

### Quantization

This repo supports low-bit loading through the quantization path in the model utilities, but it is not required for standard smoke runs or for standard 4x H100 FSDP runs.

Use quantization only if you explicitly want low-memory loading or a QLoRA-style setup.

### Current Environment Caveat

If the environment throws an OpenMP / shared-memory initialization error during `import torch`, that is an environment issue rather than a trainer-logic issue. In that case, fix the runtime environment first before using smoke-run failures as evidence of trainer breakage.
