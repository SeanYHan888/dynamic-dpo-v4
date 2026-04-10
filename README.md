# Dynamic DPO v4

This repository contains preference-training code for:

- SimPO
- IPO
- Robust DPO
- CPO
- KTO
- ORPO
- SLiC-HF
- alpha-DPO
- beta-DPO
- margin-DPO

For `KTO` on `HuggingFaceH4/ultrafeedback_binarized`, the repo uses a pairwise-derived adaptation rather than a native unary binary-feedback dataset: each `(prompt, chosen, rejected)` example is expanded into `(prompt, chosen, True)` and `(prompt, rejected, False)` rows before being passed to TRL 0.10.1's `KTOTrainer`.

The main training entrypoints live in [scripts](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts), shared runner-side infrastructure lives in [utils](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/utils), ops workflows live in [tools](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/tools), and runnable YAML configs live in [training_configs](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/training_configs). The inherited [alignment](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/alignment) package is intentionally kept unchanged.

## Setup

From the repo root:

```bash
uv sync
```

`wandb` is part of the standard dependency set, so a plain `uv sync` installs it.
`hf-transfer` is also installed by default, and the codebase enables `HF_HUB_ENABLE_HF_TRANSFER=1` automatically so Hub downloads use parallel transfer by default.

If you are preparing for a real Linux H100 run with FlashAttention 2, use:

```bash
uv sync --extra ultra
```

The `ultra` extra installs `flash-attn==2.5.8` and is Linux-only.

If you are using Hugging Face Hub or Weights & Biases, make sure you are already authenticated in the shell environment where you launch runs.

## Repo Layout

- [scripts](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/scripts): flat training layer with runners, trainers, and `trainer_configs.py`
- [utils/runtime.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/utils/runtime.py): runner-oriented setup, dataset prep, model init kwargs, and final training lifecycle helpers
- [utils/checkpoint_io.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/utils/checkpoint_io.py): validated save/export/upload helpers for model artifacts and margin summaries
- [utils/preprocessing_cache.py](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/utils/preprocessing_cache.py): tokenized dataset cache and reuse logic
- [tools](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/tools): CLI workflows for offline tokenization, FSDP export, and checkpoint validation/upload
- [training_configs](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/training_configs): runnable training YAML configs
- [accelerate_configs](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/accelerate_configs): accelerate launch configs
- [tests/restructure_validation.md](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/tests/restructure_validation.md): saved validation report after the layout restructure

## Step-by-Step Usage

### 1. Sanity Check the Environment

Run this once after syncing dependencies:

```bash
uv run python -m py_compile \
  scripts/run_beta_dpo.py \
  scripts/run_margin_dpo.py \
  scripts/run_alpha_dpo.py \
  scripts/run_simpo.py \
  scripts/beta_dpo_trainer.py \
  scripts/margin_dpo_trainer.py \
  scripts/tokenized_dpo_trainer.py \
  utils/runtime.py \
  utils/checkpoint_io.py \
  utils/dtypes.py \
  tools/pretokenize_preferences.py \
  tools/export_fsdp_checkpoint.py \
  tools/validate_and_upload.py
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
Parallel Hub download is already enabled by default after `uv sync`.

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

Before a real H100 FSDP run on Linux, install FlashAttention 2 with:

```bash
uv sync --extra ultra
```

That command keeps the default parallel download behavior and adds FlashAttention 2 on top.

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

## Validation Reports

After structural refactors, save the validation summary under [tests/restructure_validation.md](/Users/seanmacbook/Research/dpo/dynamic-dpo-v4/tests/restructure_validation.md) so syntax checks, test runs, and import smoke checks are recorded in-repo.
