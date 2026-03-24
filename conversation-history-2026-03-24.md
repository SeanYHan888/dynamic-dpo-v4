# Conversation History

Date: 2026-03-24

This file is a shareable Markdown summary of the conversation and debugging work around beta-DPO, FSDP, preprocessing reuse, and smoke-test evaluation.

## 1. Code walkthrough and algorithm review

- Reviewed:
  - `scripts/run_beta_dpo.py`
  - `scripts/beta_dpo_trainer.py`
  - supporting trainer and preprocessing code
- Explained:
  - step-by-step control flow
  - beta-DPO loss construction
  - data flow from raw preference dataset to tokenized training batch
  - differences vs the original beta-DPO paper and the `beta-DPO/` reference repo

Key conclusions:

- `run_beta_dpo.py` is the orchestration entrypoint.
- `beta_dpo_trainer.py` implements:
  - EMA of reward-gap mean/std
  - Gaussian weighting and sample selection
  - adaptive beta scaling
  - masked DPO loss
- The new codebase differs from the original beta-DPO repo mainly in:
  - chat-template based preprocessing
  - more careful tokenization
  - more robust EMA/std handling
  - deterministic eval behavior
  - Hugging Face Trainer / Accelerate integration

## 2. Preprocessing and tokenization comparisons

- Compared the new preprocessing path with:
  - original beta-DPO preprocessing
  - alpha-DPO preprocessing in this repo

Key conclusions:

- New vs original beta-DPO:
  - not byte-for-byte equivalent before tokenization
  - not guaranteed token-for-token equivalent after tokenization
- New beta-DPO vs alpha-DPO in this repo:
  - preprocessing is effectively the same
  - differences are in the loss/training logic after forward pass

## 3. Replication and training recommendations

- Discussed what could change final model quality relative to the original beta-DPO method:
  - data serialization
  - tokenization/truncation policy
  - algorithmic microbatch size
  - EMA details
  - eval behavior
  - optimizer and warmup schedule
  - runtime stack differences

Recommendations made:

- Use conservative learning rate for UltraFeedback:
  - `learning_rate: 5e-7`
  - `warmup_ratio: 0.05`
- Keep the safer EMA implementation instead of reverting std init to `0`
- Treat beta-DPO batch size as the per-forward global microbatch, not the optimizer-step batch

## 4. Added test and run YAMLs

Created or updated these config files:

- `training_configs/llama-3-8b-base-beta-dpo-4xh100-test.yaml`
- `training_configs/llama-3-8b-base-beta-dpo-2gpu-smoke.yaml`
- `training_configs/llama-3-8b-base-beta-dpo-2gpu-stability-debug.yaml`
- `training_configs/llama-3-8b-base-beta-dpo-4gpu-first-real-run.yaml`

These were used to explore:

- 4xH100 test-run settings
- 2-GPU smoke-test stability
- adaptive-beta sensitivity (`alpha = 0.0`, `0.05`, `0.01`)
- recommended first 4-GPU real run settings

## 5. Preprocessing cache and tokenized dataset reuse

Implemented two opt-in features:

- persistent Hugging Face datasets cache
- tokenized dataset reuse with manifest-based validation

Files changed:

- `alignment/configs.py`
- `scripts/trainer_configs.py`
- `scripts/run_preference_utils.py`
- `scripts/tokenized_dpo_trainer.py`
- `scripts/simpo_trainer.py`
- `scripts/alpha_dpo_trainer.py`
- all current YAMLs under `training_configs/`

Shared helper module:

- moved to `utils/preprocessing_cache.py`

Behavior:

- `use_persistent_hf_cache: true` stores HF dataset cache under a stable repo-local location by default
- `reuse_tokenized_dataset: true` reuses saved tokenized datasets only if a manifest matches exactly
- manifest includes:
  - tokenizer/template info
  - preprocessing settings
  - code hashes
  - raw dataset fingerprint

## 6. FSDP, DeepSpeed, and FlashAttention debugging

We diagnosed multiple runtime issues:

### Path/config issues

- incorrect launch path:
  - `recipes/accelerate_configs/zero3.yaml` did not exist in this repo
- correct repo-local paths were:
  - `accelerate_configs/zero3.yaml`
  - `accelerate_configs/fsdp.yaml`

### DeepSpeed mismatch

- Installing `deepspeed==0.18.8` was incompatible with this repo's pinned stack.
- Repo expected `deepspeed==0.12.2`.

### FlashAttention ABI mismatch

- A rebuilt FlashAttention binary produced an undefined symbol error.
- Recommendation was to use `--attn_implementation=sdpa` first, and only reintroduce FlashAttention after the core training path was stable.

### FSDP frozen-reference issue

- Earlier FSDP runs failed due to reference-model device placement.
- Later code changes added a fix path in the trainer to prepare the reference model under FSDP correctly.

### Precomputed reference log-probs

- `precompute_ref_log_probs: true` was later enabled and wired, allowing:
  - a dataset-wide precompute pass for frozen-reference chosen/rejected log-probs
  - release of the live reference model during the actual training loop

This was confirmed to be normal behavior when that flag is enabled.

## 7. Beta-DPO smoke run analysis

Reviewed several output directories:

- `outputs/llama-3-8b-base-beta-dpo-4xh100-test`
- `outputs/llama-3-8b-base-beta-dpo-2gpu-smoke`
- `outputs/llama-3-8b-base-beta-dpo-2gpu-stability-debug`
- `outputs/llama-3-8b-base-beta-dpo-2gpu-stability-debug-new`
- `outputs/alpha0.01-llama-3-8b-base-beta-dpo-2gpu-stability-debug`

### Important conclusions from run inspection

#### 2-GPU smoke with original aggressive adaptive-beta settings

- Not healthy
- large losses and huge grad norms
- not close to original beta-DPO behavior

#### `alpha = 0.0` stability-debug run

- Much healthier as an engineering sanity check
- adaptive beta effectively disabled
- acceptable for validating the FSDP + precomputed-reference path

#### `alpha = 0.05` run

- Still too unstable
- adaptive beta already too aggressive for the 2-GPU tiny-batch regime

#### `alpha = 0.01` run

- Best adaptive-beta smoke run so far
- still not equivalent to original beta-DPO training
- but reasonable enough as a cheap 2-GPU adaptive-beta smoke test

## 8. Recommended first real 4-GPU run

Recommended config for the first serious 4-GPU run:

- `per_device_train_batch_size: 1`
- `per_device_eval_batch_size: 1`
- `gradient_accumulation_steps: 16`
- `beta: 0.1`
- `alpha: 0.01`
- `rho: 1.0`
- `ema_momentum: 0.9`
- `beta_min: 1e-3`
- `precompute_ref_log_probs: true`
- `use_persistent_hf_cache: true`
- `reuse_tokenized_dataset: true`
- `sync_global_mask: true`
- `require_equal_local_batch_size: true`
- `dataloader_drop_last: true`

Reasoning:

- `alpha: 0.01` was the first adaptive-beta setting that behaved reasonably in 2-GPU smoke tests
- `rho: 1.0` avoids per-rank drop behavior when the global microbatch is still small

## 9. Git and branch state

Current branch discussed:

- `codex/beta-margin-token-maping-save`

Committed and pushed:

- commit: `ff78405`
- message: `Add 2-GPU beta-DPO stability debug config`

This commit added:

- `training_configs/llama-3-8b-base-beta-dpo-2gpu-stability-debug.yaml`

## 10. Useful launch commands discussed

### 2-GPU smoke with FSDP and SDPA

```bash
ACCELERATE_LOG_LEVEL=info uv run accelerate launch \
  --config_file accelerate_configs/fsdp_smoke.yaml \
  --num_processes 2 \
  scripts/run_beta_dpo.py \
  training_configs/llama-3-8b-base-beta-dpo-2gpu-smoke.yaml \
  --attn_implementation=sdpa
```

### 2-GPU stability debug

```bash
ACCELERATE_LOG_LEVEL=info uv run accelerate launch \
  --config_file accelerate_configs/fsdp_smoke.yaml \
  --num_processes 2 \
  scripts/run_beta_dpo.py \
  training_configs/llama-3-8b-base-beta-dpo-2gpu-stability-debug.yaml
```

### First real 4-GPU run

```bash
ACCELERATE_LOG_LEVEL=info uv run accelerate launch \
  --config_file accelerate_configs/fsdp.yaml \
  --num_processes 4 \
  scripts/run_beta_dpo.py \
  training_configs/llama-3-8b-base-beta-dpo-4gpu-first-real-run.yaml
```

## 11. Final takeaway

The main conclusions from this conversation were:

- FSDP + precomputed reference log-probs is the practical path for this beta-DPO setup.
- 2-GPU smoke runs are useful for code-path validation, but not faithful to the original beta-DPO batch regime.
- Adaptive beta is very sensitive in tiny-batch settings.
- `alpha: 0.01` is the best smoke-test adaptation setting observed so far.
- The first serious 4-GPU run should prioritize stability over aggressive adaptation.
