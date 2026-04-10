# Epsilon-DPO Parity Checklist

Use this checklist before trusting that the repo-native `scripts/epsilon_dpo_trainer.py` is behaviorally aligned with `reference-codes/e-dpo`.

## Preconditions

1. Create two Python environments:
   - Current repo env: this repo's stack (`trl==0.10.1`).
   - Reference env: `reference-codes/e-dpo` stack (`trl==0.13.0`).
2. Pick one shared debug model checkpoint that both environments can load.
   - Prefer a very small causal LM for trajectory checks.
   - Use the same tokenizer in both stacks.
3. Disable training-time randomness where possible.
   - Keep dropout disabled.
   - Use deterministic small batches.
   - Do not enable PEFT, quantization, or custom kernels for the first parity pass.

## What To Lock Down

1. Preprocessing parity:
   - Use the repo harness to preprocess config-backed datasets once with the current stack.
   - Feed the resulting `prompt/chosen/rejected` strings into both trainers.
2. Model/init parity:
   - Same `model_name_or_path`.
   - Same tokenizer path.
   - Same `beta`, `epsilon`, batch size, gradient accumulation, and learning rate.
3. Runtime parity:
   - Same optimizer class and hyperparameters.
   - Same ordered microbatches.
   - Same number of optimizer steps.

## Frozen-Batch Parity

Run this first. It checks token scores, ε-steps, rewards, losses, and beta boundary updates on the exact same batch.

```bash
python tools/epsilon_dpo_parity.py \
  --mode compare \
  --reference-python /path/to/reference-env/bin/python \
  --model-name-or-path <shared-debug-model> \
  --examples-json '[{"prompt":"User: name a prime number.\nAssistant:","chosen":" 13","rejected":" 12"},{"prompt":"User: reply with yes or no.\nAssistant:","chosen":" yes","rejected":" maybe"}]'
```

## Config-Dataset Trajectory Parity

Run this second. It reuses repo preprocessing on a real config dataset, then compares a short deterministic training trajectory across both trainers.

Recommended first pass:

```bash
python tools/epsilon_dpo_parity.py \
  --mode compare_trajectory \
  --reference-python /path/to/reference-env/bin/python \
  --config-path training_configs/llama3-8b-base/e-dpo/llama-3-8b-base-epsilon-dpo-helpful.yaml \
  --model-name-or-path <shared-debug-model> \
  --max-examples 4 \
  --train-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --trajectory-optimizer-steps 2 \
  --learning-rate 1e-6
```

What this proves:

1. The same config-backed preprocessing yields the same effective training examples.
2. Per-microbatch losses and logged metrics match.
3. Beta evolves identically across training calls.
4. Final trainable parameter tensors match after the debug optimizer steps.

## Pass Criteria

Treat parity as passing only if all of the following hold:

1. Frozen-batch compare passes with no tensor mismatches.
2. Short trajectory compare passes with no mismatches in:
   - per-microbatch loss
   - per-microbatch metrics
   - beta before/after each update
   - per-step model-state summaries
   - final trainable parameter tensors
3. The config-dataset trajectory check passes with the exact tokenizer/chat-template path you plan to use for real training.

## Escalation Path

If the short debug trajectory passes, then increase one axis at a time:

1. More optimizer steps.
2. Config-native gradient accumulation.
3. The actual target tokenizer/model checkpoint.
4. Only after that: PEFT, quantization, alternate attention kernels, or larger batch sizes.

If parity fails, debug in this order:

1. Example text after preprocessing.
2. Token boundaries between prompt and completion.
3. Sequence log-probs on one frozen batch.
4. ε-step decisions.
5. Beta update timing across accumulation boundaries.

## Shared-Batch Scoring Debug

If frozen-batch or trajectory parity fails, run one instrumented shared-batch debug pass:

```bash
python tools/epsilon_dpo_parity.py \
  --mode compare_scoring_debug \
  --reference-python /path/to/reference-env/bin/python \
  --config-path training_configs/llama3-8b-base/e-dpo/llama-3-8b-base-epsilon-dpo-helpful.yaml \
  --model-name-or-path <shared-debug-model> \
  --max-examples 2 \
  --train-batch-size 2 \
  --gradient-accumulation-steps 1
```

This prints a summary containing:

1. Whether prompt/completion token views match.
2. The first mismatch in loss-bearing token ids, if any.
3. The first mismatch in per-token log-probs, if any.

The underlying reference/current debug dumps are saved to temporary `.pt` files and their paths are printed in the summary.
