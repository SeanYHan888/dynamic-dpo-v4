# Scripts

This directory is intentionally flat and contains only training entrypoints, trainer implementations, and trainer config dataclasses.

## Runners

- `run_simpo.py`
- `run_alpha_dpo.py`
- `run_beta_dpo.py`
- `run_epsilon_dpo.py`
- `run_margin_dpo.py`
- `run_sft.py`

Each runner owns CLI orchestration only:

- parse YAML/CLI config
- call shared runner-side helpers from `utils/runtime.py`
- instantiate the correct trainer
- start training/eval/save flow

## Trainers

- `simpo_trainer.py`
- `alpha_dpo_trainer.py`
- `beta_dpo_trainer.py`
- `epsilon_dpo_trainer.py`
- `margin_dpo_trainer.py`
- `tokenized_dpo_trainer.py`

`tokenized_dpo_trainer.py` is the shared DPO-style trainer base. It also temporarily contains the shared preference tokenization helper section so tokenization stays grouped with trainer behavior and can be split later if needed.

## Configs

- `trainer_configs.py`

This file contains repo-specific training config dataclasses layered on top of `alignment.DPOConfig`.

## What Does Not Belong Here

The following live outside `scripts/`:

- `utils/`: shared runner-side runtime, dtype helpers, preprocessing cache, checkpoint save/upload logic
- `tools/`: offline tokenization, FSDP export, checkpoint validation/upload workflows
- `alignment/`: inherited infrastructure kept unchanged
