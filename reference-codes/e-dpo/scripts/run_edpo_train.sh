#!/usr/bin/env bash
set -euo pipefail

# from repo root
if [ ! -d .venv ]; then
  if command -v python3.10 >/dev/null 2>&1; then
    python3.10 -m venv .venv
  else
    echo "python3.10 not found. This repo expects Python 3.10."
    echo "Install Python 3.10, then rerun this script."
    exit 1
  fi
fi
source .venv/bin/activate

if ! python -c "import accelerate, deepspeed, trl, transformers, datasets, omegaconf" >/dev/null 2>&1; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

export HF_TOKEN="hf_your_token_here"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="your_wandb_key_here"
export WANDB_PROJECT="e-dpo-experiment"

NUM_PROCESSES=$(nvidia-smi -L | wc -l | tr -d ' ')
python -m accelerate.commands.launch \
  --config_file configs/accelerate.yaml \
  --num_processes "${NUM_PROCESSES}" \
  train.py \
  --config configs/llama3_instruct.yaml
