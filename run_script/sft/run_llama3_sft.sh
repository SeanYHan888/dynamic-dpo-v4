#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_script/sft/run_llama3_sft.sh <ultrachat|hh-helpful|hh-harmless> [extra run_sft args...]

Environment overrides:
  SCRATCH_ROOT                       Default: /scratch/$USER/dynamic-dpo-v4
  BASE_MODEL_ID                      Default: meta-llama/Meta-Llama-3-8B
  BASE_MODEL_DIR                     Default: $SCRATCH_ROOT/base_models/Meta-Llama-3-8B
  HF_REPO_ID                         Override the upload target repo id
  RUN                                Override the generated run name
  SAVE_STEPS                         Default: 200
  SAVE_TOTAL_LIMIT                   Default: 2
  SKIP_UPLOAD                        Set to 1 to validate only
  DELETE_STALE_REMOTE_ARTIFACTS      Set to 1 to delete stale remote model files before upload
  DOWNLOAD_BASE_MODEL_IF_MISSING     Default: 1
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

VARIANT="$1"
shift
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_MODEL_ID="${BASE_MODEL_ID:-meta-llama/Meta-Llama-3-8B}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}/dynamic-dpo-v4}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-$SCRATCH_ROOT/base_models/Meta-Llama-3-8B}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
DELETE_STALE_REMOTE_ARTIFACTS="${DELETE_STALE_REMOTE_ARTIFACTS:-0}"
DOWNLOAD_BASE_MODEL_IF_MISSING="${DOWNLOAD_BASE_MODEL_IF_MISSING:-1}"

case "$VARIANT" in
  ultrachat)
    CONFIG_PATH="training_configs/llama3-8b-base/sft/llama-3-8b-base-sft-ultrachat.yaml"
    RUN_PREFIX="llama-3-8b-base-sft-ultrachat-8xh200"
    DEFAULT_REPO_ID="W-61/llama-3-8b-base-sft-ultrachat-8xh200"
    ;;
  hh-helpful)
    CONFIG_PATH="training_configs/llama3-8b-base/sft/llama-3-8b-base-sft-hh-helpful.yaml"
    RUN_PREFIX="llama-3-8b-base-sft-hh-helpful-8xh200"
    DEFAULT_REPO_ID="W-61/llama-3-8b-base-sft-hh-helpful-8xh200"
    ;;
  hh-harmless)
    CONFIG_PATH="training_configs/llama3-8b-base/sft/llama-3-8b-base-sft-hh-harmless.yaml"
    RUN_PREFIX="llama-3-8b-base-sft-hh-harmless-8xh200"
    DEFAULT_REPO_ID="W-61/llama-3-8b-base-sft-hh-harmless-8xh200"
    ;;
  *)
    echo "Unknown SFT variant: $VARIANT" >&2
    usage
    exit 1
    ;;
esac

HF_REPO_ID="${HF_REPO_ID:-$DEFAULT_REPO_ID}"
RUN="${RUN:-$RUN_PREFIX-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$SCRATCH_ROOT/outputs/$RUN"
LOG_PATH="$RUN_DIR/train.log"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_hf_cli() {
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI=(huggingface-cli)
  elif command -v hf >/dev/null 2>&1; then
    HF_CLI=(hf)
  else
    echo "Missing Hugging Face CLI. Install either 'huggingface-cli' or 'hf'." >&2
    exit 1
  fi
}

is_base_model_complete() {
  local required=(
    "$BASE_MODEL_DIR/config.json"
    "$BASE_MODEL_DIR/model.safetensors.index.json"
    "$BASE_MODEL_DIR/tokenizer.json"
    "$BASE_MODEL_DIR/tokenizer_config.json"
    "$BASE_MODEL_DIR/special_tokens_map.json"
  )
  local path
  for path in "${required[@]}"; do
    [[ -f "$path" ]] || return 1
  done
  compgen -G "$BASE_MODEL_DIR/model-*.safetensors" >/dev/null
}

download_base_model() {
  mkdir -p "$(dirname "$BASE_MODEL_DIR")"
  if [[ "${HF_CLI[0]}" == "huggingface-cli" ]]; then
    "${HF_CLI[@]}" download "$BASE_MODEL_ID" --local-dir "$BASE_MODEL_DIR"
  else
    "${HF_CLI[@]}" download "$BASE_MODEL_ID" --local-dir "$BASE_MODEL_DIR"
  fi
}

validate_hf_auth() {
  if [[ "${HF_CLI[0]}" == "huggingface-cli" ]]; then
    "${HF_CLI[@]}" whoami >/dev/null
  else
    "${HF_CLI[@]}" auth whoami >/dev/null
  fi
}

patch_model_card_base_model() {
  local readme_path="$1"
  python - "$readme_path" "$BASE_MODEL_ID" <<'PY'
from pathlib import Path
import sys

readme_path = Path(sys.argv[1])
base_model_id = sys.argv[2]

if not readme_path.exists():
    raise SystemExit(f"README.md not found: {readme_path}")

lines = readme_path.read_text(encoding="utf-8").splitlines()
updated = False
for idx, line in enumerate(lines):
    if line.startswith("base_model: "):
        lines[idx] = f"base_model: {base_model_id}"
        updated = True
        break

if updated:
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

require_command accelerate
require_command python
require_hf_cli
validate_hf_auth

mkdir -p "$SCRATCH_ROOT"/{hf,tmp,wandb,xdg,outputs,base_models}
mkdir -p "$RUN_DIR"

export HF_HOME="$SCRATCH_ROOT/hf"
export HF_HUB_CACHE="$SCRATCH_ROOT/hf/hub"
export HF_DATASETS_CACHE="$SCRATCH_ROOT/hf/datasets"
export TMPDIR="$SCRATCH_ROOT/tmp"
export WANDB_DIR="$SCRATCH_ROOT/wandb"
export XDG_CACHE_HOME="$SCRATCH_ROOT/xdg"
export PYTHONUNBUFFERED=1
unset TRANSFORMERS_CACHE || true

if ! is_base_model_complete; then
  if [[ "$DOWNLOAD_BASE_MODEL_IF_MISSING" != "1" ]]; then
    echo "Base model is missing or incomplete at $BASE_MODEL_DIR" >&2
    exit 1
  fi
  echo "Downloading $BASE_MODEL_ID to $BASE_MODEL_DIR"
  download_base_model
fi

if ! is_base_model_complete; then
  echo "Base model is still incomplete after download: $BASE_MODEL_DIR" >&2
  exit 1
fi

echo "Starting $VARIANT SFT run: $RUN"
echo "Config: $CONFIG_PATH"
echo "Output dir: $RUN_DIR"
echo "Hub repo: $HF_REPO_ID"

accelerate launch \
  --config_file accelerate_configs/fsdp.yaml \
  scripts/run_sft.py \
  "$CONFIG_PATH" \
  --model_name_or_path="$BASE_MODEL_DIR" \
  --output_dir="$RUN_DIR" \
  --run_name="$RUN" \
  --save_strategy=steps \
  --save_steps="$SAVE_STEPS" \
  --save_total_limit="$SAVE_TOTAL_LIMIT" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_PATH"

python tools/validate_and_upload.py \
  --checkpoint-dir "$RUN_DIR" \
  --skip-upload

patch_model_card_base_model "$RUN_DIR/README.md"

if [[ "$SKIP_UPLOAD" == "1" ]]; then
  echo "Validation completed. Skipping upload by request."
  exit 0
fi

UPLOAD_ARGS=(
  python tools/validate_and_upload.py
  --checkpoint-dir "$RUN_DIR"
  --repo-id "$HF_REPO_ID"
  --commit-message "Upload validated SFT checkpoint"
)

if [[ "$DELETE_STALE_REMOTE_ARTIFACTS" == "1" ]]; then
  UPLOAD_ARGS+=(--delete-stale-remote-artifacts)
fi

"${UPLOAD_ARGS[@]}"

echo "Completed training, validation, and upload for $RUN"
