#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_script/e_dpo/run_llama3_epsilon_dpo.sh <ultrafeedback|hh-helpful|hh-harmless> [extra run_epsilon_dpo args...]

Environment overrides:
  SCRATCH_ROOT                       Default: /scratch/$USER/dynamic-dpo-v4
  HF_REPO_ID                         Override the upload target repo id
  RUN                                Override the generated run name
  MODEL_NAME_OR_PATH                 Override the policy/reference model source
  SOURCE_MODEL_ID                    Override the expected Hub model id used in the model card
  SAVE_STEPS                         Default: 200
  SAVE_TOTAL_LIMIT                   Default: 2
  SKIP_UPLOAD                        Set to 1 to validate only
  DELETE_STALE_REMOTE_ARTIFACTS      Set to 1 to delete stale remote model files before upload
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

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}/dynamic-dpo-v4}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
DELETE_STALE_REMOTE_ARTIFACTS="${DELETE_STALE_REMOTE_ARTIFACTS:-0}"

case "$VARIANT" in
  ultrafeedback)
    CONFIG_PATH="training_configs/llama3-8b-base/e-dpo/llama-3-8b-base-epsilon-dpo-ultrafeedback.yaml"
    RUN_PREFIX="llama-3-8b-base-epsilon-dpo-ultrafeedback-8xh200"
    DEFAULT_SOURCE_MODEL_ID="W-61/llama-3-8b-base-sft-ultrachat-8xh200"
    DEFAULT_REPO_ID="W-61/llama-3-8b-base-epsilon-dpo-ultrafeedback-8xh200"
    ;;
  hh-helpful)
    CONFIG_PATH="training_configs/llama3-8b-base/e-dpo/llama-3-8b-base-epsilon-dpo-helpful.yaml"
    RUN_PREFIX="llama-3-8b-base-epsilon-dpo-hh-helpful-8xh200"
    DEFAULT_SOURCE_MODEL_ID="W-61/llama-3-8b-base-sft-hh-helpful-8xh200"
    DEFAULT_REPO_ID="W-61/llama-3-8b-base-epsilon-dpo-hh-helpful-8xh200"
    ;;
  hh-harmless)
    CONFIG_PATH="training_configs/llama3-8b-base/e-dpo/llama-3-8b-base-epsilon-dpo-harmless.yaml"
    RUN_PREFIX="llama-3-8b-base-epsilon-dpo-hh-harmless-8xh200"
    DEFAULT_SOURCE_MODEL_ID="W-61/llama-3-8b-base-sft-hh-harmless-8xh200"
    DEFAULT_REPO_ID="W-61/llama-3-8b-base-epsilon-dpo-hh-harmless-8xh200"
    ;;
  *)
    echo "Unknown Epsilon-DPO variant: $VARIANT" >&2
    usage
    exit 1
    ;;
esac

SOURCE_MODEL_ID="${SOURCE_MODEL_ID:-$DEFAULT_SOURCE_MODEL_ID}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$SOURCE_MODEL_ID}"
HF_REPO_ID="${HF_REPO_ID:-$DEFAULT_REPO_ID}"
if [[ -n "${RUN:-}" ]]; then
  if [[ "$RUN" == "$RUN_PREFIX"* ]]; then
    RUN="$RUN"
  else
    echo "Ignoring inherited RUN='$RUN' because it does not match expected prefix '$RUN_PREFIX'." >&2
    RUN="$RUN_PREFIX-$(date +%Y%m%d-%H%M%S)"
  fi
else
  RUN="$RUN_PREFIX-$(date +%Y%m%d-%H%M%S)"
fi
RUN_DIR="$SCRATCH_ROOT/outputs/$RUN"
LOG_PATH="$RUN_DIR/train.log"
HF_DATASET_CACHE_DIR="$SCRATCH_ROOT/hf/datasets"
TOKENIZED_CACHE_DIR="$SCRATCH_ROOT/tokenized_preferences"

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

validate_hf_auth() {
  if [[ "${HF_CLI[0]}" == "huggingface-cli" ]]; then
    "${HF_CLI[@]}" whoami >/dev/null
  else
    "${HF_CLI[@]}" auth whoami >/dev/null
  fi
}

patch_model_card_base_model() {
  local readme_path="$1"
  local base_model_id="$2"
  python - "$readme_path" "$base_model_id" <<'PY'
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

mkdir -p "$SCRATCH_ROOT"/{hf,tmp,wandb,xdg,outputs}
mkdir -p "$TOKENIZED_CACHE_DIR"
mkdir -p "$RUN_DIR"

export HF_HOME="$SCRATCH_ROOT/hf"
export HF_HUB_CACHE="$SCRATCH_ROOT/hf/hub"
export HF_DATASETS_CACHE="$HF_DATASET_CACHE_DIR"
export TMPDIR="$SCRATCH_ROOT/tmp"
export WANDB_DIR="$SCRATCH_ROOT/wandb"
export XDG_CACHE_HOME="$SCRATCH_ROOT/xdg"
export PYTHONUNBUFFERED=1
unset TRANSFORMERS_CACHE || true

echo "Starting $VARIANT Epsilon-DPO run: $RUN"
echo "Config: $CONFIG_PATH"
echo "Model source: $MODEL_NAME_OR_PATH"
echo "Output dir: $RUN_DIR"
echo "Hub repo: $HF_REPO_ID"

accelerate launch \
  --config_file accelerate_configs/fsdp.yaml \
  scripts/run_epsilon_dpo.py \
  "$CONFIG_PATH" \
  --model_name_or_path="$MODEL_NAME_OR_PATH" \
  --output_dir="$RUN_DIR" \
  --run_name="$RUN" \
  --hf_cache_dir="$HF_DATASET_CACHE_DIR" \
  --tokenized_dataset_cache_dir="$TOKENIZED_CACHE_DIR" \
  --save_strategy=steps \
  --save_steps="$SAVE_STEPS" \
  --save_total_limit="$SAVE_TOTAL_LIMIT" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_PATH"

python tools/validate_and_upload.py \
  --checkpoint-dir "$RUN_DIR" \
  --skip-upload

patch_model_card_base_model "$RUN_DIR/README.md" "$SOURCE_MODEL_ID"

if [[ "$SKIP_UPLOAD" == "1" ]]; then
  echo "Validation completed. Skipping upload by request."
  exit 0
fi

UPLOAD_ARGS=(
  python tools/validate_and_upload.py
  --checkpoint-dir "$RUN_DIR"
  --repo-id "$HF_REPO_ID"
  --commit-message "Upload validated Epsilon-DPO checkpoint"
)

if [[ "$DELETE_STALE_REMOTE_ARTIFACTS" == "1" ]]; then
  UPLOAD_ARGS+=(--delete-stale-remote-artifacts)
fi

"${UPLOAD_ARGS[@]}"

echo "Completed training, validation, and upload for $RUN"
