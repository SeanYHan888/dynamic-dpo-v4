#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: dpo_offline/run_llama3_margin_dpo.sh <ultrafeedback|hh-helpful|hh-harmless> [extra run_margin_dpo args...]

Offline defaults:
  ultrafeedback -> latest ultrachat SFT checkpoint under /scratch/$USER/dynamic-dpo-v4/outputs
  hh-helpful    -> latest hh-helpful SFT checkpoint under /scratch/$USER/dynamic-dpo-v4/outputs
  hh-harmless   -> latest hh-harmless SFT checkpoint under /scratch/$USER/dynamic-dpo-v4/outputs

Environment overrides:
  SCRATCH_ROOT                       Default: /scratch/$USER/dynamic-dpo-v4
  MODEL_NAME_OR_PATH                 Override the local policy/reference model directory
  LOCAL_SFT_MODEL_DIR                Alias for MODEL_NAME_OR_PATH
  ULTRAFEEDBACK_SFT_DIR              Override the default offline ultrafeedback source checkpoint
  HH_HELPFUL_SFT_DIR                 Override the default offline hh-helpful source checkpoint
  HH_HARMLESS_SFT_DIR                Override the default offline hh-harmless source checkpoint
  SOURCE_MODEL_ID                    Override the model card base_model metadata
  HF_REPO_ID                         Override the upload target repo id
  RUN                                Override the generated run name
  SAVE_STEPS                         Default: 200
  SAVE_TOTAL_LIMIT                   Default: 2
  SKIP_UPLOAD                        Default: 0
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
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}/dynamic-dpo-v4}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
DELETE_STALE_REMOTE_ARTIFACTS="${DELETE_STALE_REMOTE_ARTIFACTS:-0}"

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

is_local_model_complete() {
  local model_dir="$1"
  local required=(
    "$model_dir/config.json"
    "$model_dir/model.safetensors.index.json"
    "$model_dir/tokenizer.json"
    "$model_dir/tokenizer_config.json"
    "$model_dir/special_tokens_map.json"
  )
  local path
  for path in "${required[@]}"; do
    [[ -f "$path" ]] || return 1
  done
  compgen -G "$model_dir/model-*.safetensors" >/dev/null
}

latest_matching_dir() {
  local pattern="$1"
  local matches=()
  local path

  while IFS= read -r path; do
    matches+=("$path")
  done < <(compgen -G "$pattern" | sort)

  if [[ ${#matches[@]} -eq 0 ]]; then
    return 1
  fi

  printf '%s\n' "${matches[$((${#matches[@]} - 1))]}"
}

case "$VARIANT" in
  ultrafeedback)
    CONFIG_PATH="training_configs/llama3-8b-base/dpo/llama-3-8b-base-margin-dpo-ultrafeedback.yaml"
    RUN_PREFIX="llama-3-8b-base-margin-dpo-ultrafeedback-8xh200"
    DEFAULT_REPO_ID="W-61/llama-3-8b-base-margin-dpo-ultrafeedback-8xh200"
    VARIANT_MODEL_ENV="${ULTRAFEEDBACK_SFT_DIR:-}"
    DEFAULT_MODEL_GLOB="$SCRATCH_ROOT/outputs/llama-3-8b-base-sft-ultrachat-8xh200-*"
    ;;
  hh-helpful)
    CONFIG_PATH="training_configs/llama3-8b-base/dpo/llama-3-8b-base-margin-dpo-hh-helpful.yaml"
    RUN_PREFIX="llama-3-8b-base-margin-dpo-hh-helpful-8xh200"
    DEFAULT_REPO_ID="W-61/llama-3-8b-base-margin-dpo-hh-helpful-8xh200"
    VARIANT_MODEL_ENV="${HH_HELPFUL_SFT_DIR:-}"
    DEFAULT_MODEL_GLOB="$SCRATCH_ROOT/outputs/llama-3-8b-base-sft-hh-helpful-8xh200-*"
    ;;
  hh-harmless)
    CONFIG_PATH="training_configs/llama3-8b-base/dpo/llama-3-8b-base-margin-dpo-hh-harmless.yaml"
    RUN_PREFIX="llama-3-8b-base-margin-dpo-hh-harmless-8xh200"
    DEFAULT_REPO_ID="W-61/llama-3-8b-base-margin-dpo-hh-harmless-8xh200"
    VARIANT_MODEL_ENV="${HH_HARMLESS_SFT_DIR:-}"
    DEFAULT_MODEL_GLOB="$SCRATCH_ROOT/outputs/llama-3-8b-base-sft-hh-harmless-8xh200-*"
    ;;
  *)
    echo "Unknown Margin-DPO variant: $VARIANT" >&2
    usage
    exit 1
    ;;
esac

AUTO_DISCOVERED_MODEL_DIR=""
if AUTO_DISCOVERED_MODEL_DIR="$(latest_matching_dir "$DEFAULT_MODEL_GLOB" 2>/dev/null)"; then
  :
else
  AUTO_DISCOVERED_MODEL_DIR=""
fi

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${LOCAL_SFT_MODEL_DIR:-${VARIANT_MODEL_ENV:-$AUTO_DISCOVERED_MODEL_DIR}}}"
if [[ -z "$MODEL_NAME_OR_PATH" ]]; then
  echo "No offline SFT checkpoint found for $VARIANT." >&2
  echo "Set MODEL_NAME_OR_PATH or the variant-specific *_SFT_DIR environment variable." >&2
  exit 1
fi

SOURCE_MODEL_ID="${SOURCE_MODEL_ID:-$MODEL_NAME_OR_PATH}"
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
MARGIN_LOG_PATH="$RUN_DIR/margin_logs"

require_command accelerate
require_command python

if ! is_local_model_complete "$MODEL_NAME_OR_PATH"; then
  echo "Offline model is missing required files or weights: $MODEL_NAME_OR_PATH" >&2
  exit 1
fi

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

echo "Starting offline $VARIANT Margin-DPO run: $RUN"
echo "Config: $CONFIG_PATH"
echo "Local SFT checkpoint: $MODEL_NAME_OR_PATH"
echo "Output dir: $RUN_DIR"
echo "Hub repo: $HF_REPO_ID"
echo "Skip upload: $SKIP_UPLOAD"

accelerate launch \
  --config_file accelerate_configs/fsdp.yaml \
  scripts/run_margin_dpo.py \
  "$CONFIG_PATH" \
  --model_name_or_path="$MODEL_NAME_OR_PATH" \
  --output_dir="$RUN_DIR" \
  --run_name="$RUN" \
  --hf_cache_dir="$HF_DATASET_CACHE_DIR" \
  --tokenized_dataset_cache_dir="$TOKENIZED_CACHE_DIR" \
  --margin_log_path="$MARGIN_LOG_PATH" \
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

require_hf_cli
validate_hf_auth

UPLOAD_ARGS=(
  python tools/validate_and_upload.py
  --checkpoint-dir "$RUN_DIR"
  --repo-id "$HF_REPO_ID"
  --commit-message "Upload validated Margin-DPO checkpoint"
)

if [[ "$DELETE_STALE_REMOTE_ARTIFACTS" == "1" ]]; then
  UPLOAD_ARGS+=(--delete-stale-remote-artifacts)
fi

"${UPLOAD_ARGS[@]}"

echo "Completed offline training, validation, and upload for $RUN"
