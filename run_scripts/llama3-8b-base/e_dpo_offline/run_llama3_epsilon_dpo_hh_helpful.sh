#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_llama3_epsilon_dpo.sh" hh-helpful "$@"
