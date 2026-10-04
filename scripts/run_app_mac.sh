#!/usr/bin/env bash
# Start the Streamlit app on macOS:  bash scripts/run_app_mac.sh
# Force the CPU if the Apple GPU misbehaves:  FEATURESCALPEL_DEVICE=cpu bash scripts/run_app_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || { echo "Run scripts/setup_mac.sh first."; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate
exec streamlit run experiment_app.py "$@"
