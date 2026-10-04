#!/usr/bin/env bash
# One-time setup on macOS (Apple Silicon). Run from anywhere:  bash scripts/setup_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "No python3 found. Install Python 3.11-3.13 (https://www.python.org or 'brew install python@3.12')."; exit 1; }
echo "Using $($PY --version) at $(command -v $PY)"

if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r mac_requirements.txt

echo
echo "Device the code will use:"
python - <<'PYEOF'
import torch
from src.sae_utils import get_default_device
from src.device_utils import describe_device
d = get_default_device()
print("  torch", torch.__version__, "->", describe_device(d))
if d == "cpu":
    print("  (no Apple GPU found: everything will run on the CPU, which works but is slower)")
PYEOF

echo
echo "Next:"
echo "  1) bash scripts/get_data_mac.sh        # downloads the CounterFact data the test tools need (not needed for the Prototype lab)"
echo "  2) python tools/check_mac_parity.py    # checks this Mac reproduces the saved results"
echo "  3) bash scripts/run_app_mac.sh         # starts the app at http://localhost:8501"
