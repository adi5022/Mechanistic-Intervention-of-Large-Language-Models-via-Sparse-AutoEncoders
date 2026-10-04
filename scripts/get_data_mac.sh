#!/usr/bin/env bash
# Download and build the data that is not stored in git:  bash scripts/get_data_mac.sh
# Needed by the test tools (tools/test_generalisation.py, test_generation_quality.py, compare_sweep_vs_gradient.py, ...).
# The Prototype lab in the app does not need it.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || { echo "Run scripts/setup_mac.sh first."; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate

URL="https://rome.baulab.info/data/dsets/counterfact.json"
SHA="d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
HARD_SHA="657b475e3cc37c7fb7c24a2c5c9d376791c70370cc894dca5ac3db31a3526ec8"
mkdir -p datasets data

if [ ! -f datasets/counterfact.json ]; then
  echo "Downloading counterfact.json (45 MB) from $URL"
  curl -L --fail -o datasets/counterfact.json "$URL"
fi
GOT=$(shasum -a 256 datasets/counterfact.json | cut -d' ' -f1)
if [ "$GOT" != "$SHA" ]; then echo "WARNING: counterfact.json SHA-256 is $GOT, expected $SHA"; else echo "counterfact.json checksum OK"; fi

if [ ! -f data/counterfact_hard_set.json ]; then
  echo "Building data/counterfact_hard_set.json (about 1 minute)"
  python tools/build_counterfact_set.py
fi
GOT=$(shasum -a 256 data/counterfact_hard_set.json | cut -d' ' -f1)
if [ "$GOT" != "$HARD_SHA" ]; then
  echo "NOTE: counterfact_hard_set.json SHA-256 is $GOT, expected $HARD_SHA."
  echo "      A different checksum can be harmless (floating-point differences between machines can change a borderline rank), but"
  echo "      it means the prompt list is not byte-identical to the Windows one. Check data/counterfact_hard_set_summary.json."
else
  echo "counterfact_hard_set.json checksum OK"
fi
